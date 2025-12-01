import pybullet as p
import pybullet_data
import numpy as np
import random
import os

class GantrySim:
    def __init__(self, render=True):
        # 1. Setup Physics
        connection_mode = p.GUI if render else p.DIRECT
        p.connect(connection_mode)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(3.5, 90, -30, [0,0,0.5])
        
        # CRITICAL: This line finds plane.urdf in the installed library
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 2. Build World
        self._create_env()
        self._spawn_robot()
        self._spawn_box()
        
        # 3. State Variables
        self.conveyor_speed = 1.5
        self.grasp_constraint = None
        self.dt = 1.0/240.0

    def _create_env(self):
        # Load Plane
        p.loadURDF("plane.urdf")

        # 1. Conveyor Table (Long along X)
        # Dimensions: [Length/2 (X), Width/2 (Y), Height/2 (Z)]
        belt_h = 0.2
        belt_l = 10.0 # Total length 10m
        belt_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[belt_l/2, 0.4, belt_h/2], rgbaColor=[0, 0.2, 0.8, 1])
        belt_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[belt_l/2, 0.4, belt_h/2])
        p.createMultiBody(0, belt_col, belt_visual, [0.0, 0, belt_h/2])
        self.belt_surface_z = belt_h

        # 2. Static Frame (Aligned with X-Axis)
        beam_h = 1.4
        rail_l = 4.2 # Rail length 4.2m
        
        # Shapes
        leg_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.06, length=beam_h, rgbaColor=[0.5, 0.5, 0.5, 1])
        # Rail: Long in X [2.1, 0.05, 0.05]
        rail_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[rail_l/2, 0.05, 0.05], rgbaColor=[0.4, 0.4, 0.4, 1])
        
        # Frame Dimensions
        frame_width_y = 0.7  # Distance from center to side rail (Y)
        leg_spacing_x = 2.0  # Distance from center to leg (X)
        
        # Build the two sides of the frame
        for y_side in [-frame_width_y, frame_width_y]: 
            # A. Create the Long Rail running along X at this Y position
            p.createMultiBody(baseVisualShapeIndex=rail_shape, basePosition=[0, y_side, beam_h + 0.05])
            
            # B. Create the two Legs supporting this rail (Front and Back in X)
            for x_pos in [-leg_spacing_x, leg_spacing_x]:
                p.createMultiBody(baseVisualShapeIndex=leg_shape, basePosition=[x_pos, y_side, beam_h/2])
            
        self.robot_base_z = beam_h + 0.15

    def _spawn_robot(self):
        # UPDATED: Look in assets folder first
        if os.path.exists("assets/gantry.urdf"):
            path = "assets/gantry.urdf"
        else:
            path = "gantry.urdf" # Fallback if user didn't move it
            
        self.robot_id = p.loadURDF(path, [0, 0, self.robot_base_z], useFixedBase=True)

    def _spawn_box(self):
        box_dims = [0.15, 0.25, 0.15]
        box_z = self.belt_surface_z + 0.15
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_dims, rgbaColor=[0.1, 0.1, 0.1, 1])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_dims)
        
        self.box_state = {
            'x': -3.5,
            'y': random.uniform(-0.2, 0.25),
            'yaw': random.uniform(-0.7, 0.7),
            'z': box_z
        }
        
        orn = p.getQuaternionFromEuler([0, 0, self.box_state['yaw']])
        self.box_id = p.createMultiBody(0.5, col, vis, [self.box_state['x'], self.box_state['y'], box_z], orn)

    def step(self, cmd_vel, gripper_cmd):
        # 1. Apply Motor Commands
        p.setJointMotorControl2(self.robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[0], force=500)
        p.setJointMotorControl2(self.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[1], force=500)
        p.setJointMotorControl2(self.robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[2], force=500)
        p.setJointMotorControl2(self.robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[3], force=200)
        
        # 2. Apply Gripper
        target_finger = 0.05 if gripper_cmd == "CLOSE" else 0.0
        p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=target_finger, force=100)
        p.setJointMotorControl2(self.robot_id, 5, p.POSITION_CONTROL, targetPosition=target_finger, force=100)
        
        # 3. Physics Logic
        self._update_physics(gripper_cmd)
        
        p.stepSimulation()

    def _update_physics(self, gripper_cmd):
        # MAGIC GRIP
        if gripper_cmd == "CLOSE" and self.grasp_constraint is None:
            ee_pos = p.getLinkState(self.robot_id, 3)[0]
            box_pos = p.getBasePositionAndOrientation(self.box_id)[0]
            if np.linalg.norm(np.array(ee_pos) - np.array(box_pos)) < 0.5:
                self.grasp_constraint = p.createConstraint(self.robot_id, 3, self.box_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,-0.2], [0,0,0])
                print(">> SIM: Magic Grip Activated")
                
        elif gripper_cmd == "OPEN" and self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            print(">> SIM: Released")
            pos, orn = p.getBasePositionAndOrientation(self.box_id)
            self.box_state['x'] = pos[0] 
            self.box_state['y'] = pos[1]
            self.box_state['yaw'] = 0.0
            
        # CONVEYOR
        if self.grasp_constraint is None:
            self.box_state['x'] += self.conveyor_speed * self.dt
            if self.box_state['x'] > 5.0:
                self.box_state['x'] = -2.5
                self.box_state['y'] = random.uniform(-0.25, 0.25)
                self.box_state['yaw'] = random.uniform(-0.7, 0.7)
                print(f">> SIM: New Box Spawned | X: {self.box_state['x']:.2f}")

            orn = p.getQuaternionFromEuler([0, 0, self.box_state['yaw']])
            p.resetBasePositionAndOrientation(self.box_id, [self.box_state['x'], self.box_state['y'], self.box_state['z']], orn)

    def get_data(self):
        # Camera
        ee_state = p.getLinkState(self.robot_id, 3)
        cam_pos, cam_orn = ee_state[0], ee_state[1]
        rot_matrix = p.getMatrixFromQuaternion(cam_orn)
        up_vec = [rot_matrix[1], rot_matrix[4], rot_matrix[7]] 
        
        view_mat = p.computeViewMatrix(cam_pos, [cam_pos[0], cam_pos[1], 0], up_vec)
        proj_mat = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 4.0)
        w, h, rgb, _, _ = p.getCameraImage(240, 240, view_mat, proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # Use .copy() to prevent OpenCV errors
        img = np.reshape(np.array(rgb), (240, 240, 4)).astype(np.uint8)[:, :, :3].copy()
        
        # Joints
        joints = [p.getJointState(self.robot_id, i)[0] for i in range(4)]
        
        # Cheat Data
        box_pos, box_orn = p.getBasePositionAndOrientation(self.box_id)
        box_yaw = p.getEulerFromQuaternion(box_orn)[2]
        cheat_data = {'box_x': box_pos[0], 'box_y': box_pos[1], 'box_yaw': box_yaw}
        
        return img, joints, cheat_data