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
        # if render:
        #     p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "output.mp4") 

        # 2. Build World
        self._create_env()
        self._spawn_robot()
        self._spawn_objects()
        
        # 3. State Variables
        self.conveyor_speed = 1.0
        self.grasp_constraint = None
        self.grasped_body = None  # Track which ID is held
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

    def _spawn_objects(self):
        # 1. Spawn Microwave
        mw_dims = [0.15, 0.25, 0.15]
        mw_z = self.belt_surface_z + 0.15
        mw_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=mw_dims, rgbaColor=[0.1, 0.1, 0.1, 1])
        mw_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=mw_dims)
        
        self.mw_state = {
            'x': -3.5,
            'y': random.uniform(-0.2, 0.25),
            'yaw': random.uniform(-0.7, 0.7),
            'z': mw_z
        }
        mw_orn = p.getQuaternionFromEuler([0, 0, self.mw_state['yaw']])
        self.mw_id = p.createMultiBody(0.5, mw_col, mw_vis, [self.mw_state['x'], self.mw_state['y'], mw_z], mw_orn)

        # 2. Spawn Box (Target)
        box_z = self.belt_surface_z + 0.01
        
        # State: X is relative to MW, but Y and Yaw are independent
        self.box_state = {
            'x': self.mw_state['x'] + 0.8,    # 0.8m in front of MW
            'y': random.uniform(-0.2, 0.25),
            'yaw': random.uniform(-0.7, 0.7),
            'z': box_z
        }
        
        box_orn = p.getQuaternionFromEuler([0, 0, self.box_state['yaw']])
        
        # Load from ASSETS folder
        if os.path.exists("assets/box.urdf"):
            box_path = "assets/box.urdf"
        else:
            box_path = "box.urdf" # Fallback
            
        self.box_id = p.loadURDF(box_path, [self.box_state['x'], self.box_state['y'], box_z], box_orn)

    def step(self, cmd_vel, gripper_cmd):
        # 1. Apply Motor Commands
        p.setJointMotorControl2(self.robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[0], force=500)
        p.setJointMotorControl2(self.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[1], force=500)
        p.setJointMotorControl2(self.robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[2], force=500)
        p.setJointMotorControl2(self.robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=cmd_vel[3], force=200)
        
        # 2. Apply Gripper
        target_finger = 0.03 if gripper_cmd == "CLOSE" else 0.0
        p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=target_finger, force=100)
        p.setJointMotorControl2(self.robot_id, 5, p.POSITION_CONTROL, targetPosition=target_finger, force=100)
        
        # 3. Physics Logic
        self._update_physics(gripper_cmd)
        
        p.stepSimulation()

    def _update_physics(self, gripper_cmd):
        # --- 1. GRASPING LOGIC (Same as before) ---
        if gripper_cmd == "CLOSE" and self.grasp_constraint is None:
            ee_pos = p.getLinkState(self.robot_id, 3)[0]
            
            # Check Box
            box_pos = p.getBasePositionAndOrientation(self.box_id)[0]
            if np.linalg.norm(np.array(ee_pos) - np.array(box_pos)) < 0.5:
                self.grasp_constraint = p.createConstraint(self.robot_id, 3, self.box_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,-0.2], [0,0,0])
                self.grasped_body = self.box_id
                print(">> SIM: Magic Grip (Box)")
            
            # Check MW
            elif np.linalg.norm(np.array(ee_pos) - np.array(p.getBasePositionAndOrientation(self.mw_id)[0])) < 0.5:
                self.grasp_constraint = p.createConstraint(self.robot_id, 3, self.mw_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,-0.2], [0,0,0])
                self.grasped_body = self.mw_id
                print(">> SIM: Magic Grip (MW)")

        elif gripper_cmd == "OPEN" and self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            self.grasped_body = None
            print(">> SIM: Released")

        # --- 2. PHYSICS & CONVEYOR LOGIC (NEW) ---
        # Instead of teleporting, we trust the physics engine and apply velocity.
        
        # A. Update Python State FROM PyBullet (Truth is now in the Sim)
        mw_pos, mw_orn = p.getBasePositionAndOrientation(self.mw_id)
        self.mw_state['x'], self.mw_state['y'], self.mw_state['z'] = mw_pos
        self.mw_state['yaw'] = p.getEulerFromQuaternion(mw_orn)[2]

        box_pos, box_orn = p.getBasePositionAndOrientation(self.box_id)
        self.box_state['x'], self.box_state['y'], self.box_state['z'] = box_pos
        self.box_state['yaw'] = p.getEulerFromQuaternion(box_orn)[2]

        # B. Apply Conveyor Velocity to BOX (Always moving)
        # We use resetBaseVelocity so it pushes things but respects collisions
        p.resetBaseVelocity(self.box_id, [self.conveyor_speed, 0, 0])

        # C. Apply Conveyor Velocity to MW (Conditional)
        if self.grasped_body != self.mw_id:
            # If it's low (on the belt), drive it.
            # If it's high (falling), let gravity do the work.
            if mw_pos[2] < 0.4: # Belt surface is ~0.2
                p.resetBaseVelocity(self.mw_id, [self.conveyor_speed, 0, 0])
            # Else: Do nothing, let it free fall!

        # --- 3. RESPAWN LOGIC ---
        if self.mw_state['x'] > 5.0 and self.grasped_body != self.mw_id:
            # Reset MW
            self.mw_state['x'] = -2.5
            self.mw_state['y'] = random.uniform(-0.25, 0.25)
            self.mw_state['yaw'] = random.uniform(-0.7, 0.7)
            
            # Reset Box
            self.box_state['x'] = self.mw_state['x'] + 0.8
            self.box_state['y'] = random.uniform(-0.25, 0.25)
            self.box_state['yaw'] = random.uniform(-0.7, 0.7)
            
            # Force Teleport (Only on Respawn)
            mw_orn = p.getQuaternionFromEuler([0, 0, self.mw_state['yaw']])
            p.resetBasePositionAndOrientation(self.mw_id, [self.mw_state['x'], self.mw_state['y'], self.mw_state['z']], mw_orn)
            
            box_orn = p.getQuaternionFromEuler([0, 0, self.box_state['yaw']])
            p.resetBasePositionAndOrientation(self.box_id, [self.box_state['x'], self.box_state['y'], self.box_state['z']], box_orn)
            
            print(f">> SIM: Cycle Reset")

    def _project_point(self, point_3d, view_mat, proj_mat, width, height):
        """
        Projects a 3D world point (x,y,z) into 2D pixel coordinates (u,v).
        """
        # 1. View Transform (World -> Camera)
        # PyBullet matrices are flat column-major lists. Reshape carefully.
        vm = np.array(view_mat).reshape((4, 4), order='F')
        pm = np.array(proj_mat).reshape((4, 4), order='F')
        
        # Homogeneous Coordinates [x, y, z, 1]
        vec = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        
        # Camera Space
        cam_pos = np.dot(vm, vec)
        
        # 2. Projection Transform (Camera -> Clip)
        clip_pos = np.dot(pm, cam_pos)
        
        # 3. Perspective Divide (Clip -> NDC)
        # Avoid divide by zero
        if clip_pos[3] == 0: return 0, 0
        ndc = clip_pos[:3] / clip_pos[3]
        
        # 4. Viewport Transform (NDC -> Pixels)
        # NDC x,y range is [-1, 1]. Map to [0, width] and [0, height]
        # Note: Image Y axis is usually inverted (top-down) vs OpenGL (bottom-up)
        u = (ndc[0] + 1) * (width / 2)
        v = (1 - ndc[1]) * (height / 2)
        
        return int(u), int(v)

    def get_data(self):
        # --- Camera Setup (Existing) ---
        ee_state = p.getLinkState(self.robot_id, 3)
        cam_pos, cam_orn = ee_state[0], ee_state[1]
        rot_matrix = p.getMatrixFromQuaternion(cam_orn)
        up_vec = [rot_matrix[1], rot_matrix[4], rot_matrix[7]] 
        
        view_mat = p.computeViewMatrix(cam_pos, [cam_pos[0], cam_pos[1], 0], up_vec)
        proj_mat = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 4.0)
        w, h, rgb, _, _ = p.getCameraImage(240, 240, view_mat, proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        img = np.reshape(np.array(rgb), (240, 240, 4)).astype(np.uint8)[:, :, :3].copy()
        
        # --- Joints (Existing) ---
        joints = [p.getJointState(self.robot_id, i)[0] for i in range(4)]
        
        # --- Data Extraction (Modified) ---
        mw_pos, mw_orn = p.getBasePositionAndOrientation(self.mw_id)
        mw_yaw = p.getEulerFromQuaternion(mw_orn)[2]
        
        box_pos, box_orn = p.getBasePositionAndOrientation(self.box_id)
        box_yaw = p.getEulerFromQuaternion(box_orn)[2]
        
        # NEW: Project 3D positions to 2D Pixels
        mw_u, mw_v = self._project_point(mw_pos, view_mat, proj_mat, 240, 240)
        box_u, box_v = self._project_point(box_pos, view_mat, proj_mat, 240, 240)
        
        cheat_data = {
            # World Data (Legacy/Debug)
            'mw_x': mw_pos[0], 'mw_y': mw_pos[1], 'mw_yaw': mw_yaw,
            'box_x': box_pos[0], 'box_y': box_pos[1], 'box_yaw': box_yaw,
            
            # Pixel Data (The "Virtual Eye")
            'mw_u': mw_u, 'mw_v': mw_v,
            'box_u': box_u, 'box_v': box_v
        }
        
        return img, joints, cheat_data