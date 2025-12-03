import numpy as np
import os
import csv
from src.vision import VisionPipeline
from src.vision2 import VisionPipeline as VisionPipeline2 #this is using minRect method
import cv2

import pybullet as p


class GantryController:
    def __init__(self):
        # State Machine
        self.state = "IDLE"
        
        # Config
        self.home_y = -1.8
        self.rail_limit = 2.0
        self.conveyor_speed = 1.5
        
        # Gains
        self.kp = 8.0
        self.kp_rot = 5.0
        
        # Memory
        self.target_yaw = 0.0

        self.vision = VisionPipeline()
        self.vision2 = VisionPipeline2()
        self.counter = 0
        
        self.box_plane_z = 0.2 + 0.15  # belt_h + box_height/2  = 0.35

        # CSV file for logging
        self.log_file = "vision_vs_cheat_log.csv"

        # If the file doesn't exist, create and write header
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame",
                    "cheat_x", "cheat_y", "cheat_yaw",
                    "vision_cx", "vision_cy", "vision_yaw"
                ])
    

    def pixel_to_world_on_plane(self, u, v, cam_pos, cam_orn,
                                img_w=240, img_h=240, fov_deg=60.0):
        """
        Map image pixel (u, v) from the wrist camera to a world (X, Y)
        by casting a ray from the camera and intersecting with plane z = box_plane_z.
        """
        
        print("Mapping pixel to world coordinates...")

        cam_pos = np.array(cam_pos, dtype=np.float64)

        # --- 1. Camera intrinsics from FOV ---
        aspect = img_w / float(img_h)
        f_y = (img_h / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        f_x = f_y * aspect
        c_x = img_w / 2.0
        c_y = img_h / 2.0

        # Pixel -> normalized camera coords
        x_cam = (u - c_x) / f_x
        y_cam = -(v - c_y) / f_y   # minus: image y down, camera y up

        # --- 2. Camera orientation: basis vectors in WORLD frame ---
        rot = p.getMatrixFromQuaternion(cam_orn)
        rot = np.array(rot).reshape(3, 3)

        # In _render_camera_from_link, you used:
        # forward = [-rot[2], -rot[5], -rot[8]]
        # up      = [ rot[1],  rot[4],  rot[7]]
        forward = np.array([-rot[0,2], -rot[1,2], -rot[2,2]], dtype=np.float64)
        up      = np.array([ rot[0,1],  rot[1,1],  rot[2,1]], dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up      /= np.linalg.norm(up)
        right   = np.cross(forward, up)
        right   /= np.linalg.norm(right)

        # --- 3. Ray direction in WORLD frame ---
        dir_world = x_cam * right + y_cam * up + 1.0 * forward
        dir_world /= np.linalg.norm(dir_world)

        # --- 4. Intersect ray with horizontal plane z = box_plane_z ---
        z_plane = self.box_plane_z
        t = (z_plane - cam_pos[2]) / dir_world[2]
        P = cam_pos + t * dir_world  # intersection point

        X_world, Y_world = float(P[0]), float(P[1])
        return X_world, Y_world
    
    def process_vision(self, img, cheat_data, cam_pos, cam_orn):
        """
        Use vision to estimate box pose, convert to world frame, 
        and still log comparison with cheat data.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.counter += 1
        frame_id = self.counter
        print(f"Processing vision frame #: {frame_id}")

        rot_deg, cx, cy = self.vision2.process_vision(img)

        rot_rad = None
        X_world = None
        Y_world = None

        if rot_deg is not None and cx is not None and cy is not None:
            rot_rad = (rot_deg * np.pi) / 180.0
            print("Feature Vector from Vision (image frame):",
                  {"yaw_img(rad)": rot_rad, "cx": cx, "cy": cy})

            # --- NEW: image → world ---
            X_world, Y_world = self.pixel_to_world_on_plane(
                cx, cy, cam_pos, cam_orn,
                img_w=img.shape[1], img_h=img.shape[0], fov_deg=60.0
            )

            # For now, use vision yaw directly as world yaw (we can add an offset later if needed)
            yaw_world = rot_rad

            print("Vision mapped to world frame:",
                  {"X_world": X_world, "Y_world": Y_world, "yaw_world": yaw_world})
        else:
            print("Vision failed to detect object.")

        # Cheat data for logging
        cheat_x = cheat_data['box_x']
        cheat_y = cheat_data['box_y']
        cheat_yaw = cheat_data['box_yaw']

        # log both
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                cheat_x, cheat_y, cheat_yaw,
                X_world, Y_world, rot_rad
            ])

        # If vision succeeded, use it; otherwise fall back to cheat data
        # if X_world is not None:
        #     return X_world, Y_world, yaw_world
        # else:
        return cheat_x, cheat_y, cheat_yaw

    # def process_vision(self, img, cheat_data):
    #     """
    #     Compare cheat data vs vision output and save to CSV.
    #     """
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    

    #     self.counter += 1
    #     frame_id = self.counter
    #     print(f"Processing vision frame #: {frame_id}")

    #     # --- Vision processing ---
    #     rot_deg, cx, cy = self.vision2.process_vision(img)

    #     if rot_deg is not None:
    #         rot_rad = (rot_deg * np.pi) / 180.0
    #         print("Feature Vector from Vision:",
    #             {"yaw(rad)": rot_rad, "cx": cx, "cy": cy})
    #     else:
    #         rot_rad, cx, cy = None, None, None
    #         print("Vision failed to detect object.")


    #     # --- Cheat data from simulation ---
    #     cheat_x = cheat_data['box_x']
    #      
    #     cheat_y = cheat_data['box_y']
    #     cheat_yaw = cheat_data['box_yaw']    # Already in radians

    #     print(f"Cheat data: x={cheat_x}, y={cheat_y}, yaw={cheat_yaw}")


    #     # --- LOG BOTH TO CSV ---
    #     with open(self.log_file, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             frame_id,
    #             cheat_x, cheat_y, cheat_yaw,
    #             cx, cy, rot_rad
    #         ])


    #     # --- Return cheat data as before ---
    #     return cheat_x, cheat_y, cheat_yaw

    # def process_vision(self, img, cheat_data):
    #     """
    #     TODO: PARTNER WILL IMPLEMENT OPENCV HERE.
    #     For now, returns Ground Truth from simulation.
    #     """
        
    #     # if self.counter == 190:
    #     #     print("Saving debug vision frame...")
    #     #     cv2.imwrite("vision_debug_frame.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 

    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    

    #     self.counter += 1
    #     print("Processing vision frame #: ", self.counter)
    #     rot, cx, cy = self.vision2.process_vision(img)
    #     print("Cheat data: ", cheat_data)
    #     if rot is not None:
    #         print("Feature Vector from Vison : ", {(rot*np.pi)/180, cx, cy})
    #         print("rot_radian from vision: ", (rot*np.pi)/180)
    #     else:
    #         print("rot_radian from vision: None")
    #     # print("vision data", {rot, x, y})
        


    #     return cheat_data['box_x'], cheat_data['box_y'], cheat_data['box_yaw']
        # return cx, cy, rot*np.pi/180 if rot is not None else None

    def update(self, img, joints, cheat_data,  cam_pos, cam_orn):
        """
        Inputs: 
            img: Camera image
            joints: [y, x, z, yaw]
            cheat_data: Ground truth dictionary
        Returns: 
            cmd_vel: [vy, vx, vz, vyaw]
            gripper_cmd: "OPEN" or "CLOSE"
        """
        # 1. Perception
        box_x, box_y, box_yaw = self.process_vision(img, cheat_data, cam_pos, cam_orn)
        print(f">> VISION OUTPUTS: box_x={box_x}, box_y={box_y}, box_yaw={box_yaw}")

        j_y, j_x, j_z, j_yaw = joints
        if box_yaw is not None:
            # 2. Init Outputs
            vy, vx, vz, vyaw = 0, 0, 0, 0
            gripper = "OPEN"

            # 3. Logic
            if self.state == "IDLE":
                vy = self.kp * (self.home_y - j_y)
                vx = self.kp * (0.0 - j_x)
                vz = self.kp * (0.6 - j_z) # Home High (0.6 limit)
                vyaw = self.kp_rot * (0.0 - j_yaw)
                
                if box_y > -2.0 and box_y < -1.0:
                    self.state = "TRACKING"
                    self.target_yaw = box_yaw

            elif self.state == "TRACKING":
                # Clamp Y to rails
                safe_y = max(-self.rail_limit, min(self.rail_limit, box_y))
                
                vy = self.kp * (safe_y - j_y) + self.conveyor_speed
                vx = self.kp * (box_x - j_x)
                vyaw = self.kp_rot * (self.target_yaw - j_yaw)
                
                # Check Alignment (Pos + Rot)
                if abs(box_y - j_y) < 0.05 and abs(box_x - j_x) < 0.02 and abs(self.target_yaw - j_yaw) < 0.1:
                    self.state = "DESCEND"

            elif self.state == "DESCEND":
                safe_y = max(-self.rail_limit, min(self.rail_limit, box_y))
                
                vy = self.kp * (safe_y - j_y) + self.conveyor_speed
                vx = self.kp * (box_x - j_x)
                vyaw = self.kp_rot * (self.target_yaw - j_yaw)
                
                target_z = -0.25
                vz = self.kp * (target_z - j_z)
                
                if abs(target_z - j_z) < 0.02:
                    self.state = "GRASP"
                    
            elif self.state == "GRASP":
                vy = self.kp * (box_y - j_y) + self.conveyor_speed
                vx = self.kp * (box_x - j_x)
                vyaw = self.kp_rot * (self.target_yaw - j_yaw)
                vz = 0
                gripper = "CLOSE"
                
                self.state = "RETRACT"

            elif self.state == "RETRACT":
                vy = 0.0 # Stop tracking conveyor
                vz = self.kp * (0.0 - j_z)
                vyaw = self.kp_rot * (self.target_yaw - j_yaw)
                gripper = "CLOSE"
                
                if j_z > -0.1:
                    self.state = "CARRY"

            elif self.state == "CARRY":
                vy = self.kp * (2.0 - j_y) # End of line
                vx = self.kp * (0.0 - j_x) # Center X
                vyaw = self.kp_rot * (0.0 - j_yaw) # Straighten Yaw
                vz = self.kp * (0.0 - j_z)
                gripper = "CLOSE"
                
                if abs(2.0 - j_y) < 0.1:
                    self.state = "RELEASE"

            elif self.state == "RELEASE":
                gripper = "OPEN"
                # Reset logic
                self.target_yaw = 0.0
                self.state = "IDLE"
                print(">> CONTROLLER: Cycle Complete")
            
            print(f">> CONTROLLER STATE: {self.state} | Commands: vy={vy:.2f}, vx={vx:.2f}, vz={vz:.2f}, vyaw={vyaw:.2f}, gripper={gripper}")
            return [vy, vx, vz, vyaw], gripper
        
        else:
            # 2a. Vision Data is NOT available (box_yaw is None).
            print(f">> WARNING: Vision failed in state {self.state}. Executing safe commands.")

            if self.state == "IDLE":
                # If IDLE, continue moving to HOME.
                vy = self.kp * (self.home_y - j_y)
                vx = self.kp * (0.0 - j_x)
                vz = self.kp * (0.6 - j_z)
                vyaw = self.kp_rot * (0.0 - j_yaw)
                gripper = "OPEN" # Safe
                
            elif self.state in ["TRACKING", "DESCEND"]:
                # If actively tracking, STOP moving *except* maybe a small conveyor speed offset
                # This prevents the robot from wildly guessing box position.
                # A safer option: Stop all motion and revert to IDLE.
                vy = 0.0 # Stop Y motion
                vx = 0.0 # Stop X motion
                vz = 0.0 # Stop Z motion
                vyaw = 0.0 # Stop rotation
                gripper = "OPEN" # Safe state gripper
                self.state = "IDLE" # Revert to safe state
                
            elif self.state == "GRASP":
                # Maintain the grasping action if currently grasping
                gripper = "CLOSE"
                self.state = "RETRACT" # Force state forward
                
            elif self.state == "RETRACT":
                # Continue retracting motion using only joint data
                vy = 0.0 # Stop Y motion
                vz = self.kp * (0.0 - j_z)
                vyaw = self.kp_rot * (self.target_yaw - j_yaw)
                gripper = "CLOSE"
                if j_z > -0.1:
                    self.state = "CARRY"

            elif self.state == "CARRY":
                # Continue carrying motion using only joint data
                vy = self.kp * (2.0 - j_y)
                vx = self.kp * (0.0 - j_x)
                vyaw = self.kp_rot * (0.0 - j_yaw)
                vz = self.kp * (0.0 - j_z)
                gripper = "CLOSE"
                if abs(2.0 - j_y) < 0.1:
                    self.state = "RELEASE"
                    
            elif self.state == "RELEASE":
                # Finish the release
                gripper = "OPEN"
                self.target_yaw = 0.0
                self.state = "IDLE"
                print(">> CONTROLLER: Cycle Complete (Vision Failed during Tracking/Descend)")


        # --- 4. Return Final Commands ---
        return [vy, vx, vz, vyaw], gripper