import numpy as np
import os
import csv
from src.vision import VisionPipeline
from src.vision2 import VisionPipeline as VisionPipeline2 #this is using minRect method
import cv2


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
    
    def process_vision(self, img, cheat_data):
        """
        Compare cheat data vs vision output and save to CSV.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    

        self.counter += 1
        frame_id = self.counter
        print(f"Processing vision frame #: {frame_id}")

        # --- Vision processing ---
        rot_deg, cx, cy = self.vision2.process_vision(img)

        if rot_deg is not None:
            rot_rad = (rot_deg * np.pi) / 180.0
            print("Feature Vector from Vision:",
                {"yaw(rad)": rot_rad, "cx": cx, "cy": cy})
        else:
            rot_rad, cx, cy = None, None, None
            print("Vision failed to detect object.")


        # --- Cheat data from simulation ---
        cheat_x = cheat_data['box_x']
        cheat_y = cheat_data['box_y']
        cheat_yaw = cheat_data['box_yaw']    # Already in radians

        print(f"Cheat data: x={cheat_x}, y={cheat_y}, yaw={cheat_yaw}")


        # --- LOG BOTH TO CSV ---
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                cheat_x, cheat_y, cheat_yaw,
                cx, cy, rot_rad
            ])


        # --- Return cheat data as before ---
        return cheat_x, cheat_y, cheat_yaw

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

    def update(self, img, joints, cheat_data):
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
        box_x, box_y, box_yaw = self.process_vision(img, cheat_data)
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