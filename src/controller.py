import numpy as np

class GantryController:
    def __init__(self):
        # State Machine
        self.state = "IDLE"
        
        # Config
        self.home_x = -1.8
        self.rail_limit = 2.0
        self.conveyor_speed = 1.5
        
        # Gains
        self.kp = 8.0
        self.kp_rot = 5.0
        
        # Memory
        self.target_yaw = 0.0
    
    def process_vision(self, img, cheat_data):
        """
        TODO: PARTNER WILL IMPLEMENT OPENCV HERE.
        For now, returns Ground Truth from simulation.
        """
        return cheat_data['box_x'], cheat_data['box_y'], cheat_data['box_yaw']

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
        j_x, j_y, j_z, j_yaw = joints
        
        # 2. Init Outputs
        vx, vy, vz, vyaw = 0, 0, 0, 0
        gripper = "OPEN"

        # 3. Logic
        if self.state == "IDLE":
            vy = self.kp * (0.0 - j_y)
            vx = self.kp * (self.home_x - j_x)
            vz = self.kp * (0.6 - j_z) # Home High (0.6 limit)
            vyaw = self.kp_rot * (0.0 - j_yaw)
            
            if box_x > -2.0 and box_x < -1.0:
                self.state = "TRACKING"
                self.target_yaw = box_yaw

        elif self.state == "TRACKING":
            # Clamp Y to rails
            safe_y = max(-self.rail_limit, min(self.rail_limit, box_y))
            
            vy = self.kp * (safe_y - j_y) 
            vx = self.kp * (box_x - j_x) + self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            
            # Check Alignment (Pos + Rot)
            if abs(box_y - j_y) < 0.05 and abs(box_x - j_x) < 0.02 and abs(self.target_yaw - j_yaw) < 0.1:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            safe_y = max(-self.rail_limit, min(self.rail_limit, box_y))
            
            vy = self.kp * (safe_y - j_y) 
            vx = self.kp * (box_x - j_x)+ self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            
            target_z = -0.25
            vz = self.kp * (target_z - j_z)
            
            if abs(target_z - j_z) < 0.02:
                self.state = "GRASP"
                
        elif self.state == "GRASP":
            vy = self.kp * (box_y - j_y) 
            vx = self.kp * (box_x - j_x) + self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            vz = 0
            gripper = "CLOSE"
            
            self.state = "RETRACT"

        elif self.state == "RETRACT":
            vx = 0.0 # Stop tracking conveyor
            vz = self.kp * (0.0 - j_z)
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            gripper = "CLOSE"
            
            if j_z > -0.1:
                self.state = "CARRY"

        elif self.state == "CARRY":
            vx = self.kp * (2.0 - j_x) # End of line
            vy = self.kp * (0.0 - j_y) # Center X
            vyaw = self.kp_rot * (0.0 - j_yaw) # Straighten Yaw
            vz = self.kp * (0.0 - j_z)
            gripper = "CLOSE"
            
            if abs(2.0 - j_x) < 0.1:
                self.state = "RELEASE"

        elif self.state == "RELEASE":
            gripper = "OPEN"
            # Reset logic
            self.target_yaw = 0.0
            self.state = "IDLE"
            print(">> CONTROLLER: Cycle Complete")

        return [vx, vy, vz, vyaw], gripper