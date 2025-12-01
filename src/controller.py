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
        self.kp = 10.0
        self.kp_rot = 7.0
        
        # Memory
        self.target_yaw = 0.0
    
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
        mw_x = cheat_data['mw_x']
        mw_y = cheat_data['mw_y']
        mw_yaw = cheat_data['mw_yaw']
        box_x = cheat_data['box_x']
        box_y = cheat_data['box_y']
        box_yaw = cheat_data['box_yaw']
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
            
            if mw_x > -2.0 and mw_x < -1.0:
                self.state = "TRACKING"
                self.target_yaw = mw_yaw

        elif self.state == "TRACKING":
            # Clamp Y to rails
            safe_y = max(-self.rail_limit, min(self.rail_limit, mw_y))
            
            vy = self.kp * (safe_y - j_y) 
            vx = self.kp * (mw_x - j_x) + self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            
            # Check Alignment (Pos + Rot)
            if abs(mw_y - j_y) < 0.05 and abs(mw_x - j_x) < 0.02 and abs(self.target_yaw - j_yaw) < 0.1:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            safe_y = max(-self.rail_limit, min(self.rail_limit, mw_y))
            
            vy = self.kp * (safe_y - j_y) 
            vx = self.kp * (mw_x - j_x)+ self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            
            target_z = -0.25
            vz = self.kp * (target_z - j_z)
            
            if abs(target_z - j_z) < 0.02:
                self.state = "GRASP"
                
        elif self.state == "GRASP":
            vy = self.kp * (mw_y - j_y) 
            vx = self.kp * (mw_x - j_x) + self.conveyor_speed
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            vz = 0
            gripper = "CLOSE"
            
            self.state = "RETRACT"

        elif self.state == "RETRACT":
            vx = 0.0 # Stop tracking conveyor
            vz = self.kp * (0.5 - j_z)
            vyaw = self.kp_rot * (self.target_yaw - j_yaw)
            gripper = "CLOSE"
            
            # Wait for clearance height
            if j_z > 0.45:
                self.state = "APPROACH_BOX"

        elif self.state == "APPROACH_BOX":
            # CHANGE: Hover over the moving box
            vx = self.kp * (box_x - j_x) + self.conveyor_speed
            vy = self.kp * (box_y - j_y)
            vyaw = self.kp_rot * (box_yaw - j_yaw) # Match box rotation
            vz = self.kp * (0.5 - j_z) # Stay high
            gripper = "CLOSE"
            
            # Check alignment with box
            if abs(box_x - j_x) < 0.05 and abs(box_y - j_y) < 0.05 and abs(box_yaw - j_yaw) < 0.1:
                self.state = "INSERT"

        elif self.state == "INSERT":
            # CHANGE: Lower into box while tracking
            vx = self.kp * (box_x - j_x) + self.conveyor_speed
            vy = self.kp * (box_y - j_y)
            vyaw = self.kp_rot * (box_yaw - j_yaw)
            
            # Target Z=0.0 places it inside the box
            target_z = 0.0 
            vz = self.kp * (target_z - j_z)
            gripper = "CLOSE"
            
            if abs(target_z - j_z) < 0.05:
                self.state = "RELEASE_IN_BOX"

        elif self.state == "RELEASE_IN_BOX":
            # CHANGE: Release and drift
            vx = self.conveyor_speed
            vy = 0; vz = 0; vyaw = 0
            gripper = "OPEN"
            
            # Simple 1-step exit or add a timer if needed
            self.target_yaw = 0.0
            self.state = "IDLE"
            print(">> CONTROLLER: Box Packed!")

        return [vx, vy, vz, vyaw], gripper