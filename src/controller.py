import numpy as np
import time

class GantryController:
    def __init__(self):
        # State Machine
        self.state = "IDLE"
        
        # Config
        self.home_x = -1.8
        self.rail_limit = 2.0
        self.conveyor_speed = 1.0
        
        # MOTION PROFILING CONFIG (S-Curve)
        self.max_accel = 4.0      
        self.max_accel_rot = 10.0 
        self.max_jerk = 50.0      
        self.max_jerk_rot = 100.0 
        
        # GAINS (PD Control)
        self.kp = 7.5      # Gas Pedal
        self.kd = 0.5       # Brake Pedal (Shock Absorber)
        
        self.kp_rot = 5.0
        self.kd_rot = 0.2
        
        # Memory
        self.target_yaw = 0.0
        
        # Memory for PD Control (Derivative Term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        self.prev_error_yaw = 0.0
        
        # Ramper State
        self.last_time = time.time()
        self.current_vel = np.array([0.0, 0.0, 0.0, 0.0]) 
        self.current_accel = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _ramp_scurve(self, v_target, v_curr, a_curr, max_a, max_j, dt):
        """
        Generates S-Curve profile by limiting Jerk (da/dt) and Accel (dv/dt).
        """
        if dt <= 0.0: return v_curr, a_curr

        # 1. Ideal Accel
        a_req = (v_target - v_curr) / dt
        
        # 2. Hard Limit Accel
        if abs(a_req) > max_a:
            a_req = np.sign(a_req) * max_a
            
        # 3. Limit Jerk
        jerk_req = (a_req - a_curr) / dt
        
        if abs(jerk_req) > max_j:
            jerk_effective = np.sign(jerk_req) * max_j
        else:
            jerk_effective = jerk_req
            
        # 4. Integrate
        a_next = a_curr + (jerk_effective * dt)
        
        if abs(a_next) > max_a:
            a_next = np.sign(a_next) * max_a
            
        v_next = v_curr + (a_next * dt)
        return v_next, a_next

    def update(self, img, joints, cheat_data):
        # 0. Time Delta
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.1 or dt <= 0.0: dt = 0.0 

        # 1. Perception
        mw_x = cheat_data['mw_x']
        mw_y = cheat_data['mw_y']
        mw_yaw = cheat_data['mw_yaw']
        box_x = cheat_data['box_x']
        box_y = cheat_data['box_y']
        box_yaw = cheat_data['box_yaw']
        j_x, j_y, j_z, j_yaw = joints
        
        # 2. Define Targets (State Machine sets these)
        # Default to holding current position
        t_x, t_y, t_z, t_yaw = j_x, j_y, j_z, j_yaw
        ff_vx = 0.0 # Feedforward Velocity
        gripper = "OPEN"

        # --- LOGIC START ---
        if self.state == "IDLE":
            t_y = 0.0
            t_x = self.home_x
            t_z = 0.6 
            t_yaw = 0.0
            
            if mw_x > -2.0 and mw_x < -1.0:
                self.state = "TRACKING"
                self.target_yaw = mw_yaw

        elif self.state == "TRACKING":
            t_y = max(-self.rail_limit, min(self.rail_limit, mw_y))
            t_x = mw_x
            t_yaw = self.target_yaw
            ff_vx = self.conveyor_speed # Critical for tracking!
            
            if abs(mw_y - j_y) < 0.05 and abs(mw_x - j_x) < 0.02 and abs(self.target_yaw - j_yaw) < 0.1:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            t_y = max(-self.rail_limit, min(self.rail_limit, mw_y))
            t_x = mw_x
            t_yaw = self.target_yaw
            ff_vx = self.conveyor_speed
            t_z = -0.20 # Tuned height
            
            if abs(t_z - j_z) < 0.02:
                self.state = "GRASP"
                
        elif self.state == "GRASP":
            t_y = mw_y
            t_x = mw_x
            t_yaw = self.target_yaw
            ff_vx = self.conveyor_speed
            t_z = j_z # Hold height
            gripper = "CLOSE"
            self.state = "RETRACT"

        elif self.state == "RETRACT":
            t_x = j_x # Don't fight X position, just apply FF
            ff_vx = self.conveyor_speed 
            t_z = 0.5
            t_yaw = self.target_yaw
            gripper = "CLOSE"
            
            if j_z > 0.45:
                self.state = "APPROACH_BOX"

        elif self.state == "APPROACH_BOX":
            t_x = box_x
            t_y = box_y
            t_yaw = box_yaw
            ff_vx = self.conveyor_speed
            t_z = 0.5
            gripper = "CLOSE"
            
            if abs(box_x - j_x) < 0.05 and abs(box_y - j_y) < 0.05 and abs(box_yaw - j_yaw) < 0.1:
                self.state = "INSERT"

        elif self.state == "INSERT":
            t_x = box_x
            t_y = box_y
            t_yaw = box_yaw
            ff_vx = self.conveyor_speed
            t_z = -0.10 # Tuned drop height
            gripper = "CLOSE"
            
            if abs(t_z - j_z) < 0.05:
                self.state = "RELEASE_IN_BOX"

        elif self.state == "RELEASE_IN_BOX":
            # 1. Open Gripper
            t_x = box_x
            t_y = box_y
            t_yaw = box_yaw
            # Stay down (-0.10) for one cycle to ensure fingers clear the handle/object
            t_z = -0.10 
            ff_vx = self.conveyor_speed
            gripper = "OPEN"
            
            # Move to clearance phase
            self.state = "CLEAR_BOX"

        elif self.state == "CLEAR_BOX":
            # 2. Go Up (Z) while tracking Forward (X)
            # We track box_x to move "vertically" relative to the moving box
            t_x = box_x 
            t_y = box_y
            # Maintain yaw so we don't rotate into the box walls while lifting
            t_yaw = box_yaw 
            ff_vx = self.conveyor_speed
            
            t_z = 0.6 # Target: Safe High Position
            gripper = "OPEN"
            
            # Check if we are high enough to go home safely
            if abs(t_z - j_z) < 0.05:
                self.target_yaw = 0.0 # Reset Yaw for homing
                self.state = "IDLE"
                print(">> CONTROLLER: Box Packed & Cleared!")
        # --- LOGIC END ---

        # 3. PD CONTROL CALCULATION
        # Calculate Errors
        err_x = t_x - j_x
        err_y = t_y - j_y
        err_z = t_z - j_z
        err_yaw = t_yaw - j_yaw
        
        # Calculate Derivatives (dE/dt)
        if dt > 0:
            d_x = (err_x - self.prev_error_x) / dt
            d_y = (err_y - self.prev_error_y) / dt
            d_z = (err_z - self.prev_error_z) / dt
            d_yaw = (err_yaw - self.prev_error_yaw) / dt
        else:
            d_x = d_y = d_z = d_yaw = 0.0

        # Save for next loop
        self.prev_error_x = err_x
        self.prev_error_y = err_y
        self.prev_error_z = err_z
        self.prev_error_yaw = err_yaw
        
        # Calculate Raw Velocity (P + D + FF)
        vx_req = (self.kp * err_x) + (self.kd * d_x) + ff_vx
        vy_req = (self.kp * err_y) + (self.kd * d_y)
        vz_req = (self.kp * err_z) + (self.kd * d_z)
        vyaw_req = (self.kp_rot * err_yaw) + (self.kd_rot * d_yaw)

        # 4. S-Curve Profiling (Ramp the Raw Commands)
        vx_cur, vy_cur, vz_cur, vyaw_cur = self.current_vel
        ax_cur, ay_cur, az_cur, ayaw_cur = self.current_accel
        
        vx, ax = self._ramp_scurve(vx_req, vx_cur, ax_cur, self.max_accel, self.max_jerk, dt)
        vy, ay = self._ramp_scurve(vy_req, vy_cur, ay_cur, self.max_accel, self.max_jerk, dt)
        vz, az = self._ramp_scurve(vz_req, vz_cur, az_cur, self.max_accel, self.max_jerk, dt)
        vyaw, ayaw = self._ramp_scurve(vyaw_req, vyaw_cur, ayaw_cur, self.max_accel_rot, self.max_jerk_rot, dt)
        
        self.current_vel = np.array([vx, vy, vz, vyaw])
        self.current_accel = np.array([ax, ay, az, ayaw])

        return [vx, vy, vz, vyaw], gripper