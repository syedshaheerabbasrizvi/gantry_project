import cv2
import time
import numpy as np
# Import from your packages
from src.sim_interface import GantrySim
from src.controller import GantryController

def main():
    # 1. Setup
    sim = GantrySim(render=True)
    brain = GantryController()
    
    print("Initialization Complete. Running...")
    
    # --- DIAGNOSTICS CONFIG ---
    print_interval = 1        # Print every 0.5 seconds
    last_print = time.time()
    
    # Performance Monitoring
    loop_times = []
    max_loops_to_avg = 100
    
    # State Monitoring
    last_state = "IDLE"
    state_start_time = time.time()
    
    while True:
        loop_start = time.time()
        
        # --- SENSE ---
        img, joints, cheat_data = sim.get_data()
        
        # --- PLAN ---
        vel_cmd, grip_cmd = brain.update(img, joints, cheat_data)
        
        # --- ACT ---
        sim.step(vel_cmd, grip_cmd)
        
        # --- VISUALIZE ---
        # Draw Camera Crosshair
        h, w = img.shape[:2]
        cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 1)
        cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 1)
        cv2.imshow("Robot Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # --- COMPREHENSIVE DIAGNOSTICS ---
        current_time = time.time()
        
        # 1. Update State Timer
        if brain.state != last_state:
            state_duration = 0.0
            state_start_time = current_time
            last_state = brain.state
        else:
            state_duration = current_time - state_start_time
            
        # 2. Calculate Real-World Metrics
        # Unpack joints: [x, y, z, yaw] (Fixed order from Controller)
        j_x, j_y, j_z, j_yaw = joints
        
        # Target (Microwave) relative to Robot
        mw_x = cheat_data['mw_x']
        mw_y = cheat_data['mw_y']
        
        err_x = mw_x - j_x
        err_y = mw_y - j_y
        dist_to_target = np.sqrt(err_x**2 + err_y**2)
        
        # Box relative to Microwave (Scene sanity check)
        box_offset_x = cheat_data['box_x'] - mw_x
        
        # 3. Print Dashboard
        if current_time - last_print > print_interval:
            # Calc FPS
            avg_dt = sum(loop_times) / len(loop_times) if loop_times else 0.0
            sim_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
            loop_times = [] # Reset buffer
            
            print("\n" + "="*50)
            print(f" SYSTEM HEARTBEAT | Sim Rate: {sim_hz:.1f} Hz")
            print("="*50)
            
            # Section A: Brain Status
            print(f" [CONTROLLER]")
            print(f"  State       : {brain.state:<10} (Active: {state_duration:.1f}s)")
            print(f"  Gripper     : {grip_cmd:<10} (Holding: {sim.grasped_body if sim.grasped_body else 'None'})")
            
            # Section B: Tracking Accuracy
            print(f" [TRACKING ERROR]")
            print(f"  X-Axis      : {err_x:+.4f} m  {'[OK]' if abs(err_x)<0.02 else '[LAG]'}")
            print(f"  Y-Axis      : {err_y:+.4f} m  {'[OK]' if abs(err_y)<0.05 else '[OFF]'}")
            print(f"  Euclidean   : {dist_to_target:.4f} m")
            
            # Section C: Robot State
            print(f" [ROBOT POSE]")
            print(f"  Position    : X={j_x:+.2f}, Y={j_y:+.2f}, Z={j_z:+.2f}")
            print(f"  Velocity CMD: Vx={vel_cmd[0]:.2f}, Vy={vel_cmd[1]:.2f}, Vz={vel_cmd[2]:.2f}")
            
            # Section D: Environment
            print(f" [SCENE]")
            print(f"  MW Pos      : X={mw_x:.2f}")
            print(f"  Box Offset  : {box_offset_x:+.2f} m (Should be ~0.80)")
            print("-" * 50)
            
            last_print = current_time

        # FPS Buffer Logic
        dt = time.time() - loop_start
        loop_times.append(dt)
        if len(loop_times) > max_loops_to_avg:
            loop_times.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()