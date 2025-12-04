import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Import from your packages
from src.sim_interface import GantrySim
from src.controller import GantryController

def main():
    # 1. Setup
    sim = GantrySim(render=True)
    brain = GantryController()
    
    print("Initialization Complete. Running... (Press 'q' to quit and see graphs)")
    
    # --- DATA RECORDER SETUP ---
    history = {
        'time': [],
        'robot_x': [], 'target_x': [],
        'robot_y': [], 'target_y': [],
        'robot_z': [], 
        'robot_yaw': [], 'target_yaw': []
    }
    
    start_time = time.time()
    last_print = time.time()
    print_interval = 0.5 # Seconds
    
    while True:
        # --- SENSE ---
        img, joints, cheat_data = sim.get_data()
        
        # --- PLAN ---
        vel_cmd, grip_cmd = brain.update(img, joints, cheat_data)
        
        # --- ACT ---
        sim.step(vel_cmd, grip_cmd)
        
        # --- PRINT PIXEL COORDINATES (Heartbeat) ---
        if time.time() - last_print > print_interval:
            print(f"[{time.time()-start_time:.1f}s] CAM PIXELS | "
                  f"MW: ({cheat_data['mw_u']:3d}, {cheat_data['mw_v']:3d}) | "
                  f"BOX: ({cheat_data['box_u']:3d}, {cheat_data['box_v']:3d})")
            last_print = time.time()

        # --- RECORD DATA ---
        current_time = time.time() - start_time
        
        # Identify Target based on State for plotting context
        if brain.state in ["APPROACH_BOX", "INSERT", "RELEASE_IN_BOX", "CLEAR_BOX"]:
            t_x = cheat_data['box_x']
            t_y = cheat_data['box_y']
            t_yaw = cheat_data['box_yaw']
        else:
            t_x = cheat_data['mw_x']
            t_y = cheat_data['mw_y']
            t_yaw = cheat_data['mw_yaw']

        # Robot State (Joints: y, x, z, yaw -> careful with mapping!)
        # Check your controller mapping. Usually: joints[0]=X or Y? 
        # In Sim: joints = [p.getJointState...]. 
        # URDF Joint 0=Slider_X, 1=Slider_Y, 2=Slider_Z, 3=Yaw
        # Let's map explicitly based on your Sim/Controller logic:
        # Sim returns joints list: [j0, j1, j2, j3]
        j_x, j_y, j_z, j_yaw = joints
        
        history['time'].append(current_time)
        history['robot_x'].append(j_x); history['target_x'].append(t_x)
        history['robot_y'].append(j_y); history['target_y'].append(t_y)
        history['robot_z'].append(j_z) # Z target is internal logic, mostly just want to see robot Z
        history['robot_yaw'].append(j_yaw); history['target_yaw'].append(t_yaw)
        
        # --- VISUALIZE ---
        # Draw crosshair at center
        h, w = img.shape[:2]
        cv2.line(img, (w//2, h//2-10), (w//2, h//2+10), (0,255,0), 1)
        cv2.line(img, (w//2-10, h//2), (w//2+10, h//2), (0,255,0), 1)
        
        # Draw target pixel (Just for debug visual)
        if brain.state in ["APPROACH_BOX", "INSERT"]:
            cv2.circle(img, (cheat_data['box_u'], cheat_data['box_v']), 5, (0, 0, 255), 2)
        elif brain.state in ["TRACKING", "DESCEND", "GRASP"]:
            cv2.circle(img, (cheat_data['mw_u'], cheat_data['mw_v']), 5, (255, 0, 0), 2)

        cv2.imshow("Robot Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # --- PLOTTING ---
    print("Generating Graphs...")
    plot_data(history)

def plot_data(history):
    t = history['time']
    
    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    
    # 1. X-Axis (Tracking)
    axs[0, 0].plot(t, history['target_x'], 'r--', label='Target')
    axs[0, 0].plot(t, history['robot_x'], 'b-', label='Robot X')
    axs[0, 0].set_title('X-Axis (Conveyor Tracking)')
    axs[0, 0].set_ylabel('Position (m)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Y-Axis (Alignment)
    axs[0, 1].plot(t, history['target_y'], 'r--', label='Target')
    axs[0, 1].plot(t, history['robot_y'], 'g-', label='Robot Y')
    axs[0, 1].set_title('Y-Axis (Cross-Track)')
    axs[0, 1].set_ylabel('Position (m)')
    axs[0, 1].grid(True)
    
    # 3. Z-Axis (Vertical Profile)
    axs[1, 0].plot(t, history['robot_z'], 'k-', label='Robot Z')
    axs[1, 0].set_title('Z-Axis (Height Profile)')
    axs[1, 0].set_ylabel('Height (m)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].grid(True)
    # Add reference lines for pickup/drop heights if useful
    axs[1, 0].axhline(y=-0.2, color='gray', linestyle=':', alpha=0.5, label='Pickup')
    axs[1, 0].legend()

    # 4. Yaw (Rotation)
    axs[1, 1].plot(t, history['target_yaw'], 'r--', label='Target')
    axs[1, 1].plot(t, history['robot_yaw'], 'm-', label='Robot Yaw')
    axs[1, 1].set_title('Yaw (Orientation)')
    axs[1, 1].set_ylabel('Angle (rad)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()