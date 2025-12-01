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
        'robot_x': [],
        'target_x': [],
        'error_x': [],
        'cmd_vel_x': []
    }
    start_time = time.time()
    
    while True:
        # --- SENSE ---
        img, joints, cheat_data = sim.get_data()
        
        # --- PLAN ---
        vel_cmd, grip_cmd = brain.update(img, joints, cheat_data)
        
        # --- ACT ---
        sim.step(vel_cmd, grip_cmd)
        
        # --- RECORD DATA ---
        # Only record if we are properly initialized (avoid initial 0.0 glitches)
        current_time = time.time() - start_time
        
        # Identify Target (MW or Box depending on state)
        # This helps the graph make sense during different phases
        if brain.state in ["APPROACH_BOX", "INSERT"]:
            target_x = cheat_data['box_x']
        else:
            target_x = cheat_data['mw_x']

        j_x = joints[0] # Assuming Joint 0 is X
        
        history['time'].append(current_time)
        history['robot_x'].append(j_x)
        history['target_x'].append(target_x)
        history['error_x'].append(target_x - j_x)
        history['cmd_vel_x'].append(vel_cmd[0])
        
        # --- VISUALIZE ---
        cv2.imshow("Robot Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # --- PLOTTING ---
    print("Generating Graphs...")
    plot_data(history)

def plot_data(history):
    t = history['time']
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Position Tracking
    ax1.plot(t, history['target_x'], 'r--', label='Target (MW/Box)')
    ax1.plot(t, history['robot_x'], 'b-', label='Robot X')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('X-Axis Tracking Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Tracking Error
    ax2.plot(t, history['error_x'], 'k-', label='Error (Target - Robot)')
    # Draw the "Good Tracking" zone (+/- 2cm)
    ax2.axhline(y=0.02, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(y=-0.02, color='g', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Tracking Error')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Velocity Commands (Shows the Ramping)
    ax3.plot(t, history['cmd_vel_x'], 'm-', label='Cmd Vel X')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Velocity Profile (Ramped)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()