
import cv2
import time
# Import from the new src package
from src.sim_interface import GantrySim
from src.controller import GantryController

def main():
    # 1. Setup
    sim = GantrySim(render=True)
    brain = GantryController()
    
    print("Initialization Complete. Running...")
    
    while True:
        # --- SENSE ---
        img, joints, cheat_data = sim.get_data()
        
        # --- PLAN ---
        vel_cmd, grip_cmd = brain.update(img, joints, cheat_data)
        
        # --- ACT ---
        sim.step(vel_cmd, grip_cmd)
        
        # --- VISUALIZE ---
        # Draw a crosshair to show camera rotation
        cv2.line(img, (120, 100), (120, 140), (0, 255, 0), 1)
        cv2.line(img, (100, 120), (140, 120), (0, 255, 0), 1)
        
        cv2.imshow("Robot Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()