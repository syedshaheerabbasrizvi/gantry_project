import cv2
import time
# Import from the new src package
from src.sim_interface import GantrySim
from src.controller import GantryController
from src.vision_test import VisionPipeline


def main():
    # 1. Setup
    sim = GantrySim(render=True)
    brain = GantryController()
    
    print("Initialization Complete. Running...")
    
    # Timing
    start_time = time.perf_counter()
    step_count = 0
    total_loop_time = 0.0

    vision = VisionPipeline()
    
    try:
        while True:
            loop_start = time.perf_counter()
             # --- SENSE ---
            wrist_img, top_img, joints, cheat_data, cam_pos, cam_orn  = sim.get_data()
            
            # test code for vision testing
            processed_image = vision.process_vision(wrist_img)

            # --- PLAN ---
            vel_cmd, grip_cmd = brain.update(wrist_img, joints, cheat_data, cam_pos, cam_orn)
             
            # --- ACT ---
            sim.step(vel_cmd, grip_cmd)
             
            # --- VISUALIZE ---
            # Draw crosshair on wrist camera
            cv2.line(wrist_img, (120, 100), (120, 140), (0, 255, 0), 1)
            cv2.line(wrist_img, (100, 120), (140, 120), (0, 255, 0), 1)
 
            cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("Top Camera",   cv2.cvtColor(top_img,   cv2.COLOR_RGB2BGR))
            cv2.imshow("Processed image",   cv2.cvtColor(processed_image,   cv2.COLOR_RGB2BGR))

 
            step_count += 1
            loop_dt = time.perf_counter() - loop_start
            total_loop_time += loop_dt

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_hz = step_count / total_time if total_time > 0 else 0.0
        avg_loop_ms = (total_loop_time / step_count * 1000) if step_count > 0 else 0.0
        camera_fps = step_count / total_time
        print(f"Simulation run: steps={step_count}, wall_time={total_time:.3f}s, avg_hz={avg_hz:.2f}, avg_loop_ms={avg_loop_ms:.2f}ms, camera_fps={camera_fps:.3f}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

