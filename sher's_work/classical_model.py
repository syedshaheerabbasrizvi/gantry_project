import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time

class RealSenseColorDetector:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Create align object
        self.align = rs.align(rs.stream.color)
        
        # Color ranges for different objects (HSV format)
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 150, 0]), np.array([140, 255, 255]))
            ],
            'green': [
                (np.array([40, 40, 40]), np.array([80, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))
            ]
        }
        
        self.detected_objects = []
        
    def create_trackbars(self):
        """Create trackbars for adjusting color ranges in real-time"""
        cv2.namedWindow('Color Adjustments')
        
        # Create trackbars for HSV range
        cv2.createTrackbar('H_min', 'Color Adjustments', 0, 179, lambda x: None)
        cv2.createTrackbar('S_min', 'Color Adjustments', 100, 255, lambda x: None)
        cv2.createTrackbar('V_min', 'Color Adjustments', 100, 255, lambda x: None)
        cv2.createTrackbar('H_max', 'Color Adjustments', 10, 179, lambda x: None)
        cv2.createTrackbar('S_max', 'Color Adjustments', 255, 255, lambda x: None)
        cv2.createTrackbar('V_max', 'Color Adjustments', 255, 255, lambda x: None)
        
    def get_trackbar_values(self):
        """Get current trackbar values"""
        return {
            'h_min': cv2.getTrackbarPos('H_min', 'Color Adjustments'),
            's_min': cv2.getTrackbarPos('S_min', 'Color Adjustments'),
            'v_min': cv2.getTrackbarPos('V_min', 'Color Adjustments'),
            'h_max': cv2.getTrackbarPos('H_max', 'Color Adjustments'),
            's_max': cv2.getTrackbarPos('S_max', 'Color Adjustments'),
            'v_max': cv2.getTrackbarPos('V_max', 'Color Adjustments')
        }
    
    def detect_color_objects(self, color_image, color_name='red'):
        """Detect objects based on color"""
        # Convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Create mask based on color range
        mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        
        if color_name in self.color_ranges:
            for lower, upper in self.color_ranges[color_name]:
                mask += cv2.inRange(hsv, lower, upper)
        else:
            # Use trackbar values
            values = self.get_trackbar_values()
            lower = np.array([values['h_min'], values['s_min'], values['v_min']])
            upper = np.array([values['h_max'], values['s_max'], values['v_max']])
            mask = cv2.inRange(hsv, lower, upper)
        
        # Special processing for black detection
        if color_name == 'black':
            # Additional processing to improve black detection
            # Convert to grayscale for additional thresholding
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Use adaptive threshold for better black detection
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)
            
            # Combine with HSV mask
            mask = cv2.bitwise_and(mask, adaptive_thresh)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def get_contour_centers(self, mask):
        """Find contours and return their centers"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                # Method 1: Centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy, contour, area))
                
        return centers
    
    def get_3d_coordinates(self, depth_frame, centers):
        """Convert 2D centers to 3D coordinates"""
        object_data = []
        
        for cx, cy, contour, area in centers:
            # Get depth at center point
            depth = depth_frame.get_distance(cx, cy)
            
            if depth > 0:  # Valid depth measurement
                # Convert to 3D coordinates
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                
                # Estimate object size
                x, y, w, h = cv2.boundingRect(contour)
                
                obj_info = {
                    'coordinates_3d': {
                        'x': round(point_3d[0], 3),
                        'y': round(point_3d[1], 3),
                        'z': round(point_3d[2], 3)
                    },
                    'pixel_coordinates': {
                        'center_x': cx,
                        'center_y': cy
                    },
                    'size': {
                        'area': area,
                        'width': w,
                        'height': h
                    },
                    'timestamp': time.time()
                }
                object_data.append(obj_info)
                
        return object_data
    
    def visualize_detection(self, color_image, mask, centers, object_data):
        """Create comprehensive visualization"""
        # Create main display
        main_display = np.zeros((1000, 1600, 3), dtype=np.uint8)
        
        # 1. Original Color Image
        color_display = color_image.copy()
        cv2.putText(color_display, "1. Original Color", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        main_display[0:480, 0:640] = cv2.resize(color_display, (640, 480))
        
        # 2. HSV Image
        hsv_display = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        hsv_display = cv2.cvtColor(hsv_display, cv2.COLOR_HSV2BGR)
        cv2.putText(hsv_display, "2. HSV Color Space", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        main_display[0:480, 640:1280] = cv2.resize(hsv_display, (640, 480))
        
        # 3. Color Mask
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_display, "3. Color Mask", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        main_display[480:960, 0:640] = cv2.resize(mask_display, (640, 480))
        
        # 4. Final Detection with 3D Coordinates
        final_display = color_image.copy()
        
        for i, (cx, cy, contour, area) in enumerate(centers):
            # Draw contour
            cv2.drawContours(final_display, [contour], -1, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(final_display, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(final_display, (cx, cy), 4, (255, 255, 255), -1)
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(final_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add object info if we have 3D data
            if i < len(object_data):
                coords = object_data[i]['coordinates_3d']
                coord_text = f"({coords['x']:.2f}, {coords['y']:.2f}, {coords['z']:.2f})m"
                cv2.putText(final_display, coord_text, (cx - 80, cy - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                area_text = f"Area: {area}"
                cv2.putText(final_display, area_text, (cx - 40, cy + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(final_display, "4. Final Detection + 3D Coords", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        main_display[480:960, 640:1280] = cv2.resize(final_display, (640, 480))
        
        # 5. Information Panel
        info_panel = np.zeros((480, 320, 3), dtype=np.uint8) + 50
        
        cv2.putText(info_panel, "COLOR DETECTION INFO", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(info_panel, f"Objects Found: {len(centers)}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display object coordinates
        y_offset = 120
        for i, obj in enumerate(object_data):
            coords = obj['coordinates_3d']
            text = f"Obj {i+1}: ({coords['x']:.2f}, {coords['y']:.2f})"
            cv2.putText(info_panel, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            depth_text = f"     Z: {coords['z']:.2f}m"
            cv2.putText(info_panel, depth_text, (20, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 50
        
        # Controls
        cv2.putText(info_panel, "CONTROLS:", (20, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(info_panel, "Q - Quit", (20, y_offset + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, "S - Save Data", (20, y_offset + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, "1-4 - Colors", (20, y_offset + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        main_display[0:480, 1280:1600] = info_panel
        
        return main_display
    
    def run_detection(self, use_trackbars=True):
        """Main detection loop"""
        if use_trackbars:
            self.create_trackbars()
        
        current_color = 'black'
        frame_count = 0
        start_time = time.time()
        
        try:
            print("Starting color-based object detection...")
            print("Controls: 1=Red, 2=Blue, 3=Green, 4=Yellow, Q=Quit, S=Save")
            
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Detect objects by color
                mask = self.detect_color_objects(color_image, current_color)
                centers = self.get_contour_centers(mask)
                object_data = self.get_3d_coordinates(depth_frame, centers)
                
                # Store data
                self.detected_objects.extend(object_data)
                
                # Print to console
                frame_count += 1
                if frame_count % 30 == 0 and object_data:
                    print(f"\n--- Frame {frame_count} ---")
                    for i, obj in enumerate(object_data):
                        coords = obj['coordinates_3d']
                        print(f"Object {i+1}: X={coords['x']:.3f}m, Y={coords['y']:.3f}m, Z={coords['z']:.3f}m")
                
                # Create visualization
                visualization = self.visualize_detection(color_image, mask, centers, object_data)
                
                # Add FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(visualization, f"FPS: {fps:.1f}", (1400, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(visualization, f"Color: {current_color}", (1400, 500), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display
                cv2.imshow('Color-Based Object Detection', visualization)
                cv2.resizeWindow('Color-Based Object Detection', 1600, 1000)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_to_json()
                    print("Data saved!")
                elif key == ord('1'):
                    current_color = 'red'
                    print("Switched to RED detection")
                elif key == ord('2'):
                    current_color = 'blue'
                    print("Switched to BLUE detection")
                elif key == ord('3'):
                    current_color = 'green'
                    print("Switched to GREEN detection")
                elif key == ord('4'):
                    current_color = 'yellow'
                    print("Switched to YELLOW detection")
                # And add this in the key press section:
                # elif key == ord('5'):
                #     current_color = 'black'
                #     print("Switched to BLACK detection")
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            print(f"\nSession Summary:")
            print(f"Frames processed: {frame_count}")
            print(f"Total objects detected: {len(self.detected_objects)}")
    
    def save_to_json(self, filename="color_detection_objects.json"):
        """Save detected objects to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.detected_objects, f, indent=2)
        print(f"Saved {len(self.detected_objects)} objects to {filename}")

# Method 2: Shape-Based Detection (Alternative)
class ShapeDetector:
    def detect_shapes(self, color_image):
        """Detect objects based on shape (basic implementation)"""
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                # Approximate the contour
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify shape based on number of vertices
                if len(approx) == 3:
                    shape = "triangle"
                elif len(approx) == 4:
                    shape = "rectangle"
                elif len(approx) > 8:
                    shape = "circle"
                else:
                    shape = "unknown"
                
                shapes.append((contour, shape, area))
        
        return shapes

if __name__ == "__main__":
    print("=" * 60)
    print("COLOR-BASED OBJECT DETECTION - No YOLO Required!")
    print("=" * 60)
    print("\nPlace colored objects in front of the camera.")
    print("The system will detect them and provide 3D coordinates!")
    
    detector = RealSenseColorDetector()
    detector.run_detection(use_trackbars=True)