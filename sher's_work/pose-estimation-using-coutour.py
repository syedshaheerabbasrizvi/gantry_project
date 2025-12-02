# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Path to your .bag file
# bag_file = "/home/sherali/Documents/Label_1.bag"  # change 'username' to your user folder

# # Create pipeline
# pipeline = rs.pipeline()
# config = rs.config()

# # Load recorded bag file
# config.enable_device_from_file(bag_file, repeat_playback=False)

# # Configure streams
# config.enable_stream(rs.stream.color)
# config.enable_stream(rs.stream.depth)

# # Start pipeline
# pipeline.start(config)

# # Align depth to color
# align_to = rs.stream.color
# align = rs.align(align_to)

# while True:
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)

#     color_frame = aligned_frames.get_color_frame()
#     depth_frame = aligned_frames.get_depth_frame()
#     if not color_frame or not depth_frame:
#         break

#     # Convert to numpy arrays
#     color_image = np.asanyarray(color_frame.get_data())
#     depth_image = np.asanyarray(depth_frame.get_data())

#     cv2.imshow("Color", color_image)
#     cv2.imshow("Depth", cv2.convertScaleAbs(depth_image, alpha=0.03))

#     if cv2.waitKey(1) & 0xFF == 27:  # press ESC to stop
#         break

# pipeline.stop()
# cv2.destroyAllWindows()








# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Path to your .bag file
# bag_file = "/home/sherali/Documents/Label_1.bag" 

# # Create pipeline
# pipeline = rs.pipeline()
# config = rs.config()

# # Load recorded bag file
# config.enable_device_from_file(bag_file, repeat_playback=False)

# # Configure streams
# config.enable_stream(rs.stream.color)
# config.enable_stream(rs.stream.depth)

# # Start pipeline
# pipeline.start(config)

# # Align depth to color
# align_to = rs.stream.color
# align = rs.align(align_to)

# # --- START OF MODIFICATIONS ---

# # 1. READ A SINGLE FRAME
# try:
#     # Wait for the first set of frames (Color and Depth)
#     frames = pipeline.wait_for_frames() 
# except RuntimeError as e:
#     print(f"Error reading frame: {e}. The .bag file might be empty or corrupt.")
#     pipeline.stop()
#     cv2.destroyAllWindows()
#     exit() # Exit if no frame is read

# aligned_frames = align.process(frames)

# color_frame = aligned_frames.get_color_frame()
# depth_frame = aligned_frames.get_depth_frame()

# if color_frame and depth_frame:
#     # Convert to numpy arrays
#     color_image = np.asanyarray(color_frame.get_data())
#     depth_image = np.asanyarray(depth_frame.get_data())

#     # Display the images
#     cv2.imshow("Color", color_image)
#     cv2.imshow("Depth", cv2.convertScaleAbs(depth_image, alpha=0.03))

#     # 2. WAIT INDEFINITELY UNTIL A KEY IS PRESSED (e.g., to close the window)
#     print("Frame displayed. Press any key to close the windows...")
#     cv2.waitKey(0) # Change from cv2.waitKey(1) to cv2.waitKey(0)
# else:
#     print("Could not retrieve aligned color or depth frame.")

# # --- END OF MODIFICATIONS ---

# pipeline.stop()
# cv2.destroyAllWindows()




import pyrealsense2 as rs
import numpy as np
import cv2

# Path to your .bag file
bag_file = "/home/sherali/Documents/label_2.bag" 

# --- Setup and Frame Capture (from previous step) ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    # Wait for the first set of frames
    frames = pipeline.wait_for_frames() 
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
except RuntimeError as e:
    print(f"Error reading frame: {e}. Exiting.")
    pipeline.stop()
    cv2.destroyAllWindows()
    exit()

if color_frame is None or depth_frame is None:
    print("Could not retrieve aligned color or depth frame. Exiting.")
    pipeline.stop()
    cv2.destroyAllWindows()
    exit()

# Convert frames to numpy arrays
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Create a copy of the color image for drawing the final result
detection_image = color_image.copy()

# --- START OF MICROWAVE DETECTION (Step 3) ---

## 1. Preprocessing and Edge Detection
gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150) # You might need to tune these thresholds (50, 150)

# Display the intermediate edge detection result for debugging
cv2.imshow("Canny Edges (Debug)", edges) 

## 2. Find and Filter Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- NEW VISUALIZATION: Display ALL found contours ---
all_contours_image = color_image.copy() # Create a fresh copy for drawing all contours
cv2.drawContours(all_contours_image, contours, -1, (255, 0, 0), 2) # Draw all contours in blue
cv2.imshow("All Contours (Debug)", all_contours_image)
# --- END NEW VISUALIZATION ---


microwave_corners = None
best_area = 0 # Will now track the largest *bounding box* area

for c in contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(c)
    bbox_area = w * h # Area of the bounding rectangle
    
    # 🎯 Filtering 1: Bounding Box Area Check (Tune this value, start lower than 20000)
    # This filters out small noise contours, even if their approximation is 4-sided
    if bbox_area < 8000:  # <--- Significantly lowered threshold
        continue
    
    # Calculate perimeter and approximate the contour polygon
    perimeter = cv2.arcLength(c, True)
    # The approximation tolerance (0.02 * perimeter) determines how 'tight' the polygon must be
    # Increased tolerance to 0.06 to allow for more imperfect/rounded rectangles
    approx = cv2.approxPolyDP(c, 0.06 * perimeter, True) 
    
    # 🎯 Filtering 2: Shape Check (must have 4 vertices)
    if len(approx) == 4:  
        
        # We now compare the bounding box area
        if bbox_area > best_area:
            best_area = bbox_area
            microwave_corners = approx.reshape(-1, 2)
            
            # Draw the *best candidate so far* on the detection image
            # Draw on a copy to see only the best one at the end
            # We are drawing the 4-sided approximation
            cv2.drawContours(detection_image, [approx], -1, (0,255,0), 3) 
            
            # Optional: Draw the bounding box of the best candidate for visual confirmation
            # cv2.rectangle(detection_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
# --- END OF MICROWAVE DETECTION ---

## 3. Visualization and Final Display
if microwave_corners is not None:
    print(f"✅ Microwave detected with area: {best_area:.0f}")
    # Optional: Draw the four corners as small circles for clear visualization of the vertices
    for corner in microwave_corners:
        x, y = corner
        cv2.circle(detection_image, (x, y), 5, (0, 0, 255), -1) # Red dots for corners
else:
    print("❌ Microwave (4-sided contour) NOT found based on current filters.")

# Display the initial frames and the final detection result
cv2.imshow("Original Color Image", color_image)
cv2.imshow("Final Microwave Detection", detection_image)
cv2.imshow("Depth", cv2.convertScaleAbs(depth_image, alpha=0.03))


# Wait indefinitely until a key is pressed to close the windows
print("\nFrame displayed. Press any key to close the windows and exit...")
cv2.waitKey(0) 

pipeline.stop()
cv2.destroyAllWindows()