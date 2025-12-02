import pyrealsense2 as rs
import numpy as np
import cv2

# Path to your .bag file
bag_file = "/home/sherali/Documents/Label_1.bag" 

# --- Setup and Frame Capture (from previous step) ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)
pipeline.start(config)

# Get depth scale early (needed for depth masking)
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

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

# --- START OF DEPTH MASKING STRATEGY (NEW STEP) ---

## 1. Create a Depth Mask
# You need to estimate the max/min depth (in meters) for the microwave in your scene.
# I'll use a likely range for a nearby object (1.0m to 2.5m). ADJUST THESE VALUES!
min_depth_m = 0.8  
max_depth_m = 2.5  

# Convert meter distances to raw depth units
min_depth_raw = int(min_depth_m / depth_scale)
max_depth_raw = int(max_depth_m / depth_scale)

# Create the depth mask (255 where depth is in range, 0 otherwise)
depth_mask = cv2.inRange(depth_image, min_depth_raw, max_depth_raw)

# Apply the Depth Mask to the Color Image
masked_color_image = cv2.bitwise_and(color_image, color_image, mask=depth_mask)
cv2.imshow("Depth Masked Color Image (Debug)", masked_color_image)
# --- END OF DEPTH MASKING STRATEGY ---


## 2. Preprocessing and Edge Detection on the MASKED image
gray = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150) # Use the same Canny thresholds

# Display the intermediate edge detection result for debugging
cv2.imshow("Canny Edges on Masked Image (Debug)", edges) 

## 3. Find and Filter Contours
# Find contours from the cleaned-up edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

microwave_corners = None
best_bbox_area = 0 

for c in contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(c)
    bbox_area = w * h # Area of the bounding rectangle
    
    # 🎯 Filtering 1: Bounding Box Area Check (Tune this value)
    # We filter based on the size of the *bounding box*, which is more reliable.
    if bbox_area < 15000:  
        continue
    
    # Calculate perimeter and approximate the contour polygon
    perimeter = cv2.arcLength(c, True)
    # 🎯 Filtering 2: Approximation Tolerance (Increased for better quad detection)
    approx = cv2.approxPolyDP(c, 0.06 * perimeter, True) 
    
    # 🎯 Filtering 3: Shape Check (must have 4 vertices)
    if len(approx) == 4:  
        
        # 🎯 Filtering 4: Aspect Ratio Check (Microwave is wider than tall)
        aspect_ratio = w / float(h)
        if aspect_ratio < 1.0 or aspect_ratio > 3.0: 
             continue
        
        # If multiple 4-sided shapes are found, take the largest one by bounding box area
        if bbox_area > best_bbox_area:
            best_bbox_area = bbox_area
            microwave_corners = approx.reshape(-1, 2)
            
            # Use the bounding box for the final visualization
            cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 255, 0), 3)


# --- END OF MICROWAVE DETECTION ---


## 4. Visualization and Final Display
if microwave_corners is not None:
    print(f"✅ Microwave detected with bounding box area: {best_bbox_area:.0f}")
else:
    print("❌ Microwave (4-sided contour) NOT found based on current filters.")

# Display the initial frames and the final detection result
cv2.imshow("Original Color Image", color_image)
cv2.imshow("Final Microwave Detection (Bounding Box)", detection_image)
cv2.imshow("Depth", cv2.convertScaleAbs(depth_image, alpha=0.03))


# Wait indefinitely until a key is pressed to close the windows
print("\nFrame displayed. Press any key to close the windows and exit...")
cv2.waitKey(0) 

pipeline.stop()
cv2.destroyAllWindows()