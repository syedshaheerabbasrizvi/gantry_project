"""
Microwave Detection + Pose Estimation using HSV Segmentation and Depth (Intel RealSense D555)
Author: Sher Ali Khan + ChatGPT
Date: 2025-10-31
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# ===============================================================
# 1️⃣  SETUP PIPELINE AND LOAD .BAG FILE
# ===============================================================

bag_file = "/home/sherali/Documents/middle_view02.bag"

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)
pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# ===============================================================
# 2️⃣  READ ONE FRAME (RGB + DEPTH)
# ===============================================================

try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
except RuntimeError as e:
    print(f"Error reading frame: {e}")
    pipeline.stop()
    exit()

if color_frame is None or depth_frame is None:
    print("Could not retrieve aligned color or depth frame. Exiting.")
    pipeline.stop()
    exit()

# Convert frames to numpy arrays
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Display raw color + depth
cv2.imshow("Step 1: Original Color Image", color_image)
cv2.imshow("Step 1: Raw Depth Image", cv2.convertScaleAbs(depth_image, alpha=0.03))
cv2.waitKey(500)

# ===============================================================
# 3️⃣  COLOR/BRIGHTNESS SEGMENTATION (ISOLATE MICROWAVE)
# ===============================================================

# Convert to HSV and extract brightness channel
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_image)

# Threshold for dark regions (microwave)
upper_brightness_limit = 50  # tune this value if needed
dark_mask = cv2.inRange(V, 0, upper_brightness_limit)

cv2.imshow("Step 2: Initial Dark Mask", dark_mask)
cv2.waitKey(500)

# Clean the mask with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imshow("Step 3: Cleaned Segmentation Mask", cleaned_mask)
cv2.waitKey(500)

# ===============================================================
# 4️⃣  FIND MICROWAVE BOUNDING BOX
# ===============================================================

contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
microwave_bbox = None
bbox_image = color_image.copy()

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) > 5000:
        x, y, w, h = cv2.boundingRect(largest_contour)
        microwave_bbox = (x, y, w, h)
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(f"✅ Microwave detected. Bounding box area = {w*h}")
    else:
        print("❌ Largest contour too small to be the microwave.")
else:
    print("❌ No contours detected.")

cv2.imshow("Step 4: Detected Bounding Box", bbox_image)
cv2.waitKey(500)

# ===============================================================
# 5️⃣  ORIENTATION ESTIMATION INSIDE BOUNDING BOX (HOUGH TRANSFORM)
# ===============================================================

if microwave_bbox is not None:
    x, y, w, h = microwave_bbox

    # Crop region of interest (ROI)
    roi_mask = cleaned_mask[y:y+h, x:x+w]
    roi_color = color_image[y:y+h, x:x+w]

    # Step 1: Edge detection
    edges = cv2.Canny(roi_mask, 50, 150, apertureSize=3)

    # Step 2: Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

    if lines is not None:
        # Convert to degrees and collect
        angles = [np.degrees(theta) for rho, theta in lines[:, 0]]

        # Normalize angles to 0–180 range
        angles = [a if a <= 180 else a - 180 for a in angles]

        # Compute dominant (median) angle
        dominant_angle = np.median(angles)
        print(f"📏 Dominant edge angle (inside bounding box): {dominant_angle:.2f}°")

        # Step 3: Draw a few representative lines back on the ROI
        for i, (rho, theta) in enumerate(lines[:10, 0]):  # limit for clarity
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(roi_color, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Step 4: Display cropped region with detected lines
        cv2.imshow("Step 5: ROI Hough Lines", roi_color)
        cv2.waitKey(500)

        # Step 5: (Optional) Draw orientation line back on original image
        angle_rad = np.radians(dominant_angle)
        center_x = x + w // 2
        center_y = y + h // 2
        line_length = min(w, h) // 2
        x2 = int(center_x + line_length * np.cos(angle_rad))
        y2 = int(center_y + line_length * np.sin(angle_rad))
        orientation_image = color_image.copy()
        cv2.line(orientation_image, (center_x, center_y), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            orientation_image,
            f"{dominant_angle:.2f} deg",
            (center_x - 50, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Step 6: Global Orientation Overlay", orientation_image)
        cv2.waitKey(0)

    else:
        print("⚠️ No strong edges found inside bounding box.")


print("\nPress any key to exit...")
cv2.waitKey(0)

pipeline.stop()
cv2.destroyAllWindows()





















# # ===============================================================
# # 5️⃣  POSE ESTIMATION USING DEPTH DATA
# # ===============================================================

# pose_image = color_image.copy()

# if microwave_bbox is not None:
#     x, y, w, h = microwave_bbox

#     # Compute 4 corner points of bounding box (image coordinates)
#     corners_2D = np.array([
#         [x, y],             # top-left
#         [x + w, y],         # top-right
#         [x + w, y + h],     # bottom-right
#         [x, y + h]          # bottom-left
#     ], dtype=np.float32)

#     # Draw corners for visualization
#     for (u, v) in corners_2D:
#         cv2.circle(pose_image, (int(u), int(v)), 6, (0, 0, 255), -1)
#     cv2.imshow("Step 5: Bounding Box Corners", pose_image)
#     cv2.waitKey(500)

#     # Get camera intrinsics
#     profile = pipeline.get_active_profile()
#     intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
#     camera_matrix = np.array([
#         [intr.fx, 0, intr.ppx],
#         [0, intr.fy, intr.ppy],
#         [0, 0, 1]
#     ])
#     dist_coeffs = np.array(intr.coeffs)

#     # Depth scale
#     depth_sensor = profile.get_device().first_depth_sensor()
#     depth_scale = depth_sensor.get_depth_scale()

#     # Get image shape for safe indexing
#     height, width = depth_image.shape

#     # Get 3D points from depth (with clamped indices)
#     object_points = []
#     valid_corners_2D = []

#     for (u, v) in corners_2D:
#         u = np.clip(int(u), 0, width - 1)
#         v = np.clip(int(v), 0, height - 1)

#         Z = depth_image[v, u] * depth_scale
#         if Z == 0:  # skip invalid depth values
#             continue
        

#         X = (u - intr.ppx) * Z / intr.fx
#         Y = (v - intr.ppy) * Z / intr.fy

#         # print("These are u, v: ", [u, v])
#         # print("these are x, x, z: ", [X, Y, Z])

#         object_points.append([X, Y, Z])
#         valid_corners_2D.append([u, v])

#     # print("these are object points: ", object_points)

#     if len(object_points) == 4:
#         object_points = np.array(object_points, dtype=np.float32)
#         valid_corners_2D = np.array(valid_corners_2D, dtype=np.float32)

#         retval, rvec, tvec = cv2.solvePnP(
#             object_points,
#             valid_corners_2D,
#             camera_matrix,
#             dist_coeffs,
#             flags=cv2.SOLVEPNP_AP3P  # <--- change this
#             # flags=cv2.SOLVEPNP_ITERATIVE
#         )

#         if retval:
#             cv2.drawFrameAxes(pose_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
#             print("\n✅ Pose estimated successfully.")
#             print(f"Translation (X,Y,Z): {tvec.ravel()} meters")
#             print(f"Rotation Vector (rvec): {rvec.ravel()}")
#         else:
#             print("❌ solvePnP failed to compute pose.")
#     else:
#         print("❌ Not enough valid depth points for pose estimation.")

# cv2.imshow("Step 6: Pose Visualization (3D Axes)", pose_image)
# cv2.waitKey(500)

# # ===============================================================
# # 6️⃣  DISPLAY FINAL COMBINED VIEW
# # ===============================================================

# depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
# depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
# combined = np.hstack((pose_image, depth_colormap))

# cv2.imshow("Step 7: Final Combined View (Pose + Depth)", combined)

# print("\nPress any key to exit...")
# cv2.waitKey(0)

# pipeline.stop()
# cv2.destroyAllWindows()




































# # import pyrealsense2 as rs
# # import numpy as np
# # import cv2

# # # Path to your .bag file
# # bag_file = "/home/sherali/Documents/Label_1.bag" 

# # # --- Setup and Frame Capture ---
# # pipeline = rs.pipeline()
# # config = rs.config()
# # config.enable_device_from_file(bag_file, repeat_playback=False)
# # config.enable_stream(rs.stream.color)
# # config.enable_stream(rs.stream.depth) # Still keeping depth for completeness, but not used in the core segmentation
# # pipeline.start(config)

# # align_to = rs.stream.color
# # align = rs.align(align_to)

# # try:
# #     frames = pipeline.wait_for_frames() 
# #     aligned_frames = align.process(frames)
# #     color_frame = aligned_frames.get_color_frame()
# #     depth_frame = aligned_frames.get_depth_frame() # Keep for display
# # except RuntimeError as e:
# #     print(f"Error reading frame: {e}. Exiting.")
# #     pipeline.stop()
# #     cv2.destroyAllWindows()
# #     exit()

# # if color_frame is None or depth_frame is None:
# #     print("Could not retrieve aligned color or depth frame. Exiting.")
# #     pipeline.stop()
# #     cv2.destroyAllWindows()
# #     exit()

# # # Convert frames to numpy arrays
# # color_image = np.asanyarray(color_frame.get_data())
# # depth_image = np.asanyarray(depth_frame.get_data())

# # # Create a copy of the color image for drawing the final result
# # detection_image = color_image.copy()

# # # --- START OF COLOR/INTENSITY SEGMENTATION STRATEGY ---

# # ## 1. Convert to HSV and Isolate the Value (Brightness) Channel
# # hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
# # H, S, V = cv2.split(hsv_image) 

# # ## 2. Isolate Dark Areas (The Black Microwave)
# # # Black objects have a very low Value (Brightness)
# # # We set a threshold for V (0 to 120 is a good starting range for dark/black)
# # # TUNE THIS UPPER THRESHOLD (e.g., 100) based on how dark the microwave appears.
# # upper_brightness_limit = 100 
# # dark_mask = cv2.inRange(V, 0, upper_brightness_limit) 

# # cv2.imshow("Initial Dark Mask (Debug)", dark_mask)

# # ## 3. Clean the Mask with Morphological Operations
# # # Use a kernel large enough to connect the pieces of the microwave's surface.
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 

# # # MORPH_CLOSE (Dilation then Erosion) fills small gaps and holes within the dark area
# # cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
# # # MORPH_OPEN (Erosion then Dilation) removes small, isolated noise dots
# # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1) 

# # cv2.imshow("Cleaned Segmentation Mask (Debug)", cleaned_mask)

# # ## 4. Find the largest contour from the CLEANED MASK
# # # Find contours from the cleaned-up binary mask
# # contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # best_area = 0 
# # x, y, w, h = 0, 0, 0, 0

# # if contours:
# #     # Select the largest contour (the microwave should be the largest dark object)
# #     largest_contour = max(contours, key=cv2.contourArea)
    
# #     # Check if the largest contour is sufficiently large to be the microwave
# #     if cv2.contourArea(largest_contour) > 5000: # Tune this minimum area
# #         # Calculate the bounding box for the largest contour
# #         x, y, w, h = cv2.boundingRect(largest_contour)
# #         best_area = w * h

# #         # Draw the final bounding box on the detection image
# #         cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# #         print(f"✅ Microwave detected using Segmentation Bounding Box. Area: {best_area:.0f}")

# #     else:
# #         print("❌ Largest contour found is too small to be the microwave.")

# # else:
# #     print("❌ No large contours found in the segmentation mask.")

# # # --- END OF SEGMENTATION STRATEGY ---


# # # Display the initial frames and the final detection result
# # cv2.imshow("Original Color Image", color_image)
# # cv2.imshow("Final Microwave Detection (Bounding Box)", detection_image)
# # cv2.imshow("Depth", cv2.convertScaleAbs(depth_image, alpha=0.03)) # Displaying depth for context

# # # Wait indefinitely until a key is pressed to close the windows
# # print("\nFrame displayed. Press any key to close the windows and exit...")
# # cv2.waitKey(0) 

# # pipeline.stop()
# # cv2.destroyAllWindows()