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
    # print(f"Error reading frame: {e}")
    pipeline.stop()
    exit()

if color_frame is None or depth_frame is None:
    # print("Could not retrieve aligned color or depth frame. Exiting.")
    pipeline.stop()
    exit()

# Convert frames to numpy arrays
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Display raw color + depth
# cv2.imshow("Step 1: Original Color Image", color_image)
# cv2.imshow("Step 1: Raw Depth Image", cv2.convertScaleAbs(depth_image, alpha=0.03))
# cv2.waitKey(500)

# ===============================================================
# 3️⃣  COLOR/BRIGHTNESS SEGMENTATION (ISOLATE MICROWAVE)
# ===============================================================

# Convert to HSV and extract brightness channel
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_image)

# Threshold for dark regions (microwave)
upper_brightness_limit = 50  # tune this value if needed
dark_mask = cv2.inRange(V, 0, upper_brightness_limit)

# cv2.imshow("Step 2: Initial Dark Mask", dark_mask)
# cv2.waitKey(500)

# Clean the mask with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# cv2.imshow("Step 3: Cleaned Segmentation Mask", cleaned_mask)
# cv2.waitKey(500)

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
        # print(f"✅ Microwave detected. Bounding box area = {w*h}")
    # else:
    #     print("❌ Largest contour too small to be the microwave.")
# else:
#     print("❌ No contours detected.")

# cv2.imshow("Step 4: Detected Bounding Box", bbox_image)
# cv2.waitKey(500)

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
        print("Rotation of the object: ", dominant_angle)
        # print(f"📏 Dominant edge angle (inside bounding box): {dominant_angle:.2f}°")

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
        # cv2.imshow("Step 5: ROI Hough Lines", roi_color)
        # cv2.waitKey(500)

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

        # cv2.imshow("Step 6: Global Orientation Overlay", orientation_image)
        cv2.imwrite('../results/orientation_image.jpg', orientation_image)
        # cv2.waitKey(0)

    else:
        print("⚠️ No strong edges found inside bounding box.")


# print("\nPress any key to exit...")
cv2.waitKey(0)

pipeline.stop()
cv2.destroyAllWindows()