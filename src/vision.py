import numpy as np
import cv2

class VisionPipeline:
    def __init__(self, upper_brightness_limit=50, min_contour_area=500):
        self.upper_brightness_limit = upper_brightness_limit
        self.min_contour_area = min_contour_area

    def process_vision(self, img):
        color_image = img
        # Convert to HSV and extract brightness channel
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv_image)

        # Threshold for dark regions (microwave)
        dark_mask = cv2.inRange(V, 0, self.upper_brightness_limit)

        # Clean the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours and compute the bounding box
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        microwave_bbox = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                microwave_bbox = (x, y, w, h)
        
        # If bounding box is detected, estimate orientation using Hough Transform
        dominant_angle = None
        if microwave_bbox is not None:
            x, y, w, h = microwave_bbox
            roi_mask = cleaned_mask[y:y+h, x:x+w]
            roi_color = color_image[y:y+h, x:x+w]

            # Step 1: Edge detection
            edges = cv2.Canny(roi_mask, 50, 150, apertureSize=3)

            # Step 2: Detect lines using Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

            if lines is not None:
                angles = [np.degrees(theta) for rho, theta in lines[:, 0]]
                angles = [a if a <= 180 else a - 180 for a in angles]
                dominant_angle = np.median(angles)

        return dominant_angle

# Usage example:
# Initialize the VisionPipeline class
vision = VisionPipeline()

# Now you can call the process_vision method to process an image
# Assuming `image` is a frame captured from your simulation or real-time input
image = cv2.imread("./new_image.png")  # Or fetch it from a simulation
if image is not None:
    # 1. Display the image
    cv2.imshow("Input Image", image)

    # 2. Wait for a key press (The crucial step!)
    #    '0' means wait indefinitely until any key is pressed.
    cv2.waitKey(0) 

    # 3. Clean up the created windows
    cv2.destroyAllWindows()
else:
    print("🚨 Error: Image was not loaded. Check the file path/integrity.")
# rotation = vision.process_vision(image)

# if rotation is not None:
#     print(f"Detected rotation: {rotation}°")
# else:
#     print("No microwave detected.")
