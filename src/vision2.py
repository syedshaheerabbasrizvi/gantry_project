# ------------------------final MinRect version to be imported as library------------------------


import numpy as np
import cv2

class VisionPipeline:
    def __init__(self, upper_brightness_limit=50, min_contour_area=500):
        self.upper_brightness_limit = upper_brightness_limit
        self.min_contour_area = min_contour_area

    def process_vision(self, img):
        color_image = img.copy() 
        
        # --- Stage 1 & 2: Color Conversion and Thresholding ---
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        _, _, V = cv2.split(hsv_image)
        dark_mask = cv2.inRange(V, 0, self.upper_brightness_limit)

        # --- Stage 3: Morphological Cleaning ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)        

        # --- Stage 4: Contour Detection and Initialization ---
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dominant_angle, cx, cy = None, None, None  

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                # --- Stage 5: Orientation using Minimum Area Rectangle (MAR) ---
                
                # 1. Fit the minimum area rotated rectangle
                rect = cv2.minAreaRect(largest_contour)
                (cx_float, cy_float), (w, h), angle_raw = rect
                
                # Convert center coordinates to integers for drawing functions (Fixes error)
                cx, cy = int(cx_float), int(cy_float)

                # Adjust angle to a more intuitive representation
                angle_raw = abs(90 - angle_raw) 
                
                
                # 2. Normalize the Angle: Adjust OpenCV's angle to the standard 0-180 range
                # OpenCV's minAreaRect angle convention can be tricky. A common normalization:
                if w < h: 
                    # If height (h) is the longest side, angle is measured from the Y-axis. 
                    # Add 90 to reference the X-axis.
                    dominant_angle = angle_raw + 90
                else: 
                    # If width (w) is the longest side, angle is measured from the X-axis.
                    dominant_angle = angle_raw

        return (dominant_angle, cx, cy)












# -------------------------Vision HOG based method------------------------


# import numpy as np
# import cv2

# class VisionPipeline:
#     def __init__(self, upper_brightness_limit=50, min_contour_area=500):
#         self.upper_brightness_limit = upper_brightness_limit
#         self.min_contour_area = min_contour_area

#     def process_vision(self, img):
#         color_image = img.copy() # Use a copy to avoid side effects
        
#         print("--- Vision Pipeline Execution ---")
        
#         # --- Stage 1: Color Conversion and Channel Split ---
#         hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
#         H, S, V = cv2.split(hsv_image)
#         # Display the V-channel (Brightness)
#         cv2.imshow("1. V-Channel (Brightness)", V)
#         cv2.waitKey(500) 
#         # 

#         # --- Stage 2: Thresholding for Dark Regions ---
#         dark_mask = cv2.inRange(V, 0, self.upper_brightness_limit)
#         # Display the initial dark mask
#         cv2.imshow(f"2. Dark Mask (V < {self.upper_brightness_limit})", dark_mask)
#         cv2.waitKey(500) 
#         # 

#         # --- Stage 3: Morphological Operations (Cleaning the Mask) ---
#         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#         # cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#         # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#         # # Display the cleaned mask
#         # cv2.imshow("3. Cleaned Mask (Morphology)", cleaned_mask)
#         # cv2.waitKey(0)
#         # 


#         # --- Stage 4: Contour Detection and Bounding Box ---
#         contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour_canvas = np.zeros((dark_mask.shape[0], dark_mask.shape[1], 3), dtype=np.uint8)
#         cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 2)
#         cv2.imshow("3. Contours Detected", contour_canvas)        
#         cv2.waitKey(500)
#         microwave_bbox = None
#         output_image_bbox = color_image.copy() # Image for displaying the bbox
        
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             if cv2.contourArea(largest_contour) > self.min_contour_area:
#                 #  cordinates from top-left corner
#                 x, y, w, h = cv2.boundingRect(largest_contour)
#                 microwave_bbox = (x, y, w, h)
#                 # Draw the bounding box
#                 cv2.rectangle(output_image_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Display the image with the bounding box
#         cv2.imshow("4. Detected Bounding Box", output_image_bbox)
#         cv2.waitKey(500)
#         # 
        
#         # --- Stage 5: Edge Detection in ROI ---
#         dominant_angle = None
#         if microwave_bbox is not None:
#             x, y, w, h = microwave_bbox
#             # roi_mask = cleaned_mask[y:y+h, x:x+w]
#             roi_mask = dark_mask[y:y+h, x:x+w]
#             roi_color = color_image[y:y+h, x:x+w]

#             # Step 5a: Edge detection
#             edges = cv2.Canny(roi_mask, 50, 150, apertureSize=3)
#             # Display the edges
#             cv2.imshow("5a. Canny Edges in ROI", edges)
#             cv2.waitKey(500)
#             # 

#             # Step 5b: Detect lines using Hough Transform (and drawing them)
#             lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
#             output_image_lines = roi_color.copy()

#             if lines is not None:
#                 # Draw detected lines for visualization
#                 for rho, theta in lines[:, 0]:
#                     a = np.cos(theta)
#                     b = np.sin(theta)
#                     x0 = a * rho
#                     y0 = b * rho
#                     x1 = int(x0 + 1000 * (-b))
#                     y1 = int(y0 + 1000 * (a))
#                     x2 = int(x0 - 1000 * (-b))
#                     y2 = int(y0 - 1000 * (a))
#                     cv2.line(output_image_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
#                 angles = [np.degrees(theta) for rho, theta in lines[:, 0]]
#                 angles = [a if a <= 180 else a - 180 for a in angles]
#                 dominant_angle = np.median(angles)
        
#             # Display the ROI with detected lines
#             cv2.imshow("5b. Hough Lines on ROI", output_image_lines)
#             cv2.waitKey(0)
#             # 


#         cv2.destroyAllWindows()
#         return dominant_angle

# # Usage example:
# vision = VisionPipeline()
# image = cv2.imread("./new_image.png")
# if image is not None:
#     cv2.imshow("Input Image", image)
#     cv2.waitKey(0) 
#     rotation = vision.process_vision(image)
#     if rotation is not None:
#         print(f"Detected rotation: {rotation:.2f}°")
#     else:
#         print("No microwave detected.")
#     cv2.destroyAllWindows()
# else:
#     print("🚨 Error: Image was not loaded. Check the file path/integrity.")









