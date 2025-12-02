import pyrealsense2 as rs
import cv2
import numpy as np
import math
import os

def regionprops_binary(binary_image):
    """
    Simulates MATLAB's regionprops function for binary images
    Returns properties like orientation, centroid, and bounding box
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (assuming it's our object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    # Calculate centroid
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Calculate orientation using second order central moments
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    
    # Calculate orientation (in radians)
    orientation = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    
    # Convert to degrees
    orientation_deg = math.degrees(orientation)
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate minimum area rectangle (rotated rectangle)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    return {
        'orientation': orientation,
        'orientation_degrees': orientation_deg,
        'centroid': (cx, cy),
        'bounding_box': (x, y, w, h),
        'min_area_rect': rect,
        'min_area_box': box,
        'contour': largest_contour
    }

def preprocess_image(color_image):
    """
    Preprocess the image and convert to binary
    """
    # Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional: Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def draw_orientation(image, props):
    """
    Draw orientation and other features on the image
    """
    if props is None:
        return image
    
    # Create a copy of the image
    display_image = image.copy()
    
    # Draw centroid
    cx, cy = props['centroid']
    cv2.circle(display_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    
    # Draw bounding box
    x, y, w, h = props['bounding_box']
    cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Draw minimum area rectangle
    cv2.drawContours(display_image, [props['min_area_box']], 0, (0, 0, 255), 2)
    
    # Draw orientation line
    length = 100
    end_x = cx + length * math.cos(props['orientation'])
    end_y = cy + length * math.sin(props['orientation'])
    
    cv2.line(display_image, (int(cx), int(cy)), 
             (int(end_x), int(end_y)), (0, 255, 255), 3)
    
    # Draw contour
    cv2.drawContours(display_image, [props['contour']], -1, (255, 255, 0), 2)
    
    # Add text information
    cv2.putText(display_image, f"Orientation: {props['orientation_degrees']:.2f}°", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_image, f"Centroid: ({cx:.1f}, {cy:.1f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_image

def read_frame_from_bag(bag_file_path, frame_number=0):
    """
    Read a specific frame from a .bag file
    
    Args:
        bag_file_path: Path to the .bag file
        frame_number: Frame number to read (0 = first frame)
    
    Returns:
        color_image: Captured color frame
    """
    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Tell config that we will use a recorded device from file
    config.enable_device_from_file(bag_file_path, repeat_playback=False)
    
    # Start pipeline from file
    pipeline.start(config)
    
    try:
        # Wait for the first frame
        frames = pipeline.wait_for_frames()
        current_frame = 0
        
        # Skip to the desired frame
        while current_frame < frame_number:
            frames = pipeline.wait_for_frames()
            current_frame += 1
            if not frames:
                print(f"Reached end of bag file at frame {current_frame}")
                return None
        
        # Get color frame
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("No color frame found in the bag file")
            return None
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image
        
    except RuntimeError as e:
        print(f"Error reading from bag file: {e}")
        return None
    finally:
        pipeline.stop()

def main():
    """
    Main function to read from bag file and analyze object orientation
    """
    # HARDCODED VALUES - CHANGE THESE AS NEEDED
    bag_file_path = os.path.expanduser("~/Documents/middle_view02.bag")  # Change 'your_file.bag' to your actual filename
    frame_number = 0  # Change this to the frame number you want to analyze
    
    print(f"Reading frame {frame_number} from bag file: {bag_file_path}")
    
    # Check if file exists
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file not found at {bag_file_path}")
        print("Please check the filename and path")
        return
    
    # Read frame from bag file
    color_image = read_frame_from_bag(bag_file_path, frame_number)
    
    if color_image is None:
        print("Failed to read frame from bag file")
        return
    
    print("Frame read successfully!")
    print(f"Image dimensions: {color_image.shape}")
    
    # Preprocess and create binary image
    print("Preprocessing image...")
    binary_image = preprocess_image(color_image)
    
    # Get region properties
    print("Analyzing object properties...")
    props = regionprops_binary(binary_image)
    
    if props is None:
        print("No objects found in the binary image")
        # Still show the images
        cv2.imshow('Original Image', color_image)
        cv2.imshow('Binary Image', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    # Display results
    print(f"Object Orientation: {props['orientation_degrees']:.2f} degrees")
    print(f"Centroid: {props['centroid']}")
    
    # Create visualization
    result_image = draw_orientation(color_image, props)
    
    # Display images
    cv2.imshow('Original Image', color_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Object Analysis', result_image)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save images if needed
    # cv2.imwrite('original_image.jpg', color_image)
    # cv2.imwrite('binary_image.jpg', binary_image)
    # cv2.imwrite('analysis_result.jpg', result_image)
    # print("Images saved as 'original_image.jpg', 'binary_image.jpg', 'analysis_result.jpg'")

if __name__ == "__main__":
    main()