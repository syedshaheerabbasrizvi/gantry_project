import pyrealsense2 as rs
import numpy as np
import cv2

def comprehensive_test():
    """Complete camera test that checks everything"""
    print("=" * 60)
    print("🔍 COMPREHENSIVE RealSense D435i TEST")
    print("=" * 60)
    
    pipeline = None
    try:
        # Test 1: Basic initialization
        print("1. Initializing pipeline...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Test 2: Stream configuration
        print("2. Configuring streams...")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Test 3: Pipeline start
        print("3. Starting pipeline...")
        pipeline_profile = pipeline.start(config)
        print("   ✅ Pipeline started!")
        
        # Test 4: Device info
        device = pipeline_profile.get_device()
        print(f"4. Device Info:")
        print(f"   📷 Name: {device.get_info(rs.camera_info.name)}")
        print(f"   🔢 Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"   ⚙️  Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Test 5: Frame capture
        print("5. Capturing frames...")
        frames = pipeline.wait_for_frames(5000)  # 5 second timeout
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame:
            print("   ❌ No color frame received!")
            return False
        if not depth_frame:
            print("   ❌ No depth frame received!")
            return False
            
        print("   ✅ Both color and depth frames received!")
        
        # Test 6: Convert to numpy arrays
        print("6. Converting to numpy arrays...")
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        print(f"   Color shape: {color_image.shape}")
        print(f"   Depth shape: {depth_image.shape}")
        
        # Test 7: Display test
        print("7. Testing display...")
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Show both images
        cv2.imshow('Color Feed - Press any key', color_image)
        cv2.imshow('Depth Feed - Press any key', depth_colormap)
        print("   ✅ Display windows opened - check your screen!")
        
        # Test 8: Depth measurement
        center_x, center_y = 320, 240  # Center of image
        depth_value = depth_frame.get_distance(center_x, center_y)
        print(f"8. Depth measurement test:")
        print(f"   📏 Center depth: {depth_value:.3f} meters")
        
        # Wait for key press
        print("\n🎯 TEST COMPLETE! Check the display windows.")
        print("   Press any key in the display window to exit...")
        cv2.waitKey(0)
        
        print("✅ ALL TESTS PASSED! Your camera is ready for YOLO!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False
        
    finally:
        # Cleanup
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("\n🧹 Resources cleaned up.")

if __name__ == "__main__":
    success = comprehensive_test()
    
    if success:
        print("\n🎉 🎉 🎉 CAMERA IS WORKING PERFECTLY! 🎉 🎉 🎉")
        print("You can now run the full YOLO + RealSense code!")
    else:
        print("\n🔧 Please check camera connection and try again.")