#!/usr/bin/env python3
"""
🎥 Camera Test & Capture Tool
Test your camera and capture images easily
"""

import cv2
import os
import time
from datetime import datetime

def test_camera():
    """Test camera and capture images"""
    print("🎥 Camera Test & Capture Tool")
    print("=" * 40)
    
    # Create captures folder
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    # Test camera initialization
    print("🔍 Testing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Camera not detected!")
        print("\n🔧 Troubleshooting:")
        print("• Close other camera apps (Zoom, Skype, Teams)")
        print("• Check camera permissions")
        print("• Try different USB port")
        return False
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera detected successfully!")
    print(f"📊 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    print("\n📸 Controls:")
    print("• Press SPACE to capture image")
    print("• Press 'q' to quit")
    print("• Press 'c' for continuous capture mode")
    
    capture_count = 0
    continuous_mode = False
    last_capture_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Display frame
        cv2.imshow('Camera Test - Press SPACE to capture, q to quit', frame)
        
        # Continuous capture mode (every 2 seconds)
        if continuous_mode:
            current_time = time.time()
            if current_time - last_capture_time >= 2.0:
                capture_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'captures/auto_capture_{capture_count}_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Auto-captured: {filename}")
                last_capture_time = current_time
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            capture_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'captures/manual_capture_{capture_count}_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Captured: {filename}")
        elif key == ord('c'):
            continuous_mode = not continuous_mode
            if continuous_mode:
                print("🔄 Continuous capture mode ON (every 2 seconds)")
                last_capture_time = time.time()
            else:
                print("⏹️ Continuous capture mode OFF")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test completed! Captured {capture_count} images")
    print("📂 Images saved in 'captures/' folder")
    return True

if __name__ == "__main__":
    test_camera()
