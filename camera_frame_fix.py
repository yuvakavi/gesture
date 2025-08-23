#!/usr/bin/env python3
"""
🔧 Camera Frame Fix Solution
Solves "Camera connected but not producing valid frames" issue
"""
import cv2
import time
import numpy as np
import threading
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraFrameFixer:
    """Advanced camera frame reader with multiple fix strategies"""
    
    def __init__(self):
        self.camera = None
        self.current_backend = None
        self.frame_buffer = []
        self.is_reading = False
        self.read_thread = None
        
    def enhanced_camera_init(self, camera_index: int = 0) -> bool:
        """Enhanced camera initialization with frame validation"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Backend"),
            (cv2.CAP_V4L2, "Video4Linux2")
        ]
        
        for backend, name in backends:
            logger.info(f"🧪 Testing {name}...")
            
            try:
                # Initialize camera with backend
                cap = cv2.VideoCapture(camera_index, backend)
                
                if not cap.isOpened():
                    logger.warning(f"❌ {name}: Cannot open camera")
                    continue
                
                # Set optimal properties
                self.configure_camera_properties(cap)
                
                # Test frame reading with multiple attempts
                if self.test_frame_reading(cap, name):
                    self.camera = cap
                    self.current_backend = name
                    logger.info(f"✅ {name}: Frame reading successful!")
                    return True
                else:
                    cap.release()
                    
            except Exception as e:
                logger.error(f"❌ {name}: Exception - {e}")
                continue
        
        logger.error("❌ No working camera backend found")
        return False
    
    def configure_camera_properties(self, cap: cv2.VideoCapture):
        """Configure camera properties for optimal frame reading"""
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1),  # Reduce buffer to get latest frames
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')),
        ]
        
        for prop, value in properties:
            try:
                cap.set(prop, value)
            except:
                pass  # Some properties might not be supported
    
    def test_frame_reading(self, cap: cv2.VideoCapture, backend_name: str, max_attempts: int = 10) -> bool:
        """Test frame reading with multiple strategies"""
        logger.info(f"🔍 Testing frame reading for {backend_name}...")
        
        # Strategy 1: Direct frame reading
        for attempt in range(max_attempts):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                mean_val = np.mean(frame)
                logger.info(f"✅ Attempt {attempt + 1}: Frame {w}x{h}, Mean: {mean_val:.2f}")
                
                # Validate frame content (not just black/white)
                if self.validate_frame_content(frame):
                    return True
            else:
                logger.warning(f"❌ Attempt {attempt + 1}: No valid frame")
                time.sleep(0.1)  # Small delay between attempts
        
        # Strategy 2: Flush buffer and retry
        logger.info("🔄 Flushing camera buffer and retrying...")
        for _ in range(5):
            cap.read()  # Flush old frames
        
        for attempt in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if self.validate_frame_content(frame):
                    logger.info(f"✅ Buffer flush successful on attempt {attempt + 1}")
                    return True
            time.sleep(0.1)
        
        # Strategy 3: Re-initialize camera
        logger.info("🔄 Re-initializing camera...")
        cap.release()
        time.sleep(0.5)
        
        try:
            # Re-open with same backend
            backend_map = {
                "DirectShow": cv2.CAP_DSHOW,
                "Media Foundation": cv2.CAP_MSMF,
                "Any Backend": cv2.CAP_ANY,
                "Video4Linux2": cv2.CAP_V4L2
            }
            
            backend_id = backend_map.get(backend_name, cv2.CAP_ANY)
            cap = cv2.VideoCapture(0, backend_id)
            
            if cap.isOpened():
                self.configure_camera_properties(cap)
                time.sleep(0.5)  # Allow camera to warm up
                
                for attempt in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if self.validate_frame_content(frame):
                            logger.info(f"✅ Re-initialization successful on attempt {attempt + 1}")
                            return True
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"❌ Re-initialization failed: {e}")
        
        return False
    
    def validate_frame_content(self, frame: np.ndarray) -> bool:
        """Validate that frame contains actual image data"""
        if frame is None or frame.size == 0:
            return False
        
        # Check if frame is not completely black or white
        mean_val = np.mean(frame)
        std_val = np.std(frame)
        
        # Frame should have some variation (not solid color)
        if std_val < 5:  # Very low standard deviation indicates solid color
            logger.warning(f"⚠️ Frame appears to be solid color (std: {std_val:.2f})")
            return False
        
        # Frame should not be completely black or white
        if mean_val < 10 or mean_val > 245:
            logger.warning(f"⚠️ Frame appears to be too dark/bright (mean: {mean_val:.2f})")
            return False
        
        return True
    
    def start_threaded_reading(self):
        """Start threaded frame reading for better performance"""
        if not self.camera:
            logger.error("❌ Camera not initialized")
            return False
        
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()
        logger.info("🧵 Started threaded frame reading")
        return True
    
    def _read_frames(self):
        """Background thread for reading frames"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_reading and self.camera:
            try:
                ret, frame = self.camera.read()
                
                if ret and frame is not None and frame.size > 0:
                    if self.validate_frame_content(frame):
                        # Keep only the latest frame
                        self.frame_buffer = [frame]
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                
                # If too many consecutive failures, try to reinitialize
                if consecutive_failures >= max_failures:
                    logger.warning("⚠️ Too many consecutive frame reading failures, reinitializing...")
                    self.reinitialize_camera()
                    consecutive_failures = 0
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"❌ Frame reading error: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest valid frame"""
        if self.frame_buffer:
            return self.frame_buffer[-1].copy()
        else:
            # Fallback to direct reading
            if self.camera:
                ret, frame = self.camera.read()
                if ret and frame is not None and self.validate_frame_content(frame):
                    return frame
        return None
    
    def reinitialize_camera(self):
        """Reinitialize camera if frame reading fails"""
        if self.camera:
            self.camera.release()
            time.sleep(0.5)
        
        logger.info("🔄 Reinitializing camera...")
        if self.enhanced_camera_init():
            logger.info("✅ Camera reinitialized successfully")
        else:
            logger.error("❌ Camera reinitialization failed")
    
    def stop_reading(self):
        """Stop frame reading and cleanup"""
        self.is_reading = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        cv2.destroyAllWindows()
        logger.info("🛑 Camera reading stopped")
    
    def test_camera_fix(self):
        """Test the camera fix solution"""
        logger.info("🚀 Testing Camera Frame Fix...")
        
        if not self.enhanced_camera_init():
            logger.error("❌ Camera initialization failed")
            return False
        
        logger.info(f"✅ Camera initialized with {self.current_backend}")
        
        # Start threaded reading
        if not self.start_threaded_reading():
            return False
        
        # Test frame capture for 10 seconds
        logger.info("📸 Testing frame capture for 10 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            frame = self.get_latest_frame()
            if frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    h, w = frame.shape[:2]
                    logger.info(f"📊 Captured {frame_count} frames, Current: {w}x{h}")
            time.sleep(0.1)
        
        logger.info(f"✅ Test completed: {frame_count} valid frames captured")
        self.stop_reading()
        
        return frame_count > 0

def main():
    """Main function for testing camera frame fix"""
    print("🔧 Camera Frame Fix Solution")
    print("=" * 50)
    
    fixer = CameraFrameFixer()
    
    try:
        success = fixer.test_camera_fix()
        if success:
            print("\n✅ Camera frame fix successful!")
            print("Your camera is now producing valid frames.")
        else:
            print("\n❌ Camera frame fix failed.")
            print("Please check camera hardware and drivers.")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    finally:
        fixer.stop_reading()

if __name__ == "__main__":
    main()
