#!/usr/bin/env python3
"""
🚀 Advanced Camera Fix & Auto-Capture Solution
Automatically detects camera, captures gestures, and provides voice output
"""

import cv2
import numpy as np
import streamlit as st
import threading
import time
import queue
import pyttsx3
from typing import Optional, Tuple, Dict, Any
import mediapipe as mp
from utils.mediapipe_helpers import get_hand_landmarks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoCameraDetector:
    """Advanced camera detection and auto-capture system"""
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.gesture_queue = queue.Queue()
        self.tts_engine = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.init_tts()
        
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Use female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            logger.info("✅ TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ TTS initialization failed: {e}")
            self.tts_engine = None
    
    def speak_text(self, text: str):
        """Convert text to speech with threading"""
        if self.tts_engine:
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    logger.error(f"Speech error: {e}")
            
            thread = threading.Thread(target=speak, daemon=True)
            thread.start()
        else:
            logger.warning(f"TTS not available, would say: {text}")
    
    def detect_camera(self) -> bool:
        """Detect and initialize camera with multiple backend support"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Backend"),
            (cv2.CAP_V4L2, "Video4Linux2")
        ]
        
        for backend, name in backends:
            try:
                logger.info(f"🔍 Trying {name}...")
                cap = cv2.VideoCapture(0, backend)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Set optimal resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        self.camera = cap
                        logger.info(f"✅ Camera initialized: {name}")
                        self.speak_text(f"Camera connected using {name}")
                        return True
                
                cap.release()
                
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
                continue
        
        logger.error("❌ No camera detected")
        self.speak_text("Camera not detected. Please check camera connection.")
        return False
    
    def detect_gesture(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect hand gestures from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_info = {
            'gesture': 'none',
            'confidence': 0.0,
            'hand_count': 0,
            'landmarks': None
        }
        
        if results.multi_hand_landmarks:
            gesture_info['hand_count'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                gesture_info['landmarks'] = landmarks
                
                # Simple gesture recognition
                gesture_info['gesture'] = self.classify_gesture(landmarks)
                gesture_info['confidence'] = 0.8
        
        return gesture_info
    
    def classify_gesture(self, landmarks: list) -> str:
        """Simple gesture classification"""
        if not landmarks or len(landmarks) < 63:
            return 'none'
        
        # Convert to numpy array and reshape
        landmarks = np.array(landmarks).reshape(-1, 3)
        
        # Simple thumb up detection
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        
        if thumb_tip[1] < thumb_mcp[1] and thumb_tip[1] < index_tip[1]:
            return 'thumbs_up'
        elif thumb_tip[1] > thumb_mcp[1] and thumb_tip[1] > index_tip[1]:
            return 'thumbs_down'
        
        # Count extended fingers
        finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
        finger_mcps = [landmarks[i] for i in [2, 5, 9, 13, 17]]
        
        extended_count = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            if tip[1] < mcp[1]:  # Tip is above MCP
                extended_count += 1
        
        gesture_map = {
            0: 'fist',
            1: 'one',
            2: 'peace',
            3: 'three',
            4: 'four',
            5: 'open_hand'
        }
        
        return gesture_map.get(extended_count, 'unknown')
    
    def start_auto_capture(self):
        """Start automatic camera capture and gesture detection"""
        if not self.detect_camera():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("🎥 Auto-capture started")
        self.speak_text("Camera auto-capture started. Show your gestures!")
        return True
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        last_gesture = 'none'
        gesture_count = 0
        
        while self.is_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect gestures
                gesture_info = self.detect_gesture(frame)
                
                # Process gesture changes
                current_gesture = gesture_info['gesture']
                if current_gesture != 'none' and current_gesture != last_gesture:
                    gesture_count += 1
                    
                    # Add gesture info to queue
                    gesture_data = {
                        'gesture': current_gesture,
                        'confidence': gesture_info['confidence'],
                        'count': gesture_count,
                        'timestamp': time.time(),
                        'frame': frame.copy()
                    }
                    
                    self.gesture_queue.put(gesture_data)
                    
                    # Provide voice feedback
                    feedback_text = self.get_gesture_feedback(current_gesture)
                    self.speak_text(feedback_text)
                    
                    logger.info(f"🤏 Gesture detected: {current_gesture} (#{gesture_count})")
                    last_gesture = current_gesture
                
                # Add text overlay
                cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {gesture_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Auto Camera - Gesture Detection', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                logger.error(f"Capture error: {e}")
                break
        
        self.stop_capture()
    
    def get_gesture_feedback(self, gesture: str) -> str:
        """Get voice feedback for detected gesture"""
        feedback_map = {
            'thumbs_up': "Great! Thumbs up detected",
            'thumbs_down': "Thumbs down gesture detected",
            'peace': "Peace sign detected",
            'fist': "Fist gesture detected",
            'open_hand': "Open hand detected",
            'one': "One finger detected",
            'three': "Three fingers detected",
            'four': "Four fingers detected",
            'unknown': "Unknown gesture detected"
        }
        
        return feedback_map.get(gesture, f"{gesture} gesture detected")
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        cv2.destroyAllWindows()
        logger.info("🛑 Camera capture stopped")
        self.speak_text("Camera capture stopped")
    
    def get_latest_gesture(self) -> Optional[Dict[str, Any]]:
        """Get the latest detected gesture"""
        try:
            return self.gesture_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_text_input(self, text: str) -> str:
        """Process text input and provide voice output"""
        if not text.strip():
            return "Please provide some text input"
        
        # Process the text (you can add more sophisticated processing here)
        response = f"Processing your input: {text}"
        
        # Provide voice output
        self.speak_text(response)
        
        return response

def main():
    """Main function for standalone testing"""
    print("🚀 Starting Advanced Camera Fix & Auto-Capture")
    
    detector = AutoCameraDetector()
    
    if detector.start_auto_capture():
        print("✅ Camera auto-capture started successfully!")
        print("📝 Instructions:")
        print("- Show gestures to the camera")
        print("- Press 'q' in the camera window to quit")
        print("- Voice feedback will announce detected gestures")
        
        try:
            # Keep main thread alive
            while detector.is_running:
                time.sleep(1)
                
                # Check for new gestures
                gesture = detector.get_latest_gesture()
                if gesture:
                    print(f"📊 New gesture: {gesture['gesture']} (confidence: {gesture['confidence']:.2f})")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping camera capture...")
        finally:
            detector.stop_capture()
    else:
        print("❌ Failed to start camera capture")

# Streamlit Integration
def run_streamlit_camera_fix():
    """Streamlit interface for camera fix"""
    st.title("🎥 Advanced Camera Fix & Auto-Capture")
    st.write("Automatic camera detection with gesture recognition and voice output")
    
    if 'detector' not in st.session_state:
        st.session_state.detector = AutoCameraDetector()
    
    detector = st.session_state.detector
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Auto-Capture"):
            if detector.start_auto_capture():
                st.success("✅ Camera auto-capture started!")
                st.info("Show gestures to your camera. Voice feedback will announce detected gestures.")
            else:
                st.error("❌ Failed to start camera capture")
    
    with col2:
        if st.button("🛑 Stop Capture"):
            detector.stop_capture()
            st.info("Camera capture stopped")
    
    # Text input section
    st.subheader("💬 Text Input with Voice Output")
    text_input = st.text_area("Enter your text:", placeholder="Type something here...")
    
    if st.button("🔊 Process & Speak"):
        if text_input:
            response = detector.process_text_input(text_input)
            st.success(f"Response: {response}")
        else:
            st.warning("Please enter some text first")
    
    # Display latest gesture
    st.subheader("🤏 Latest Gesture")
    latest_gesture = detector.get_latest_gesture()
    if latest_gesture:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gesture", latest_gesture['gesture'])
        with col2:
            st.metric("Confidence", f"{latest_gesture['confidence']:.2f}")
        with col3:
            st.metric("Count", latest_gesture['count'])

if __name__ == "__main__":
    main()
