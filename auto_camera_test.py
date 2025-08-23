import streamlit as st
import cv2
import numpy as np
import time

def auto_init_camera():
    """Automatically initialize camera without user interaction"""
    try:
        # Suppress OpenCV warnings
        import os
        os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
        
        # Try camera initialization with different backends
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    # Set properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test frame capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                cap.release()
            except:
                continue
        return None
    except:
        return None

# Simple test
if __name__ == "__main__":
    st.title("🚀 Auto Camera Capture")
    
    # Initialize camera automatically
    if 'cap' not in st.session_state:
        with st.spinner("🔍 Auto-detecting camera..."):
            st.session_state.cap = auto_init_camera()
    
    if st.session_state.cap is not None:
        st.success("✅ Camera auto-initialized successfully!")
        
        # Auto-start camera capture
        camera_placeholder = st.empty()
        
        if st.button("📹 Start Auto Capture"):
            for i in range(30):  # Capture 30 frames automatically
                ret, frame = st.session_state.cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)  # Mirror effect
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    time.sleep(0.1)  # 10 FPS
                else:
                    break
            st.success("✅ Auto-capture complete!")
    else:
        st.info("🎬 No camera found - Using simulation mode")
        
        # Show simulation
        simulation_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        simulation_frame[:] = [100, 150, 200]  # Blue background
        cv2.putText(simulation_frame, "AUTO SIMULATION MODE", (120, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        st.image(simulation_frame, channels="RGB", use_container_width=True)
        st.info("✅ Auto-simulation mode active!")
