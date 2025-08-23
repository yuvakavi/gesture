#!/usr/bin/env python3
"""
🌐 Streamlit Cloud Configuration and Compatibility Module
Ensures optimal performance and functionality on Streamlit Cloud
"""

import os
import streamlit as st
import numpy as np
import time

def detect_cloud_environment():
    """Detect if running on Streamlit Cloud or similar cloud platforms"""
    cloud_indicators = [
        'STREAMLIT_SHARING' in os.environ,
        'STREAMLIT_CLOUD' in os.environ,
        '/mount/src' in os.getcwd(),
        '/app' in os.getcwd(),
        'streamlit.io' in os.environ.get('HOSTNAME', ''),
        'heroku' in os.environ.get('DYNO', ''),
        os.path.exists('/.dockerenv'),
        'CODESPACE_NAME' in os.environ,
        'GITHUB_ACTIONS' in os.environ,
        'RENDER' in os.environ.get('RENDER', ''),
        'RAILWAY_ENVIRONMENT' in os.environ
    ]
    return any(cloud_indicators)

def configure_for_cloud():
    """Configure app settings for optimal cloud performance"""
    if detect_cloud_environment():
        # Disable camera features
        st.session_state.camera_simulation = True
        st.session_state.cap = None
        
        # Configure for cloud environment
        os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        
        return True
    return False

def create_simulation_frame():
    """Create an animated simulation frame for cloud deployment"""
    # Create frame more efficiently using numpy broadcasting
    height, width = 480, 640
    t = time.time()
    
    # Create coordinate grids
    i = np.arange(height).reshape(-1, 1)  # Column vector
    j = np.arange(width).reshape(1, -1)   # Row vector
    
    # Calculate color channels using vectorized operations with broadcasting
    r = np.clip(120 + 60 * np.sin(i * 0.02 + t), 0, 255).astype(np.uint8)
    g = np.clip(160 + 60 * np.cos(j * 0.02 + t), 0, 255).astype(np.uint8)
    b = np.clip(200 + 60 * np.sin((i + j) * 0.015 + t), 0, 255).astype(np.uint8)
    
    # Broadcast to full frame size and stack
    r_full = np.broadcast_to(r, (height, width))
    g_full = np.broadcast_to(g, (height, width))
    b_full = np.broadcast_to(b, (height, width))
    
    # Stack channels to create RGB frame
    frame = np.stack([r_full, g_full, b_full], axis=2)
    
    # Add some animated elements to make it more realistic
    center_x, center_y = width // 2, height // 2
    radius = int(50 + 20 * np.sin(t * 2))
    
    # Draw animated circle (simulating hand detection area)
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
    frame[mask] = [255, 200, 100]  # Orange circle for hand detection simulation
    
    return frame

def simulate_gesture_recognition():
    """Simulate gesture recognition for cloud deployment"""
    gestures = [
        {"name": "Open Hand", "confidence": 0.95, "translation": "விரித்த கை"},
        {"name": "Thumbs Up", "confidence": 0.92, "translation": "நல்லது"},
        {"name": "Peace Sign", "confidence": 0.88, "translation": "சமாதானம்"},
        {"name": "Pointing", "confidence": 0.90, "translation": "சுட்டிக்காட்டுதல்"},
        {"name": "Fist", "confidence": 0.87, "translation": "முட்டி"}
    ]
    
    gesture = np.random.choice(gestures)
    return {
        'original': gesture["name"],
        'translated': gesture["translation"],
        'confidence': gesture["confidence"],
        'timestamp': time.time(),
        'method': 'cloud_simulation'
    }

def cloud_compatible_tts(text, language='en'):
    """Cloud-compatible text-to-speech with visual feedback"""
    # Since TTS may not work in cloud, provide visual feedback
    st.markdown(f"""
    <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 15px; border-radius: 10px; color: white; margin: 10px 0; animation: pulse 2s;">
        🔊 <strong>Audio Output:</strong> {text}
        <br><small>🌐 Cloud Mode: Visual audio feedback</small>
    </div>
    """, unsafe_allow_html=True)
    
    return True

def show_cloud_info():
    """Display cloud deployment information"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
        <h3 style="margin: 0; text-align: center;">🌐 Streamlit Cloud Deployment</h3>
        <p style="margin: 10px 0; text-align: center;">Optimized for cloud performance with full functionality</p>
        <div style="display: flex; justify-content: space-around; margin: 15px 0;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🎭</div>
                <small>Gesture Simulation</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🔊</div>
                <small>Visual Audio</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🌍</div>
                <small>Translation</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">💬</div>
                <small>Text Input</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
