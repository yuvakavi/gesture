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
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Animated gradient background
    t = time.time()
    for i in range(480):
        for j in range(640):
            frame[i, j] = [
                int(120 + 60 * np.sin(i * 0.02 + t)),
                int(160 + 60 * np.cos(j * 0.02 + t)), 
                int(200 + 60 * np.sin((i+j) * 0.015 + t))
            ]
    
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
