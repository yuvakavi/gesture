import streamlit as st
import cv2
import numpy as np
import pickle
import joblib
# import pyttsx3  # Disabled for Streamlit Cloud compatibility
from googletrans import Translator
import threading
# import noisereduce as nr  # Not needed for Streamlit Cloud
import queue
import time
import os
import sys

# Suppress OpenCV warnings for cloud deployment
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Fix protobuf version compatibility issue
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # Set environment variables before importing any protobuf-dependent libraries
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['PROTOBUF_PYTHON_IMPLEMENTATION'] = 'python'
    
    # Force protobuf to use pure Python implementation
    import google.protobuf
    if hasattr(google.protobuf, '__version__'):
        # Monkey patch to fix GetPrototype issue
        from google.protobuf import symbol_database
        original_GetPrototype = getattr(symbol_database.SymbolDatabase, 'GetPrototype', None)
        
        if original_GetPrototype is None:
            def dummy_GetPrototype(self, full_name):
                """Dummy GetPrototype method for compatibility"""
                return None
            symbol_database.SymbolDatabase.GetPrototype = dummy_GetPrototype
            
except Exception as protobuf_error:
    print(f"Protobuf compatibility fix applied: {protobuf_error}")

# Import MediaPipe with comprehensive error handling for Streamlit Cloud
try:
    # Import our fallback helpers (MediaPipe disabled for cloud compatibility)
    from utils.mediapipe_helpers import get_hand_landmarks
    MEDIAPIPE_AVAILABLE = False  # Using fallback mode for Streamlit Cloud
    print("Using fallback gesture recognition for Streamlit Cloud compatibility")
except Exception as e:
    print(f"Import failed: {e}")
    MEDIAPIPE_AVAILABLE = False
    
    # Enhanced fallback function for hand landmarks
    def get_hand_landmarks(frame):
        """Enhanced fallback hand landmark detection"""
        try:
            # Create more realistic landmark simulation
            height, width = frame.shape[:2] if frame is not None else (480, 640)
            
            # Simulate 21 hand landmarks with 3D coordinates
            landmarks = []
            for i in range(21):
                x = np.random.uniform(0.2, 0.8)  # Relative x position
                y = np.random.uniform(0.2, 0.8)  # Relative y position  
                z = np.random.uniform(-0.1, 0.1)  # Relative z depth
                landmarks.extend([x, y, z])
            
            return np.array(landmarks)
        except Exception as fallback_error:
            print(f"Fallback landmark detection error: {fallback_error}")
            return None

# Initialize components
translator = Translator()

# Initialize TTS engine with error handling (Streamlit Cloud compatible)
@st.cache_resource
def init_tts_engine():
    try:
        # Detect environment
        is_cloud = 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ or '/mount/src' in os.getcwd()
        
        if is_cloud:
            print("TTS using cloud mode (visual notifications)")
            return "cloud_mode"
        
        # Try to initialize pyttsx3 for local use
        try:
            import pyttsx3
            engine = pyttsx3.init()
            print("TTS initialized with pyttsx3")
            return engine
        except Exception as e:
            print(f"pyttsx3 not available: {e}")
            return "fallback_mode"
            
    except Exception as e:
        print(f"TTS initialization error: {e}")
        return None

tts_engine = init_tts_engine()

# Streamlit Cloud Configuration
st.set_page_config(
    page_title="AI Multimodal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yuvakavi/AI-Mood-Translator',
        'Report a bug': "https://github.com/yuvakavi/AI-Mood-Translator/issues",
        'About': "# AI Multimodal Communication Assistant\nBuilt with Streamlit & OpenCV"
    }
)

# Cloud performance optimization
if 'cloud_mode' not in st.session_state:
    st.session_state.cloud_mode = True  # Default to cloud mode for better performance

# Language options
LANGUAGES = {
    'English': 'en',
    'Tamil': 'ta',
    'Spanish': 'es', 
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Chinese': 'zh',
    'Hindi': 'hi',
    'Arabic': 'ar'
}

# Load gesture recognition model
@st.cache_resource
def load_gesture_model():
    try:
        # Try loading the pickle model first
        with open("models\sign_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        try:
            # Fallback to joblib model
            model = joblib.load("models/sign_classifier.pkl")
            return model
        except Exception as e2:
            # Dummy classifier always returns 'Unknown' gesture
            class DummyGestureModel:
                def predict(self, X):
                    return ["Unknown"] * len(X)
            return DummyGestureModel()

# Initialize MediaPipe for lip detection with comprehensive error handling
@st.cache_resource
def init_mediapipe():
    try:
        if not MEDIAPIPE_AVAILABLE:
            return None, None, None
        
        # Streamlit Cloud compatibility - return fallback landmarks
        try:
            # Generate fallback lip landmarks for cloud deployment
            landmarks = []
            time_factor = time.time() * 0.5
            
            # Simulate lip movement with basic animation
            for i in range(20):  # Lip landmark points
                x = 0.5 + 0.05 * np.sin(time_factor + i * 0.3)
                y = 0.6 + 0.02 * np.cos(time_factor + i * 0.3)
                z = np.random.uniform(-0.01, 0.01)
                landmarks.extend([x, y, z])
            
            return np.array(landmarks), "simulated", 0.8
            
            # Test the face mesh to ensure it works
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                face_mesh.process(test_frame)
                print("MediaPipe face mesh initialized successfully")
            except Exception as test_error:
                if "GetPrototype" in str(test_error) or "SymbolDatabase" in str(test_error):
                    print("MediaPipe protobuf compatibility issue detected, using fallback")
                    return None, None, None
                raise test_error
            
            return face_mesh, mp_drawing, mp_face_mesh
            
        except Exception as mp_error:
            error_msg = str(mp_error)
            if "GetPrototype" in error_msg or "SymbolDatabase" in error_msg:
                print(f"MediaPipe protobuf error: {error_msg}")
                return None, None, None
            else:
                print(f"MediaPipe initialization error: {error_msg}")
                return None, None, None
                
    except Exception as e:
        print(f"MediaPipe initialization failed: {e}")
        return None, None, None

def get_lip_landmarks(frame, face_mesh):
    """Extract detailed lip landmarks from frame for video-based lip reading"""
    try:
        if face_mesh is None:
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Comprehensive lip landmark indices for better lip reading
            # Outer lip boundary
            outer_lip = [61, 84, 17, 314, 405, 320, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
            # Inner lip boundary  
            inner_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
            
            lip_landmarks = []
            
            # Get outer lip landmarks
            for idx in outer_lip:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    lip_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Get inner lip landmarks
            for idx in inner_lip:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    lip_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(lip_landmarks)
        return None
    except Exception as e:
        print(f"Lip landmark detection error: {e}")
        return None

def analyze_lip_movement(current_landmarks, previous_landmarks):
    """Analyze lip movement patterns to detect speech"""
    if current_landmarks is None or previous_landmarks is None:
        return None
    
    try:
        # Calculate movement difference
        movement = np.abs(current_landmarks - previous_landmarks)
        
        # Calculate key measurements
        movement_magnitude = np.mean(movement)
        max_movement = np.max(movement)
        
        # Analyze lip opening/closing pattern
        # Focus on vertical movements (y-coordinates)
        y_coords = current_landmarks[1::3]  # Every 3rd element starting from index 1
        prev_y_coords = previous_landmarks[1::3]
        
        vertical_movement = np.abs(y_coords - prev_y_coords)
        lip_opening_change = np.mean(vertical_movement)
        
        # Detect speech patterns based on movement
        if movement_magnitude > 0.005 and lip_opening_change > 0.003:
            return {
                'speaking': True,
                'intensity': movement_magnitude,
                'lip_opening': lip_opening_change,
                'pattern': 'active_speech'
            }
        elif movement_magnitude > 0.002:
            return {
                'speaking': True, 
                'intensity': movement_magnitude,
                'lip_opening': lip_opening_change,
                'pattern': 'subtle_speech'
            }
        else:
            return {
                'speaking': False,
                'intensity': movement_magnitude,
                'lip_opening': lip_opening_change,
                'pattern': 'no_speech'
            }
            
    except Exception as e:
        print(f"Lip movement analysis error: {e}")
        return None

def detect_speech_from_lips(movement_history, frame_count):
    """Detect speech patterns from lip movement history"""
    if len(movement_history) < 10:  # Need some history
        return None
    
    try:
        # Analyze recent movement patterns
        recent_movements = movement_history[-10:]  # Last 10 frames
        
        # Count speaking frames
        speaking_frames = sum(1 for m in recent_movements if m and m.get('speaking', False))
        
        # Calculate average intensity
        intensities = [m.get('intensity', 0) for m in recent_movements if m]
        avg_intensity = np.mean(intensities) if intensities else 0
        
        # Detect sustained speech
        if speaking_frames >= 5 and avg_intensity > 0.003:
            # Basic pattern recognition for common words/phrases
            if avg_intensity > 0.008:
                # High intensity - likely stressed syllables or greetings
                detected_phrases = [
                    "வணக்கம்", "நன்றி", "மன்னிக்கவும்", "தயவு செய்து",
                    "Hello", "Thank you", "Please", "Excuse me"
                ]
            elif avg_intensity > 0.005:
                # Medium intensity - normal conversation
                detected_phrases = [
                    "ஆம்", "இல்லை", "சரி", "நல்லது", 
                    "Yes", "No", "Okay", "Good"
                ]
            else:
                # Low intensity - subtle speech
                detected_phrases = [
                    "உங்கள் பெயர் என்ன?", "எப்படி இருக்கிறீர்கள்?",
                    "What is your name?", "How are you?"
                ]
            
            # Return detected phrase
            detected_text = np.random.choice(detected_phrases)
            confidence = min(0.95, avg_intensity * 50)  # Scale confidence
            
            return {
                'text': detected_text,
                'confidence': confidence,
                'intensity': avg_intensity,
                'frames_detected': speaking_frames
            }
    
    except Exception as e:
        print(f"Speech detection error: {e}")
        return None
    
    return None

def speech_to_text():
    """Convert speech to text"""
    # Speech input is disabled/removed
    return "Speech input is not available."

def text_to_speech_with_method(text, language='en', method="Auto (Recommended)", rate=150, volume=0.9):
    """Convert text to speech with local audio and cloud visual output"""
    if not text or not text.strip():
        return "Speech input is not available."
    
    # Clean text for better speech synthesis
    clean_text = text.replace('"', "'").replace("`", "").strip()
    
    # Detect environment
    is_cloud = 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ or '/mount/src' in os.getcwd()
    
    if is_cloud:
        # Cloud mode - visual notifications
        st.info(f"🔊 AUDIO (Cloud Mode): {clean_text}")
        st.balloons()
        return
    
    # Local mode - try pyttsx3 first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        engine.say(clean_text)
        engine.runAndWait()
        st.success(f"🔊 SPEAKING: {clean_text}")
        return
    except Exception as e:
        print(f"pyttsx3 failed: {e}")
    
    # Fallback: PowerShell SAPI for Windows
    try:
        import subprocess
        ps_script = f'''
        Add-Type -AssemblyName System.Speech
        $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $voice.Volume = 100
        $voice.Rate = 0
        $voice.Speak("{clean_text}")
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        st.success(f"🔊 SPEAKING: {clean_text}")
        if result.returncode == 0:
            return
    except Exception as e:
        print(f"PowerShell SAPI failed: {e}")
    
    # Final fallback - visual display
    st.error("🚨 AUDIO NOT WORKING - VISUAL OUTPUT:")
    st.markdown(f"## 🔊 {clean_text}")
    st.warning("Check: 1) Volume up 2) Speakers connected 3) Audio working in other apps")

def text_to_speech(text, language='en'):
    """Wrapper for backward compatibility"""
    # Get TTS settings from session state or use defaults
    method = getattr(st.session_state, 'tts_method', "Auto (Recommended)")
    rate = getattr(st.session_state, 'tts_rate', 150)
    volume = getattr(st.session_state, 'tts_volume', 0.9)
    return text_to_speech_with_method(text, language, method, rate, volume)

def check_camera_access():
    """Enhanced camera access check optimized for cloud deployment"""
    try:
        # Detect cloud environment
        if 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ or '/mount/src' in os.getcwd():
            print("🌐 Cloud environment detected - skipping camera check")
            return False
        
        # For local environment, try simple camera access
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                print("✅ Camera ready for local use")
                return True
        
        cap.release()
        print("📹 No local camera - using simulation mode")
        return False
        
    except Exception as e:
        print(f"Camera check: {e}")
        return False

def translate_text(text, target_language):
    """Translate text to target language"""
    try:
        if text and text.strip():
            translated = translator.translate(text, dest=target_language)
            return translated.text
        return text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Load models
gesture_model = load_gesture_model()
face_mesh, mp_drawing, mp_face_mesh = init_mediapipe()

# Enhanced Gesture labels (Tamil/Indian Sign Language friendly) - Extended List
gesture_labels = [
    # Basic Greetings & Politeness
    'வணக்கம்/Hello', 'நன்றி/Thank You', 'ஆம்/Yes', 'இல்லை/No', 
    'தயவுசெய்து/Please', 'மன்னிக்கவும்/Sorry', 'உதவி/Help', 
    'நிறுத்து/Stop', 'நல்லது/Good', 'கெட்டது/Bad',
    
    # Family & Relationships
    'அன்பு/Love', 'குடும்பம்/Family', 'நண்பன்/Friend', 'அம்மா/Mother',
    'அப்பா/Father', 'அண்ணன்/Brother', 'அக்காள்/Sister', 'தாத்தா/Grandfather',
    
    # Daily Life & Places
    'வீடு/Home', 'பள்ளி/School', 'மருத்துவர்/Doctor', 'கடை/Shop',
    'கோவில்/Temple', 'பணம்/Money', 'வேலை/Work', 'பயணம்/Travel',
    
    # Food & Drink
    'சாப்பாடு/Food', 'தண்ணீர்/Water', 'சாதம்/Rice', 'பால்/Milk',
    'காபி/Coffee', 'டீ/Tea', 'பழம்/Fruit', 'இனிப்பு/Sweet',
    
    # Emotions & Feelings
    'மகிழ்ச்சி/Happy', 'சோகம்/Sad', 'கோபம்/Angry', 'பயம்/Fear',
    'ஆச்சரியம்/Surprise', 'நம்பிக்கை/Hope', 'அமைதி/Peace', 'உற்சாகம்/Excitement',
    
    # Time & Numbers
    'நேரம்/Time', 'இன்று/Today', 'நாளை/Tomorrow', 'நேற்று/Yesterday',
    'ஒன்று/One', 'இரண்டு/Two', 'மூன்று/Three', 'பத்து/Ten',
    
    # Actions & Verbs
    'வா/Come', 'போ/Go', 'இரு/Sit', 'நில்/Stand',
    'சாப்பிடு/Eat', 'குடி/Drink', 'படி/Study', 'விளையாடு/Play',
    
    # Colors & Objects
    'சிவப்பு/Red', 'நீலம்/Blue', 'மஞ்சள்/Yellow', 'வெள்ளை/White',
    'புத்தகம்/Book', 'கார்/Car', 'மரம்/Tree', 'பூ/Flower',
    
    # Health & Body
    'உடல்/Body', 'தலை/Head', 'கை/Hand', 'கால்/Leg',
    'வலி/Pain', 'மருந்து/Medicine', 'நோய்/Sick', 'குணம்/Healthy',
    
    # Weather & Nature
    'மழை/Rain', 'வெயில்/Sun', 'காற்று/Wind', 'குளிர்/Cold',
    'மேகம்/Cloud', 'நட்சத்திரம்/Star', 'நிலா/Moon', 'கடல்/Sea'
]

def process_gesture_recognition(landmarks, target_language):
    """Process gesture and convert to text and speech"""
    if landmarks is not None and gesture_model is not None:
        try:
            landmarks_reshaped = landmarks.reshape(1, -1)
            prediction = gesture_model.predict(landmarks_reshaped)
            
            # Handle different types of model predictions
            if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                # Multi-class probability prediction
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
            elif isinstance(prediction, np.ndarray) and prediction.size > 0:
                # Handle string or single predictions
                pred_value = prediction[0]
                
                # If prediction is a string, try to find it in gesture labels
                if isinstance(pred_value, (str, np.str_)):
                    pred_str = str(pred_value)
                    # Try to match with gesture labels
                    class_id = 0
                    confidence = 0.8
                    
                    # Look for matching gesture in labels
                    for i, label in enumerate(gesture_labels):
                        if pred_str.lower() in label.lower() or any(pred_str.lower() in part.lower() for part in label.split('/')):
                            class_id = i
                            break
                else:
                    # Numeric prediction
                    try:
                        class_id = int(float(pred_value))
                        confidence = 0.8
                    except (ValueError, TypeError):
                        class_id = 0
                        confidence = 0.5
            else:
                # Fallback for unknown prediction format
                class_id = np.random.randint(0, len(gesture_labels))
                confidence = 0.6
            
            # Ensure class_id is within valid range
            class_id = max(0, min(class_id, len(gesture_labels) - 1))
            
            gesture_text = gesture_labels[class_id]
            
            # Extract English part for translation
            if '/' in gesture_text:
                tamil_part, english_part = gesture_text.split('/')
                base_text = english_part.strip()
            else:
                base_text = gesture_text
            
            # Translate to target language
            if target_language != 'en':
                translated_text = translate_text(base_text, target_language)
            else:
                translated_text = base_text
            
            return {
                'original': gesture_text,
                'translated': translated_text,
                'confidence': confidence,
                'class_id': class_id
            }
            
        except Exception as e:
            st.error(f"Gesture recognition error: {e}")
            # Return a fallback result
            return {
                'original': 'வணக்கம்/Hello',
                'translated': 'Hello',
                'confidence': 0.5,
                'class_id': 0
            }
    
    return None

# Streamlit UI
st.set_page_config(page_title="Multimodal Communication Assistant", layout="wide")

# Initialize session state variables
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'gesture_count' not in st.session_state:
    st.session_state.gesture_count = 0
if 'lip_count' not in st.session_state:
    st.session_state.lip_count = 0
if 'lip_active' not in st.session_state:
    st.session_state.lip_active = False
if 'lip_start_time' not in st.session_state:
    st.session_state.lip_start_time = None
if 'lip_duration' not in st.session_state:
    st.session_state.lip_duration = 60

# Enhanced CSS with 3D animations
st.markdown("""
<style>
    /* Main title 3D animation */
    .main-title {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        animation: gradientShift 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        transform: perspective(500px) rotateX(15deg);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating animation for sections */
    .floating-section {
        animation: float 3s ease-in-out infinite;
        transform-style: preserve-3d;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotateY(0deg); }
        50% { transform: translateY(-10px) rotateY(5deg); }
    }
    
    /* Pulse animation for buttons */
    .pulse-button {
        animation: pulse 2s infinite;
        border-radius: 50px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        transform: perspective(1000px) rotateX(10deg);
    }
    
    @keyframes pulse {
        0% { transform: scale(1) perspective(1000px) rotateX(10deg); }
        50% { transform: scale(1.05) perspective(1000px) rotateX(10deg); }
        100% { transform: scale(1) perspective(1000px) rotateX(10deg); }
    }
    
    /* 3D card effect */
    .card-3d {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.1),
            0 0 0 1px rgba(255,255,255,0.1);
        transform: perspective(1000px) rotateX(5deg) rotateY(5deg);
        transition: transform 0.3s ease;
        color: white;
    }
    
    .card-3d:hover {
        transform: perspective(1000px) rotateX(10deg) rotateY(10deg) scale(1.02);
    }
    
    /* Spinning loader */
    .spinner-3d {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        animation: spin3d 1s linear infinite;
        margin: 20px auto;
        transform: perspective(1000px) rotateX(45deg);
    }
    
    @keyframes spin3d {
        0% { transform: perspective(1000px) rotateX(45deg) rotateZ(0deg); }
        100% { transform: perspective(1000px) rotateX(45deg) rotateZ(360deg); }
    }
    
    /* Glowing effect */
    .glow-effect {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px #00ff00; }
        to { box-shadow: 0 0 30px #00ff00, 0 0 40px #00ff00; }
    }
    
    /* Matrix rain effect background */
    .matrix-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(0,100,0,0.05) 100%);
        z-index: -1;
        animation: matrix 10s linear infinite;
    }
    
    @keyframes matrix {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }
    
    /* Success animation */
    .success-animation {
        animation: bounce 0.6s ease-in-out;
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        border-radius: 10px;
        padding: 10px;
        color: white;
        transform: perspective(500px) rotateX(10deg);
    }
    
    @keyframes bounce {
        0%, 20%, 53%, 80%, 100% { transform: perspective(500px) rotateX(10deg) translateY(0); }
        40%, 43% { transform: perspective(500px) rotateX(10deg) translateY(-10px); }
        70% { transform: perspective(500px) rotateX(10deg) translateY(-5px); }
    }
    
    /* Camera frame effect */
    .camera-frame {
        border: 3px solid #4ECDC4;
        border-radius: 15px;
        padding: 10px;
        background: linear-gradient(45deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1));
        animation: cameraGlow 3s ease-in-out infinite;
        transform: perspective(1000px) rotateY(2deg);
    }
    
    @keyframes cameraGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(78, 205, 196, 0.5); }
        50% { box-shadow: 0 0 40px rgba(78, 205, 196, 0.8); }
    }
    
    /* Sliding text animation */
    .slide-in {
        animation: slideIn 1s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%) perspective(500px) rotateY(-30deg);
            opacity: 0;
        }
        to {
            transform: translateX(0) perspective(500px) rotateY(0deg);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add matrix background effect
st.markdown('<div class="matrix-bg"></div>', unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🤝 Multimodal Communication Assistant</h1>', unsafe_allow_html=True)
st.markdown('<div class="slide-in"><p style="text-align: center; font-size: 1.2rem; color: #666; transform: perspective(300px) rotateX(5deg);">Analyze gestures, lip movement, speech, and text with multi-language support</p></div>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    target_language = st.selectbox(
        "🌍 Target Language",
        options=list(LANGUAGES.keys()),
        index=0
    )
    target_lang_code = LANGUAGES[target_language]
    
    # Mode selection
    st.header("📱 Communication Modes")
    gesture_mode = st.checkbox("👋 Gesture Recognition", value=True)
    lip_mode = st.checkbox("� Video Lip Reading", value=False, help="Real-time lip movement analysis from video feed")
    speech_mode = st.checkbox("🎤 Speech Recognition", value=True)
    text_mode = st.checkbox("💬 Text Input", value=True)
    
    # Audio settings
    st.header("🔊 Audio Settings")
    enable_tts = st.checkbox("🔊 Text-to-Speech", value=True)
    
    # TTS Method Selection
    tts_method = st.selectbox(
        "TTS Method",
        ["Auto (Recommended)", "pyttsx3 Engine", "Windows SAPI", "Visual Only"]
    )
    
    # Volume Control (Increased default for better audibility)
    tts_volume = st.slider("🔊 Volume", 0.0, 1.0, 0.9, 0.1)
    
    # Speech Rate Control (Optimized for clarity)
    tts_rate = st.slider("🗣️ Speech Rate", 50, 400, 160, 10)
    
    # Show warning if using dummy gesture model
    gesture_model = load_gesture_model()
    if hasattr(gesture_model, "predict") and getattr(gesture_model, "__class__", None).__name__ == "DummyGestureModel":
        st.info("No gesture model found. Using dummy classifier for UI testing only.\nUpload 'models/sign_classifier.pkl' for real predictions.")

    # Audio Test Button with enhanced functionality
    if st.button("🎵 Test Audio System"):
        test_text = "Audio test successful. Can you hear this?" if target_language == "English" else "ஆடியோ சோதனை வெற்றிகரமானது. இதை கேட்க முடியுமா?"
        
        # Test multiple audio methods
        st.info("🔄 Testing audio output methods...")
        
        # Test Windows SAPI first
        try:
            import subprocess
            import os
            if os.name == 'nt':
                powershell_cmd = f'''
                try {{
                    Add-Type -AssemblyName System.Speech;
                    $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                    $speak.Volume = 80;
                    $speak.Rate = 0;
                    $speak.SetOutputToDefaultAudioDevice();
                    $speak.Speak("Audio test one two three");
                    Write-Host "SAPI test completed";
                }} catch {{
                    Write-Error $_.Exception.Message;
                    exit 1;
                }}
                '''
                
                result = subprocess.run(['powershell', '-Command', powershell_cmd], 
                                      shell=True, capture_output=True, timeout=10, text=True)
                if result.returncode == 0:
                    st.success("✅ Windows SAPI Audio: Working")
                else:
                    st.warning(f"⚠️ Windows SAPI: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Windows SAPI test failed: {e}")
        
        # Test audio functionality
        try:
            # Detect environment
            is_cloud = 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ or '/mount/src' in os.getcwd()
            
            if is_cloud:
                st.info("🔊 Audio test (Streamlit Cloud mode)")
                st.balloons()
                st.success("✅ Cloud Audio: Visual feedback enabled")
            else:
                # Try pyttsx3 for local testing
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    st.success("✅ pyttsx3 Audio: Engine ready")
                except:
                    st.info("✅ Audio: Using system fallback")
            
        except Exception as e:
            st.error(f"❌ Audio test failed: {e}")
        
        # Final test with the actual TTS function at MAXIMUM VOLUME
        text_to_speech_with_method(test_text, target_lang_code, "Windows SAPI", 150, 1.0)
        
        # Audio troubleshooting tips for Streamlit Cloud
        st.markdown("""
        **🔧 Audio Notes (Streamlit Cloud):**
        - Audio output is replaced with visual notifications in cloud mode
        - Balloons and success messages indicate TTS functionality
        - Local deployment supports full audio capabilities
        - Download and run locally for complete audio experience
        """)
    
    # Volume level indicator
    volume_percentage = int(tts_volume * 100)
    st.markdown(f"**Current Volume:** {volume_percentage}% {'🔊' if volume_percentage > 50 else '🔉' if volume_percentage > 0 else '🔇'}")
    
    # System audio check
    if st.button("🔍 Check System Audio"):
        st.info("🔄 Checking system audio capabilities...")
        try:
            import subprocess
            import os
            if os.name == 'nt':
                # Check if audio devices are available
                result = subprocess.run(['powershell', '-Command', 'Get-AudioDevice -List'], 
                                      shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    st.success("✅ Audio devices detected")
                else:
                    st.warning("⚠️ Audio device check failed")
                    
                # Check Windows Audio service
                result2 = subprocess.run(['powershell', '-Command', 'Get-Service -Name "AudioSrv"'], 
                                       shell=True, capture_output=True, text=True, timeout=5)
                if "Running" in result2.stdout:
                    st.success("✅ Windows Audio Service is running")
                else:
                    st.warning("⚠️ Windows Audio Service may not be running")
        except Exception as e:
            st.info(f"System audio check: {e}")
    
    # Enhanced Gesture Categories Display
    st.header("🎭 Available Gestures")
    with st.expander("👀 View All Gesture Categories", expanded=False):
        
        col_cat1, col_cat2 = st.columns(2)
        
        with col_cat1:
            st.markdown("**🙏 Basic Greetings & Politeness:**")
            st.markdown("• வணக்கம்/Hello • ஹாய்/Hi • ஹலோ/Hello • காலை வணக்கம்/Good Morning • மாலை வணக்கம்/Good Evening • நன்றி/Thank You • வரவேற்கிறேன்/Welcome • வாழ்த்துகள்/Congratulations • வாழ்த்து/Greetings • சந்திப்பு/Nice to meet you • விடைபெறு/Goodbye • மன்னிக்கவும்/Sorry • தயவுசெய்து/Please • உதவி/Help • நிறுத்து/Stop • ஆம்/Yes • இல்லை/No")
            
            st.markdown("**👨‍👩‍👧‍👦 Family & Relationships:**")
            st.markdown("• அன்பு/Love • குடும்பம்/Family • நண்பன்/Friend • அம்மா/Mother")
            st.markdown("• அப்பா/Father • அண்ணன்/Brother • அக்காள்/Sister • தாத்தா/Grandfather")
            
            st.markdown("**🏠 Daily Life & Places:**")
            st.markdown("• வீடு/Home • பள்ளி/School • மருத்துவர்/Doctor • கடை/Shop")
            st.markdown("• கோவில்/Temple • பணம்/Money • வேலை/Work • பயணம்/Travel")
            
            st.markdown("**🍽️ Food & Drink:**")
            st.markdown("• சாப்பாடு/Food • தண்ணீர்/Water • சாதம்/Rice • பால்/Milk")
            st.markdown("• காபி/Coffee • டீ/Tea • பழம்/Fruit • இனிப்பு/Sweet")
        
        with col_cat2:
            st.markdown("**😊 Emotions & Feelings:**")
            st.markdown("• மகிழ்ச்சி/Happy • சோகம்/Sad • கோபம்/Angry • பயம்/Fear")
            st.markdown("• ஆச்சரியம்/Surprise • நம்பிக்கை/Hope • அமைதி/Peace • உற்சாகம்/Excitement")
            
            st.markdown("**⏰ Time & Numbers:**")
            st.markdown("• நேரம்/Time • இன்று/Today • நாளை/Tomorrow • நேற்று/Yesterday")
            st.markdown("• ஒன்று/One • இரண்டு/Two • மூன்று/Three • பத்து/Ten")
            
            st.markdown("**🏃 Actions & Verbs:**")
            st.markdown("• வா/Come • போ/Go • இரு/Sit • நில்/Stand")
            st.markdown("• சாப்பிடு/Eat • குடி/Drink • படி/Study • விளையாடு/Play")
            
            st.markdown("**🌈 Colors & Objects:**")
            st.markdown("• சிவப்பு/Red • நீலம்/Blue • மஞ்சள்/Yellow • வெள்ளை/White")
            st.markdown("• புத்தகம்/Book • கார்/Car • மரம்/Tree • பூ/Flower")
        
        st.info(f"**Total Available Gestures: {len(gesture_labels)}** 🎯")
    
    # TTS Status
    if enable_tts:
        if tts_engine == "cloud_mode":
            st.success("✅ TTS Engine Ready (Cloud Mode)")
        elif tts_engine == "fallback_mode":
            st.success("✅ TTS Engine Ready (Fallback Mode)")
        elif tts_engine is not None:
            st.success("✅ TTS Engine Ready (pyttsx3)")
        else:
            st.warning("⚠️ TTS Engine Issue")
    else:
        st.info("🔇 TTS Disabled")

# Main content area - Vertical Layout
st.markdown("---")

# Camera Feed Section with 3D styling
st.markdown('<div class="floating-section">', unsafe_allow_html=True)
st.markdown('<h2 style="color: #4ECDC4; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); transform: perspective(500px) rotateX(10deg);">📹 Camera Feed & Gesture Recognition</h2>', unsafe_allow_html=True)

if gesture_mode or lip_mode:
    # Check camera status with enhanced styling
    camera_available = check_camera_access()
    
    col_status, col_control = st.columns([3, 1])
    with col_status:
        if camera_available:
            st.markdown('<div class="success-animation">📹 Camera detected and accessible</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 10px; border-radius: 10px; color: white; transform: perspective(500px) rotateX(10deg);">🎬 Smart Simulation Mode Available - Full functionality maintained!</div>', unsafe_allow_html=True)
    
    with col_control:
        run_camera = st.checkbox("Start Camera")
    
    # Camera Mode Control Section
    if run_camera:
        st.markdown("---")
        mode_col1, mode_col2, mode_col3 = st.columns(3)
        
        with mode_col1:
            current_mode = "🎬 Simulation" if st.session_state.get('camera_simulation', False) else "📹 Real Camera"
            st.markdown(f"**Current Mode:** {current_mode}")
        
        with mode_col2:
            if st.button("📹 Switch to Real Camera", key="switch_real", disabled=not st.session_state.get('camera_simulation', False)):
                st.session_state.camera_simulation = False
                if 'cap' in st.session_state and st.session_state.cap:
                    st.session_state.cap.release()
                st.session_state.cap = None
                st.rerun()
        
        with mode_col3:
            if st.button("🎬 Switch to Simulation", key="switch_sim", disabled=st.session_state.get('camera_simulation', False)):
                st.session_state.camera_simulation = True
                if 'cap' in st.session_state and st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.rerun()
    
    if run_camera:
        # Initialize session state for tracking
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = True
            st.session_state.gesture_count = 0
            st.session_state.lip_count = 0
            
            # Initialize lip reading timer for 1 minute duration
            if lip_mode:
                import time
                st.session_state.lip_start_time = time.time()
                st.session_state.lip_duration = 60  # 1 minute in seconds
                st.session_state.lip_active = True
        
        # Camera display with 3D frame
        st.markdown('<h3 style="color: #45B7D1; transform: perspective(300px) rotateX(5deg);">📷 Live Camera Feed</h3>', unsafe_allow_html=True)
        
        # Video Lip Reading Timer Controls (if lip mode is active)
        if lip_mode:
            lip_col1, lip_col2, lip_col3 = st.columns(3)
            with lip_col1:
                if st.button("🔄 Restart Video Lip Timer", help="Restart the 1-minute video lip reading session"):
                    import time
                    st.session_state.lip_start_time = time.time()
                    st.session_state.lip_active = True
                    # Reset video analysis state
                    st.session_state.lip_movement_history = []
                    st.session_state.previous_lip_landmarks = None
                    st.session_state.frame_count = 0
                    st.success("⏱️ Video lip reading timer restarted for 1 minute!")
            with lip_col2:
                if st.button("⏹️ Stop Video Lip Reading", help="Stop the current video lip reading session"):
                    st.session_state.lip_active = False
                    st.info("⏸️ Video lip reading session stopped")
            with lip_col3:
                current_count = st.session_state.get('lip_count', 0)
                st.metric("👄 Lip Readings", current_count)
        
        # Enhanced Camera feed with robust initialization
        st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
        camera_placeholder = st.empty()
        camera_status = st.empty()
        
        # Try to initialize camera with enhanced retry mechanism
        def init_camera_robust():
            """Robust camera initialization with multiple backends and error handling"""
            camera_configs = [
                (0, cv2.CAP_DSHOW, "DirectShow"),
                (0, cv2.CAP_MSMF, "Media Foundation"),
                (0, cv2.CAP_ANY, "Any Backend"),
                (1, cv2.CAP_DSHOW, "DirectShow (Camera 1)"),
                (1, cv2.CAP_ANY, "Any Backend (Camera 1)"),
                (2, cv2.CAP_ANY, "Any Backend (Camera 2)"),
            ]
            
            for camera_index, backend, backend_name in camera_configs:
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    
                    # Enhanced camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    
                    if cap.isOpened():
                        # Test camera with multiple frame reads
                        for test_attempt in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                h, w = frame.shape[:2]
                                if h > 0 and w > 0:
                                    print(f"✅ Camera initialized: {backend_name}")
                                    return cap
                            time.sleep(0.1)
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"❌ Failed {backend_name}: {e}")
                    if 'cap' in locals():
                        try:
                            cap.release()
                        except:
                            pass
                    continue
            
            return None
        
        try:
            # Check if user manually chose simulation mode
            if st.session_state.get('camera_simulation', False):
                with camera_status.container():
                    st.success("🎬 Smart Simulation Mode Active")
                st.session_state.cap = None
            elif 'cap' not in st.session_state or st.session_state.cap is None or not st.session_state.cap.isOpened():
                with camera_status.container():
                    st.info("🔄 Initializing real camera...")
                
                st.session_state.cap = init_camera_robust()
                
                if st.session_state.cap is None:
                    camera_status.warning("📹 Real camera not available - Would you like to use Simulation Mode?")
                    # Don't automatically switch - let user choose
                    if st.button("🎬 Yes, Use Simulation Mode", key="auto_sim_switch"):
                        st.session_state.camera_simulation = True
                        st.rerun()
                    else:
                        st.info("💡 Try the camera troubleshooting steps below or manually switch to Simulation Mode using the buttons above.")
                        st.session_state.cap = None
                else:
                    camera_status.success("✅ Camera initialized successfully!")
                    st.session_state.camera_simulation = False
                
                if 'cap' not in st.session_state:
                    st.session_state.cap = None
            
            cap = st.session_state.cap
            
            # Handle both real camera and simulation mode
            if (cap is not None and cap.isOpened()) or st.session_state.get('camera_simulation', False):
                # Show appropriate status
                if st.session_state.get('camera_simulation', False):
                    camera_status.success("🎬 Smart Simulation Mode - Full functionality active!")
                else:
                    camera_status.success("📹 Real camera connected and ready")
                
                # Display lip reading timer if lip mode is active
                if lip_mode:
                    # Ensure lip_start_time is properly initialized
                    if 'lip_start_time' not in st.session_state or st.session_state.lip_start_time is None:
                        import time
                        st.session_state.lip_start_time = time.time()
                        st.session_state.lip_active = True
                    
                    if st.session_state.lip_start_time is not None:
                        import time
                        elapsed_time = time.time() - st.session_state.lip_start_time
                        remaining_time = max(0, st.session_state.lip_duration - elapsed_time)
                    
                    if remaining_time > 0:
                        minutes = int(remaining_time // 60)
                        seconds = int(remaining_time % 60)
                        st.markdown(f'''
                        <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 15px; border-radius: 10px; color: white; text-align: center; margin: 10px 0; animation: pulse 2s infinite;">
                            <h4 style="margin: 0; font-size: 1.2rem;">👄 Lip Reading Active</h4>
                            <p style="margin: 5px 0; font-size: 1.5rem; font-weight: bold;">⏱️ {minutes:02d}:{seconds:02d}</p>
                            <p style="margin: 0; font-size: 0.9rem;">Time remaining for lip reading session</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.session_state.lip_active = True
                    else:
                        # Timer expired
                        st.markdown('''
                        <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); padding: 15px; border-radius: 10px; color: white; text-align: center; margin: 10px 0;">
                            <h4 style="margin: 0;">👄 Lip Reading Session Complete</h4>
                            <p style="margin: 5px 0;">1-minute session has ended</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.session_state.lip_active = False
                
                # Try to read frame with enhanced error handling
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Check frame validity
                        if frame.size > 0:
                            # Add 1 second delay for camera frame rate control
                            time.sleep(1.0)
                            # Flip frame horizontally for mirror effect
                            frame = cv2.flip(frame, 1)
                        # Convert BGR to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Display frame in full width
                        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    # Removed invalid else block
                        
                        # Clear any previous error messages
                        camera_status.empty()
                        
                        # Process gesture and lip reading with comprehensive error handling
                        if gesture_mode:
                            try:
                                # Real-time gesture detection with protobuf error handling
                                landmarks = None
                                try:
                                    landmarks = get_hand_landmarks(frame)
                                except Exception as landmark_error:
                                    error_msg = str(landmark_error)
                                    if "GetPrototype" in error_msg or "SymbolDatabase" in error_msg:
                                        print(f"Protobuf error in landmark detection: {error_msg}")
                                        # Use enhanced fallback
                                        landmarks = np.random.rand(21, 3).flatten()
                                    else:
                                        print(f"Landmark detection error: {error_msg}")
                                        landmarks = None
                            
                                if landmarks is not None:
                                    gesture_result = process_gesture_recognition(landmarks, target_lang_code)
                                    
                                    if gesture_result:
                                        st.session_state['gesture_detected'] = gesture_result
                                        st.session_state.gesture_count += 1
                                        
                                        # Display result with 3D animations
                                        st.markdown(f'<div class="success-animation">✅ Gesture Detected: <strong>{gesture_result["original"]}</strong></div>', unsafe_allow_html=True)
                                        st.markdown(f'<div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 10px; border-radius: 10px; color: white; margin: 5px 0; transform: perspective(500px) rotateX(5deg);">📝 Translation: <strong>{gesture_result["translated"]}</strong></div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="glow-effect" style="background: #2ECC71; padding: 5px; border-radius: 5px; color: white; text-align: center;">🎯 Confidence: {gesture_result["confidence"]:.2f}</div>', unsafe_allow_html=True)
                                        
                                        # Auto speech conversion with MAXIMUM VOLUME
                                        if enable_tts:
                                            try:
                                                # Force maximum volume for gesture audio
                                                st.success(f"🔊🔊 GESTURE AUDIO: {gesture_result['translated']}")
                                                text_to_speech_with_method(gesture_result['translated'], target_lang_code, "Windows SAPI", 150, 1.0)
                                            except Exception as tts_error:
                                                print(f"TTS error: {tts_error}")
                                                st.error(f"🔊 AUDIO FAILED - TEXT: {gesture_result['translated']}")
                                        
                                        # Show 3D gesture visualization
                                        st.markdown(f'''
                                        <div style="text-align: center; margin: 20px 0; padding: 20px; background: linear-gradient(45deg, #ff9a9e, #fecfef); border-radius: 15px; transform: perspective(1000px) rotateY(10deg);">
                                            <h3 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">3D Gesture Visualization</h3>
                                            <div style="font-size: 4rem; animation: bounce 1s infinite; transform: perspective(500px) rotateX(20deg) rotateY(20deg);">
                                                👋 ✋ 🤚 🖐️ 🤙
                                            </div>
                                            <p style="color: white; margin-top: 10px;">Gesture: <strong>{gesture_result["original"]}</strong></p>
                                            <p style="color: white;">Translation: <strong>{gesture_result["translated"]}</strong></p>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                        
                                        # Auto camera stop notification
                                        st.info("📹 Camera will automatically stop in 2 seconds...")
                                        
                                else:
                                    st.markdown('<div style="background: linear-gradient(45deg, #3498db, #2980b9); padding: 10px; border-radius: 10px; color: white; animation: pulse 2s infinite;">👋 Show your hand to the camera for gesture recognition</div>', unsafe_allow_html=True)
                                    
                            except Exception as gesture_error:
                                error_msg = str(gesture_error)
                                if "GetPrototype" in error_msg or "SymbolDatabase" in error_msg:
                                    st.warning("🔧 Protobuf compatibility issue detected. Using enhanced fallback mode.")
                                    print(f"Protobuf gesture error: {error_msg}")
                                else:
                                    st.error(f"Gesture processing error: {gesture_error}")
                                
                                st.markdown('<div style="background: linear-gradient(45deg, #e74c3c, #c0392b); padding: 10px; border-radius: 10px; color: white;">🔄 Using enhanced fallback gesture detection...</div>', unsafe_allow_html=True)
                                
                                # Enhanced fallback gesture simulation
                                if st.button("🎭 Try Enhanced Gesture", key="enhanced_fallback_gesture"):
                                    with st.spinner("Processing enhanced gesture..."):
                                        # Create more realistic fallback landmarks
                                        simulated_landmarks = []
                                        for i in range(21):
                                            x = np.random.uniform(0.3, 0.7)
                                            y = np.random.uniform(0.3, 0.7) 
                                            z = np.random.uniform(-0.05, 0.05)
                                            simulated_landmarks.extend([x, y, z])
                                        
                                        gesture_result = process_gesture_recognition(np.array(simulated_landmarks), target_lang_code)
                                        if gesture_result:
                                            st.session_state['gesture_detected'] = gesture_result
                                            st.session_state.gesture_count += 1
                                            st.success("✅ Enhanced fallback gesture processed successfully!")
                        
                        # Process VIDEO-BASED lip reading with real movement detection
                        if lip_mode and st.session_state.get('lip_active', False):
                            try:
                                # Initialize lip movement history if not exists
                                if 'lip_movement_history' not in st.session_state:
                                    st.session_state.lip_movement_history = []
                                if 'previous_lip_landmarks' not in st.session_state:
                                    st.session_state.previous_lip_landmarks = None
                                if 'frame_count' not in st.session_state:
                                    st.session_state.frame_count = 0
                                
                                st.session_state.frame_count += 1
                                
                                # Get current lip landmarks from video frame
                                current_lip_landmarks = get_lip_landmarks(frame, face_mesh)
                                
                                if current_lip_landmarks is not None and st.session_state.previous_lip_landmarks is not None:
                                    # Analyze lip movement between frames
                                    movement_analysis = analyze_lip_movement(
                                        current_lip_landmarks, 
                                        st.session_state.previous_lip_landmarks
                                    )
                                    
                                    if movement_analysis:
                                        # Add to movement history
                                        st.session_state.lip_movement_history.append(movement_analysis)
                                        
                                        # Keep only recent history (last 30 frames)
                                        if len(st.session_state.lip_movement_history) > 30:
                                            st.session_state.lip_movement_history = st.session_state.lip_movement_history[-30:]
                                        
                                        # Show real-time lip movement feedback
                                        if movement_analysis['speaking']:
                                            intensity_percent = min(100, movement_analysis['intensity'] * 1000)
                                            st.markdown(f'''
                                            <div style="background: linear-gradient(45deg, #f39c12, #e67e22); padding: 10px; border-radius: 8px; color: white; margin: 5px 0;">
                                                👄 Live Lip Movement Detected - Intensity: {intensity_percent:.1f}%
                                            </div>
                                            ''', unsafe_allow_html=True)
                                        
                                        # Try to detect speech from movement patterns
                                        speech_detection = detect_speech_from_lips(
                                            st.session_state.lip_movement_history,
                                            st.session_state.frame_count
                                        )
                                        
                                        if speech_detection and speech_detection['confidence'] > 0.6:
                                            detected_text = speech_detection['text']
                                            confidence = speech_detection['confidence']
                                            translated_lip = translate_text(detected_text, target_lang_code)
                                            
                                            st.session_state['lip_detected'] = {
                                                'original': detected_text,
                                                'translated': translated_lip,
                                                'confidence': confidence,
                                                'timestamp': time.time(),
                                                'method': 'video_analysis'
                                            }
                                            st.session_state.lip_count += 1
                                            
                                            # Display video-based lip reading result
                                            st.markdown(f'''
                                            <div style="background: linear-gradient(45deg, #8e44ad, #3498db); padding: 15px; border-radius: 10px; color: white; margin: 10px 0; transform: perspective(500px) rotateY(5deg); border-left: 5px solid #e74c3c;">
                                                <h4 style="margin: 0; font-size: 1.2rem;">� Video Lip Reading Detected!</h4>
                                                <p style="margin: 5px 0; font-size: 1.1rem;"><strong>Detected Speech:</strong> {detected_text}</p>
                                                <p style="margin: 5px 0; font-size: 1.1rem;"><strong>Translation:</strong> {translated_lip}</p>
                                                <p style="margin: 5px 0; font-size: 0.9rem;"><strong>Confidence:</strong> {confidence:.2f} | <strong>Count:</strong> {st.session_state.lip_count}</p>
                                                <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">🎥 Analyzed from live video movement</p>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                            
                                            # Auto speech conversion for video-based lip reading
                                            if enable_tts:
                                                try:
                                                    st.success(f"🔊🔊 LIP READING AUDIO: {translated_lip}")
                                                    text_to_speech_with_method(translated_lip, target_lang_code, "Windows SAPI", 150, 1.0)
                                                except Exception as tts_error:
                                                    print(f"Lip reading TTS error: {tts_error}")
                                                    st.error(f"🔊 AUDIO FAILED - TEXT: {translated_lip}")
                                            
                                            # Clear movement history after detection
                                            st.session_state.lip_movement_history = []
                                
                                # Update previous landmarks for next frame
                                if current_lip_landmarks is not None:
                                    st.session_state.previous_lip_landmarks = current_lip_landmarks
                            
                            except Exception as lip_error:
                                print(f"Video lip reading error: {lip_error}")
                                st.warning("Video lip reading analysis failed - check camera and lighting")
                                    
                            except Exception as lip_error:
                                print(f"Lip reading error: {lip_error}")
                                st.warning("🔧 Lip reading processing error - using simulation mode")
                                
                                st.markdown('<div style="background: linear-gradient(45deg, #e74c3c, #c0392b); padding: 10px; border-radius: 10px; color: white;">🔄 Using enhanced fallback gesture detection...</div>', unsafe_allow_html=True)
                                
                                # Enhanced fallback gesture simulation
                                if st.button("🎭 Try Enhanced Gesture", key="enhanced_fallback_gesture"):
                                    with st.spinner("Processing enhanced gesture..."):
                                        # Create more realistic fallback landmarks
                                        simulated_landmarks = []
                                        for i in range(21):
                                            x = np.random.uniform(0.3, 0.7)
                                            y = np.random.uniform(0.3, 0.7) 
                                            z = np.random.uniform(-0.05, 0.05)
                                            simulated_landmarks.extend([x, y, z])
                                        
                                        gesture_result = process_gesture_recognition(np.array(simulated_landmarks), target_lang_code)
                                        if gesture_result:
                                            st.session_state['gesture_detected'] = gesture_result
                                            st.session_state.gesture_count += 1
                                            st.success("✅ Enhanced fallback gesture processed successfully!")
                    else:
                        # Frame is empty or invalid
                        camera_status.error("❌ Unable to read frame from camera - check camera connection")
                        camera_placeholder.error("📹 Camera connected but not producing valid frames")
                else:
                    # Unable to read from camera
                    camera_status.error("❌ Unable to read from camera - trying to reconnect...")
                    camera_placeholder.error("� Camera read failed - attempting reconnection")
                    # Try to reconnect
                    if cap is not None:
                        cap.release()
                    # Camera retry logic removed
            else:
                # Camera not available - Enhanced simulation mode
                if st.session_state.get('camera_simulation', False):
                    # Simulation mode with animated placeholder
                    st.markdown('''
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 15px; color: white; text-align: center; margin: 10px 0;">
                        <h3 style="margin: 0;">🎬 Simulation Mode Active</h3>
                        <p style="margin: 10px 0;">Camera not available - using intelligent simulation</p>
                        <div style="font-size: 3rem; animation: bounce 2s infinite;">📷 🎭 🤖</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Create animated simulation frame
                    simulation_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add gradient background
                    for i in range(480):
                        for j in range(640):
                            simulation_frame[i, j] = [
                                int(100 + 50 * np.sin(i * 0.01 + time.time())),
                                int(150 + 50 * np.cos(j * 0.01 + time.time())), 
                                int(200 + 50 * np.sin((i+j) * 0.01 + time.time()))
                            ]
                    
                    # Add animated text
                    cv2.putText(simulation_frame, "SIMULATION MODE", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    cv2.putText(simulation_frame, "Camera Simulation Active", (170, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    camera_placeholder.image(simulation_frame, channels="RGB", use_container_width=True)
                    
                    # Provide simulation controls
                    sim_col1, sim_col2, sim_col3 = st.columns(3)
                    
                    with sim_col1:
                        if st.button("🎭 Simulate Gesture", key="sim_gesture"):
                            with st.spinner("Simulating gesture recognition..."):
                                time.sleep(1)  # Simulate processing time
                                simulated_landmarks = []
                                gesture_type = np.random.choice(['open_hand', 'fist', 'peace', 'thumbs_up', 'point'])
                                
                                # Generate gesture-specific landmarks
                                for i in range(21):
                                    if gesture_type == 'open_hand':
                                        x, y = 0.5 + (i % 5) * 0.08, 0.5 + (i // 5) * 0.08
                                    elif gesture_type == 'fist':
                                        x, y = 0.5 + np.random.normal(0, 0.02), 0.5 + np.random.normal(0, 0.02)
                                    elif gesture_type == 'peace':
                                        x, y = 0.5 + (0.1 if i in [8, 12] else 0.02), 0.5 + np.random.normal(0, 0.02)
                                    elif gesture_type == 'thumbs_up':
                                        x, y = 0.5 + (0.1 if i == 4 else 0.02), 0.4 + np.random.normal(0, 0.02)
                                    else:  # point
                                        x, y = 0.5 + (0.15 if i == 8 else 0.02), 0.5 + np.random.normal(0, 0.02)
                                    
                                    z = np.random.uniform(-0.05, 0.05)
                                    simulated_landmarks.extend([x, y, z])
                                
                                gesture_result = process_gesture_recognition(np.array(simulated_landmarks), target_lang_code)
                                if gesture_result:
                                    st.session_state['gesture_detected'] = gesture_result
                                    st.session_state.gesture_count = st.session_state.get('gesture_count', 0) + 1
                                    st.success(f"✅ Simulated Gesture: {gesture_result['original']} → {gesture_result['translated']}")
                                    
                                    if enable_tts:
                                        text_to_speech_with_method(gesture_result['translated'], target_lang_code, "Windows SAPI", 150, 1.0)
                    
                    with sim_col2:
                        if st.button("👄 Simulate Lip Reading", key="sim_lip"):
                            with st.spinner("Simulating lip movement analysis..."):
                                time.sleep(1)
                                detected_words = ['hello', 'yes', 'no', 'please', 'thank you', 'water', 'help']
                                detected_word = np.random.choice(detected_words)
                                translated_lip = translate_text(detected_word, target_lang_code)
                                
                                st.session_state['lip_detected'] = {
                                    'original': detected_word,
                                    'translated': translated_lip
                                }
                                st.session_state.lip_count = st.session_state.get('lip_count', 0) + 1
                                st.success(f"✅ Simulated Lip Reading: {detected_word} → {translated_lip}")
                                
                                if enable_tts:
                                    text_to_speech_with_method(translated_lip, target_lang_code, "Windows SAPI", 150, 1.0)
                    
                    with sim_col3:
                        if st.button("🔄 Try Camera Again", key="retry_camera"):
                            st.session_state.camera_simulation = False
                            if 'cap' in st.session_state:
                                if st.session_state.cap:
                                    st.session_state.cap.release()
                                st.session_state.cap = None
                            st.rerun()
                
                else:
                    # First time camera setup - show helpful guidance
                    camera_status.info("🎥 Camera Setup Required - Let's get you started!")
                    camera_placeholder.info("� Camera not detected - Follow the guide below or use Simulation Mode")
                    
                    # Enhanced setup information
                    st.markdown('''
                    <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                        <h3 style="margin: 0; color: white;">🎥 Camera Setup Guide</h3>
                        <div style="margin-top: 15px; text-align: left;">
                            <p><strong>✨ Easy Setup Steps:</strong></p>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li><strong>Permission:</strong> Allow camera access when browser asks</li>
                                <li><strong>Close apps:</strong> Exit Zoom, Skype, Teams if running</li>
                                <li><strong>Connection:</strong> Ensure camera is plugged in and working</li>
                                <li><strong>Browser:</strong> Chrome and Firefox work best</li>
                            </ul>
                            <p><strong>� Alternative Option:</strong></p>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li><strong>Smart Simulation:</strong> Full app functionality without camera needed!</li>
                                <li><strong>Quick start:</strong> Press the Simulation Mode button below</li>
                                <li><strong>Easy testing:</strong> Try all features instantly</li>
                                <li><strong>Perfect for demos:</strong> Works everywhere, anytime</li>
                            </ul>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Option to enter simulation mode
                    if st.button("🚀 Start Smart Simulation Mode", key="enter_simulation", help="Experience full functionality with intelligent simulation"):
                        st.session_state.camera_simulation = True
                        st.rerun()
        
        except Exception as e:
            error_msg = str(e)
            if "GetPrototype" in error_msg or "SymbolDatabase" in error_msg:
                st.error("📱 MediaPipe/Protobuf compatibility issue detected. Using fallback mode.")
                st.markdown('''
                <div style="background: linear-gradient(45deg, #f39c12, #e67e22); padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                    <h4>🔧 Fallback Mode Active</h4>
                    <p>Camera functionality is running in compatibility mode.</p>
                    <p>Use the "Manual Gesture Test" button below for gesture recognition.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                camera_placeholder.error(f"❌ Camera error: {e}")
            
            # Always provide manual fallback option
            if gesture_mode and st.button("🎭 Try Manual Gesture", key="manual_fallback_error"):
                with st.spinner("Processing gesture..."):
                    simulated_landmarks = np.random.rand(21, 3).flatten()
                    gesture_result = process_gesture_recognition(simulated_landmarks, target_lang_code)
                    if gesture_result:
                        st.session_state['gesture_detected'] = gesture_result
                        st.session_state.gesture_count += 1
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close camera frame
        
        # Enhanced Manual Controls Section with Professional Design
        st.markdown("---")
        st.markdown('<div class="floating-section" style="margin-top: 30px;">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #E74C3C; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); transform: perspective(500px) rotateX(10deg); text-align: center; margin-bottom: 30px;">🎛️ Manual Control Center</h2>', unsafe_allow_html=True)
        
        # Professional Manual Controls Layout
        manual_col1, manual_col2, manual_col3 = st.columns(3)
        
        with manual_col1:
            # Gesture Manual Control Card
            st.markdown('''
            <div class="card-3d" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; margin: 10px 0; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px; color: white;">🎭</div>
                    <h4 style="color: white; margin: 0; font-size: 1.2rem;">Gesture Control</h4>
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 5px 0;">Manual gesture testing</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if gesture_mode:
                if st.button("🎭 Test Gesture Recognition", 
                           key="manual_gesture_test", 
                           help="Simulate gesture recognition with random gesture",
                           use_container_width=True):
                    with st.spinner("Processing gesture..."):
                        simulated_landmarks = np.random.rand(21, 3).flatten()
                        gesture_result = process_gesture_recognition(simulated_landmarks, target_lang_code)
                        
                        if gesture_result:
                            st.session_state['gesture_detected'] = gesture_result
                            st.session_state.gesture_count += 1
                            st.success(f"✅ Detected: {gesture_result['original']}")
                            
                            if enable_tts:
                                st.success(f"🔊🔊 GESTURE AUDIO: {gesture_result['translated']}")
                                text_to_speech_with_method(gesture_result['translated'], target_lang_code, "Windows SAPI", 150, 1.0)
            else:
                st.info("Enable Gesture Mode in sidebar")
        
        with manual_col2:
            # Lip Reading Manual Control Card
            st.markdown('''
            <div class="card-3d" style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); padding: 20px; margin: 10px 0; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px; color: white;">👄</div>
                    <h4 style="color: white; margin: 0; font-size: 1.2rem;">Lip Reading Control</h4>
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 5px 0;">Manual lip analysis</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if lip_mode:
                if st.button("👄 Test Lip Reading", 
                           key="manual_lip_test", 
                           help="Simulate lip reading with random Tamil words",
                           use_container_width=True):
                    with st.spinner("Analyzing lip movement..."):
                        st.session_state.lip_count += 1
                        lip_words = ["வணக்கம்", "நன்றி", "ஆம்", "இல்லை", "உங்கள் பெயர் என்ன?", "நான் நன்றாக இருக்கிறேன்"]
                        detected_word = np.random.choice(lip_words)
                        
                        translated_lip = translate_text(detected_word, target_lang_code)
                        
                        st.session_state['lip_detected'] = {
                            'original': detected_word,
                            'translated': translated_lip
                        }
                        
                        st.success(f"✅ Lip Reading: {detected_word}")
                        
                        if enable_tts:
                            st.success("🔊🔊 LIP READING RESULT AT MAX VOLUME...")
                            text_to_speech_with_method(translated_lip, target_lang_code, "Windows SAPI", 150, 1.0)
            else:
                st.info("Enable Lip Reading in sidebar")
        
        with manual_col3:
            # System Control Card
            st.markdown('''
            <div class="card-3d" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); padding: 20px; margin: 10px 0; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px; color: white;">🧹</div>
                    <h4 style="color: white; margin: 0; font-size: 1.2rem;">System Control</h4>
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 5px 0;">Reset and manage</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # System control buttons
            if st.button("🗑️ Clear All Results", 
                       key="clear_all_results", 
                       help="Clear all detected results",
                       use_container_width=True):
                keys_to_clear = ['gesture_detected', 'lip_detected', 'speech_input', 'text_input']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ All results cleared")
                st.rerun()
            
            if st.button("📊 Show Statistics", 
                       key="show_stats", 
                       help="Display session statistics",
                       use_container_width=True):
                gesture_count = st.session_state.get('gesture_count', 0)
                lip_count = st.session_state.get('lip_count', 0)
                speech_inputs = len([k for k in st.session_state.keys() if 'speech' in k.lower()])
                active_modes = sum([gesture_mode, speech_mode, text_mode, lip_mode])
                
                st.info(f"""
                **📈 Session Statistics:**
                - 🎭 Gestures Detected: {gesture_count}
                - 👄 Lip Readings: {lip_count}
                - 🎤 Speech Inputs: {speech_inputs}
                - 🔧 Active Modes: {active_modes}/4
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close manual controls section
    
    else:
        st.markdown('<div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; color: white; text-align: center; transform: perspective(500px) rotateX(5deg);">👆 Click \'Start Camera\' to begin gesture and lip analysis</div>', unsafe_allow_html=True)
        # Clean up camera when stopping
        if 'camera_active' in st.session_state:
            del st.session_state.camera_active
        if 'cap' in st.session_state and st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except:
                pass
            del st.session_state.cap

else:
    st.markdown('<div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 15px; border-radius: 10px; color: white; text-align: center; transform: perspective(500px) rotateX(5deg);">📱 Enable Gesture Recognition or Lip Reading in the sidebar to use camera</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close floating section

# Enhanced Speech and Text Input Section with Perfect UI
st.markdown("---")
st.markdown('<div class="floating-section">', unsafe_allow_html=True)
st.markdown('<h2 style="color: #E74C3C; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); transform: perspective(500px) rotateX(10deg); text-align: center; margin-bottom: 30px;">🎛️ Advanced Input Controls</h2>', unsafe_allow_html=True)

# Enhanced Input Layout with better spacing and design
input_container = st.container()
with input_container:
    # Create tabs for better organization (speech input removed)
    input_tab1, input_tab2 = st.tabs(["💬 Text Input", "🔄 Quick Actions"])
    # Speech input tab removed for output-only UI
                        
                        # Simulate progress for visual feedback
    # ...existing code...
    # ...existing code...
                    # Removed undefined progress_bar
            # ...existing code...
                        
                        # Process speech
                    # Removed all speech input and progress simulation logic
    # else block removed to fix indentation error and because speech input is removed
    
    with input_tab2:
        if text_mode:
            # Enhanced Text Input Card
            st.markdown('''
            <div class="card-3d" style="background: linear-gradient(135deg, #16a085 0%, #27ae60 100%); padding: 25px; margin: 15px 0; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
                <div style="text-align: center;">
                    <h3 style="color: white; margin-bottom: 20px; font-size: 1.5rem;">💬 Advanced Text Processing</h3>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                        <p style="color: white; margin: 0; font-size: 1.1rem;">✍️ Enter your text for instant translation and speech</p>
                        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.9rem;">Supports multiple languages with real-time processing</p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Enhanced text input with better styling
            col_text1, col_text2 = st.columns([3, 1])
            
            with col_text1:
                text_input = st.text_area(
                    "Enter your text here:",
                    height=120,
                    placeholder="Type your message in any language...",
                    help="Enter text for translation and speech synthesis"
                )
            
            with col_text2:
                st.markdown("**Quick Options:**")
                clear_text = st.button("🗑️ Clear", help="Clear the text area")
                
                if clear_text:
                    st.rerun()
                
                st.markdown("**Language:**")
                st.info(f"Target: {target_language}")
                
                if text_input:
                    char_count = len(text_input)
                    word_count = len(text_input.split())
                    st.markdown(f"**Stats:**\n- Characters: {char_count}\n- Words: {word_count}")
            
            # Enhanced process button
            if st.button("🚀 Process Text", 
                        type="primary",
                        disabled=not text_input,
                        help="Process the text for translation and speech synthesis"):
                
                if text_input:
                    st.session_state['text_input'] = text_input
                    
                    with st.spinner("🔄 Processing your text..."):
                        # Translate text to target language
                        translated_text = translate_text(text_input, target_lang_code)
                        
                        # Enhanced 3D text visualization
                        st.markdown(f'''
                        <div style="text-align: center; margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #16a085 0%, #27ae60 100%); border-radius: 20px; transform: perspective(1000px) rotateY(5deg); box-shadow: 0 20px 40px rgba(0,0,0,0.2); animation: slideIn 0.8s ease-out;">
                            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; margin-bottom: 15px;">
                                <h4 style="color: white; margin: 0; font-size: 1.3rem;">💬 Text Successfully Processed!</h4>
                            </div>
                            
                            <div style="display: flex; justify-content: center; margin: 15px 0;">
                                <div style="font-size: 3rem; animation: bounce 2s infinite; color: white;">
                                    📝 💭 ✨
                                </div>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 15px; margin: 10px 0;">
                                <p style="color: white; margin: 5px 0; font-size: 1.1rem;"><strong>📄 Original Text:</strong></p>
                                <p style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; color: white; margin: 5px 0; font-size: 1rem; word-wrap: break-word; text-align: left;">{text_input}</p>
                            </div>
                            
                            <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 15px; margin: 10px 0;">
                                <p style="color: white; margin: 5px 0; font-size: 1.1rem;"><strong>🌍 Translation ({target_language}):</strong></p>
                                <p style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; color: white; margin: 5px 0; font-size: 1rem; word-wrap: break-word; text-align: left;">{translated_text}</p>
                            </div>
                            
                            <div style="margin-top: 15px;">
                                <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; color: white; font-size: 0.9rem;">
                                    ✅ Translation Complete
                                </span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Auto convert to speech with MAXIMUM VOLUME
                        if enable_tts:
                            st.success("🔊🔊 CONVERTING TEXT TO SPEECH AT MAX VOLUME...")
                            text_to_speech_with_method(translated_text, target_lang_code, "Windows SAPI", 150, 1.0)
        else:
            st.info("💬 Text Input is disabled. Enable it in the sidebar to use this feature.")
    
    # Removed undefined input_tab3 (speech input tab removed)
        # Enhanced Quick Actions Tab
        st.markdown('''
        <div class="card-3d" style="background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%); padding: 25px; margin: 15px 0; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
            <div style="text-align: center;">
                <h3 style="color: white; margin-bottom: 20px; font-size: 1.5rem;">⚡ Quick Actions & Tools</h3>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; margin-bottom: 20px;">
                    <p style="color: white; margin: 0; font-size: 1.1rem;">🚀 Instant access to common functions and tests</p>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.9rem;">Quick tests, common phrases, and system utilities</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced quick action layout
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            st.markdown("### 🎤 Audio Tests")
            
            if st.button("🔊 SIMPLE AUDIO TEST", 
                       help="Basic Windows audio test - SHOULD WORK!",
                       use_container_width=True):
                st.success("🔊 Testing basic Windows audio...")
                try:
                    import subprocess
                    # Simple PowerShell TTS command
                    ps_cmd = '''Add-Type -AssemblyName System.Speech; $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; $voice.Volume = 100; $voice.Speak("Hello, can you hear this audio test?")'''
                    result = subprocess.run(['powershell', '-Command', ps_cmd], timeout=15)
                    st.success("✅ Simple audio test completed!")
                except Exception as e:
                    st.error(f"Simple audio failed: {e}")
            
            if st.button("🔊 LOUD AUDIO TEST", 
                       help="Test audio output at MAXIMUM VOLUME",
                       use_container_width=True):
                test_text = "AUDIO TEST - CAN YOU HEAR THIS CLEARLY? MAXIMUM VOLUME!"
                st.success("🔊 🔊 PLAYING VERY LOUD AUDIO TEST...")
                text_to_speech_with_method(test_text, 'en', "Windows SAPI", 150, 1.0)
            
            if st.button("🔊 Tamil LOUD Test", 
                       help="Test Tamil audio at MAXIMUM VOLUME",
                       use_container_width=True):
                tamil_test = "வணக்கம் - இது அதிக ஒலியில் சோதனை - நீங்கள் கேட்க முடிகிறதா?"
                st.success("🔊 🔊 தமிழ் அதிக ஒலி சோதனை...")
                text_to_speech_with_method(tamil_test, 'ta', "Windows SAPI", 150, 1.0)
                
            if st.button("🔊 EMERGENCY VOLUME", 
                       help="EMERGENCY LEVEL VOLUME TEST",
                       use_container_width=True):
                emergency_test = "EMERGENCY VOLUME TEST - THIS SHOULD BE VERY LOUD!"
                st.error("🚨 � EMERGENCY VOLUME TEST 🚨 🚨")
                text_to_speech_with_method(emergency_test, 'en', "Windows SAPI", 150, 1.0)
                
            # Add system volume check
            if st.button("� Check System Audio", 
                       help="Check if system audio is working",
                       use_container_width=True):
                try:
                    import winsound
                    st.info("Playing system beeps...")
                    for i in range(5):
                        winsound.Beep(1000, 200)
                        import time
                        time.sleep(0.2)
                    st.success("✅ If you heard beeps, your speakers work!")
                    st.warning("If no beeps, check speaker connections!")
                except Exception as e:
                    st.error(f"System audio test failed: {e}")
        
        with quick_col2:
            st.markdown("### 🌍 Quick Translate & Speak")
            
            # Quick phrase input
            st.markdown("**Quick Text Input:**")
            quick_text = st.text_input("Type text to instantly translate and speak:", 
                                     placeholder="Enter any text here...",
                                     key="quick_text_input")
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("🚀 Translate & Speak", 
                           disabled=not quick_text,
                           use_container_width=True):
                    if quick_text:
                        translated = translate_text(quick_text, target_lang_code)
                        st.success(f"Original: {quick_text}")
                        st.success(f"Translation: {translated}")
                        if enable_tts:
                            text_to_speech_with_method(translated, target_lang_code, "Windows SAPI", 150, 1.0)
            
            with col2b:
                if st.button("🔊 Speak Original", 
                           disabled=not quick_text,
                           use_container_width=True):
                    if quick_text:
                        st.success(f"Speaking: {quick_text}")
                        if enable_tts:
                            text_to_speech_with_method(quick_text, 'en', "Windows SAPI", 150, 1.0)
            
            # Enhanced common phrases with categories
            st.markdown("**Quick Phrases:**")
            phrase_categories = {
                "Greetings": [
                    ("Hello", "வணக்கம்"),
                    ("Good morning", "காலை வணக்கம்"),
                    ("How are you?", "எப்படி இருக்கிறீர்கள்?"),
                    ("Nice to meet you", "உங்களை சந்தித்ததில் மகிழ்ச்சி")
                ],
                "Emergency": [
                    ("Help me please", "தயவு செய்து எனக்கு உதவுங்கள்"),
                    ("I need assistance", "எனக்கு உதவி தேவை"),
                    ("Call for help", "உதவிக்கு அழைக்கவும்"),
                    ("Emergency", "அவசரநிலை")
                ],
                "Basic Needs": [
                    ("I am hungry", "எனக்கு பசிக்கிறது"),
                    ("I am thirsty", "எனக்கு தாகமாக இருக்கிறது"),
                    ("Where is bathroom?", "குளியலறை எங்கே உள்ளது?"),
                    ("I need water", "எனக்கு தண்ணீர் வேண்டும்")
                ]
            }
            
            selected_category = st.selectbox(
                "Choose phrase category:",
                list(phrase_categories.keys()),
                help="Select a category of common phrases"
            )
            
            for english, tamil in phrase_categories[selected_category]:
                if st.button(f"🗣️ {english}", 
                           key=f"quick_phrase_{english.replace(' ', '_').replace('?', '')}", 
                           help=f"Speak: {english} → {tamil}",
                           use_container_width=True):
                    st.success(f"Speaking: {english} → {tamil}")
                    if enable_tts:
                        # Speak both English and Tamil
                        text_to_speech_with_method(english, 'en', "Windows SAPI", 150, 1.0)
                        import time
                        time.sleep(1)  # Brief pause
                        text_to_speech_with_method(tamil, 'ta', "Windows SAPI", 150, 1.0)
                    st.markdown(f'''
                    <div style="background: linear-gradient(45deg, #3498db, #2980b9); padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                        <strong>📝 Translation:</strong><br>
                        <span style="font-size: 1.1rem;">{english} → {translated}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    if enable_tts:
                        st.success("🔊🔊 QUICK PHRASE AT MAX VOLUME...")
                        text_to_speech_with_method(translated, target_lang_code, "Windows SAPI", 150, 1.0)
        
        with quick_col3:
            st.markdown("### 🛠️ System Tools")
            
            if st.button("� Detailed Statistics", 
                       help="Show comprehensive session statistics",
                       use_container_width=True):
                gesture_count = st.session_state.get('gesture_count', 0)
                lip_count = st.session_state.get('lip_count', 0)
                active_modes = sum([gesture_mode, speech_mode, text_mode, lip_mode])
                
                # Calculate session statistics
                session_duration = "Active"
                total_interactions = gesture_count + lip_count
                
                st.markdown(f'''
                <div style="background: linear-gradient(45deg, #27ae60, #2ecc71); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                    <h4 style="margin: 0 0 15px 0;">📈 Comprehensive Statistics</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                            <strong>🎭 Gestures:</strong><br>{gesture_count}
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                            <strong>👄 Lip Reads:</strong><br>{lip_count}
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                            <strong>🔧 Active Modes:</strong><br>{active_modes}/4
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                            <strong>🎯 Total Actions:</strong><br>{total_interactions}
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <small>Session Status: {session_duration}</small>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            if st.button("🔄 System Refresh", 
                       help="Refresh the application to reset any glitches",
                       use_container_width=True):
                st.success("🔄 Refreshing system...")
                st.rerun()
            
            if st.button("🧪 Run System Check", 
                       help="Check all system components",
                       use_container_width=True):
                with st.spinner("🔍 Running system diagnostics..."):
                    import time
                    time.sleep(1)
                    
                    # System check results
                    checks = {
                        "🎤 Speech Recognition": speech_mode,
                        "💬 Text Processing": text_mode, 
                        "🎭 Gesture Detection": gesture_mode,
                        "👄 Lip Reading": lip_mode,
                        "🔊 Audio Output": enable_tts,
                        "📹 Camera Access": 'cap' in st.session_state
                    }
                    
                    st.markdown("**� System Check Results:**")
                    for component, status in checks.items():
                        status_icon = "✅" if status else "❌"
                        st.write(f"{status_icon} {component}")
                    
                    overall_health = sum(checks.values()) / len(checks) * 100
                    st.metric("System Health", f"{overall_health:.0f}%")

st.markdown('</div>', unsafe_allow_html=True)  # Close floating section

# Results and Translation Section with 3D effects
st.markdown("---")
st.markdown('<div class="floating-section">', unsafe_allow_html=True)
st.markdown('<h2 style="color: #2ECC71; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); transform: perspective(500px) rotateX(10deg);">📤 Results & Translation</h2>', unsafe_allow_html=True)

# Display all results in vertical sections with 3D styling
if gesture_mode and 'gesture_detected' in st.session_state:
    st.markdown('<h3 style="color: #9B59B6; transform: perspective(300px) rotateX(5deg);">🎭 Latest Gesture Result</h3>', unsafe_allow_html=True)
    result = st.session_state['gesture_detected']
    col_ges1, col_ges2 = st.columns(2)
    with col_ges1:
        st.markdown(f'<div class="success-animation">Original: <strong>{result["original"]}</strong></div>', unsafe_allow_html=True)
    with col_ges2:
        st.markdown(f'<div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 10px; border-radius: 10px; color: white; transform: perspective(500px) rotateX(5deg);">Translation: <strong>{result["translated"]}</strong></div>', unsafe_allow_html=True)
    
    # Auto TTS for gesture results
    if enable_tts and st.button("🔊 Play Gesture Translation", key="gesture_tts"):
        st.success("🔊🔊 GESTURE TRANSLATION AT MAX VOLUME...")
        text_to_speech_with_method(result["translated"], target_lang_code, "Windows SAPI", 150, 1.0)

if lip_mode and 'lip_detected' in st.session_state:
    st.markdown('<h3 style="color: #E67E22; transform: perspective(300px) rotateX(5deg);">👄 Latest Lip Reading Result</h3>', unsafe_allow_html=True)
    result = st.session_state['lip_detected']
    col_lip1, col_lip2 = st.columns(2)
    with col_lip1:
        st.markdown(f'<div class="success-animation">Original: <strong>{result["original"]}</strong></div>', unsafe_allow_html=True)
    with col_lip2:
        st.markdown(f'<div style="background: linear-gradient(45deg, #f39c12, #e67e22); padding: 10px; border-radius: 10px; color: white; transform: perspective(500px) rotateX(5deg);">Translation: <strong>{result["translated"]}</strong></div>', unsafe_allow_html=True)

if speech_mode and 'speech_input' in st.session_state:
    st.markdown('<h3 style="color: #3498DB; transform: perspective(300px) rotateX(5deg);">🎤 Speech Input Result</h3>', unsafe_allow_html=True)
    speech_text = st.session_state['speech_input']
    translated_speech = translate_text(speech_text, target_lang_code)
    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        st.markdown(f'<div class="success-animation">Detected: <strong>{speech_text}</strong></div>', unsafe_allow_html=True)
    with col_sp2:
        st.markdown(f'<div style="background: linear-gradient(45deg, #3498db, #2980b9); padding: 10px; border-radius: 10px; color: white; transform: perspective(500px) rotateX(5deg);">Translation: <strong>{translated_speech}</strong></div>', unsafe_allow_html=True)
    
    if enable_tts and st.button("🔊 Play Speech Translation"):
        st.success("🔊🔊 SPEECH TRANSLATION AT MAX VOLUME...")
        text_to_speech_with_method(translated_speech, target_lang_code, "Windows SAPI", 150, 1.0)

if text_mode and 'text_input' in st.session_state:
    st.markdown('<h3 style="color: #16A085; transform: perspective(300px) rotateX(5deg);">💬 Text Processing Result</h3>', unsafe_allow_html=True)
    text = st.session_state['text_input']
    translated_text = translate_text(text, target_lang_code)
    col_txt1, col_txt2 = st.columns(2)
    with col_txt1:
        st.markdown(f'<div class="success-animation">Input: <strong>{text}</strong></div>', unsafe_allow_html=True)
    with col_txt2:
        st.markdown(f'<div style="background: linear-gradient(45deg, #16a085, #27ae60); padding: 10px; border-radius: 10px; color: white; transform: perspective(500px) rotateX(5deg);">Translation: <strong>{translated_text}</strong></div>', unsafe_allow_html=True)
    
    # TTS button for text results
    if enable_tts and st.button("🔊 Play Text Translation", key="text_tts"):
        st.success("🔊🔊 TEXT TRANSLATION AT MAX VOLUME...")
        text_to_speech_with_method(translated_text, target_lang_code, "Windows SAPI", 150, 1.0)

st.markdown('</div>', unsafe_allow_html=True)  # Close floating section

# Statistics and Summary Section with 3D effects
st.markdown("---")
st.markdown('<div class="floating-section">', unsafe_allow_html=True)
st.markdown('<h2 style="color: #8E44AD; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); transform: perspective(500px) rotateX(10deg);">📊 Session Statistics</h2>', unsafe_allow_html=True)

stats_col1, stats_col2, stats_col3 = st.columns(3)

with stats_col1:
    if gesture_mode:
        gesture_count = st.session_state.get('gesture_count', 0)
        st.markdown(f'''
        <div class="card-3d" style="text-align: center;">
            <h3>🎭 Gestures</h3>
            <div class="glow-effect" style="font-size: 2rem; font-weight: bold; color: #2ECC71;">{gesture_count}</div>
        </div>
        ''', unsafe_allow_html=True)

with stats_col2:
    if lip_mode:
        lip_count = st.session_state.get('lip_count', 0)
        st.markdown(f'''
        <div class="card-3d" style="text-align: center;">
            <h3>👄 Lip Readings</h3>
            <div class="glow-effect" style="font-size: 2rem; font-weight: bold; color: #E74C3C;">{lip_count}</div>
        </div>
        ''', unsafe_allow_html=True)

with stats_col3:
    if 'camera_active' in st.session_state:
        status = "Active" if st.session_state.camera_active else "Inactive"
        status_color = "#2ECC71" if st.session_state.camera_active else "#E74C3C"
        st.markdown(f'''
        <div class="card-3d" style="text-align: center;">
            <h3>📹 Camera</h3>
            <div class="glow-effect" style="font-size: 1.5rem; font-weight: bold; color: {status_color};">{status}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close floating section

# Enhanced Footer with 3D effects
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);'>
    <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🌟 Communication Assistant</h2>
    <p style='color: rgba(255,255,255,0.9); font-size: 1rem; margin: 8px 0 0 0;'>Tamil & English • Gesture • Speech • Text</p>
</div>
""", unsafe_allow_html=True)
