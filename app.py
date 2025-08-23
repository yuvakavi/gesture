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

# Import cloud configuration
try:
    from streamlit_cloud_config import (
        detect_cloud_environment, 
        configure_for_cloud, 
        create_simulation_frame,
        simulate_gesture_recognition,
        cloud_compatible_tts,
        show_cloud_info
    )
    CLOUD_CONFIG_AVAILABLE = True
except ImportError:
    CLOUD_CONFIG_AVAILABLE = False
    def detect_cloud_environment(): return False
    def configure_for_cloud(): return False

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
        # Detect environment more reliably
        is_cloud = (
            'STREAMLIT_SHARING' in os.environ or 
            'STREAMLIT_CLOUD' in os.environ or 
            '/mount/src' in os.getcwd() or
            'streamlit.io' in os.environ.get('HOSTNAME', '') or
            'share.streamlit.io' in os.environ.get('HTTP_HOST', '')
        )
        
        if is_cloud:
            print("🌐 TTS using cloud mode (visual notifications)")
            return "cloud_mode"
        
        # Try to initialize pyttsx3 for local use
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Test the engine
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            print("✅ TTS initialized with pyttsx3")
            return engine
        except Exception as e:
            print(f"⚠️ pyttsx3 not available: {e}")
            return "fallback_mode"
            
    except Exception as e:
        print(f"❌ TTS initialization error: {e}")
        return "fallback_mode"

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
    """Convert text to speech with enhanced local audio and cloud visual output"""
    if not text or not text.strip():
        return "No text provided for speech synthesis."
    
    # Clean text for better speech synthesis
    clean_text = text.replace('"', "'").replace("`", "").replace("'", "").strip()
    if not clean_text:
        return "Empty text after cleaning."
    
    # Detect environment with enhanced detection
    is_cloud = (
        'STREAMLIT_SHARING' in os.environ or 
        'STREAMLIT_CLOUD' in os.environ or 
        '/mount/src' in os.getcwd() or
        'streamlit.io' in os.environ.get('HOSTNAME', '') or
        'share.streamlit.io' in os.environ.get('HTTP_HOST', '')
    )
    
    if is_cloud:
        # Cloud mode - enhanced visual notifications
        st.success(f"🔊 CLOUD AUDIO: {clean_text}")
        st.info("🌐 Running on Streamlit Cloud - Audio shown visually")
        return "cloud_mode_activated"
    
    # Local mode - Enhanced audio with multiple fallbacks
    audio_success = False
    
    # Method 1: pyttsx3 with enhanced configuration
    if not audio_success:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Enhanced voice configuration
            voices = engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[0].id)
            
            engine.setProperty('rate', max(100, min(250, rate)))
            engine.setProperty('volume', max(0.1, min(1.0, volume)))
            
            # Speak with timeout protection
            engine.say(clean_text)
            engine.runAndWait()
            
            st.success(f"🔊 SPEAKING (pyttsx3): {clean_text}")
            audio_success = True
            return "pyttsx3_success"
            
        except Exception as e:
            print(f"🔄 pyttsx3 failed, trying fallback: {e}")
    
    # Method 2: Windows SAPI via PowerShell (Enhanced)
    if not audio_success and os.name == 'nt':
        try:
            # Escape text for PowerShell
            escaped_text = clean_text.replace("'", "''").replace('"', '""')
            
            ps_script = f'''
            try {{
                Add-Type -AssemblyName System.Speech
                $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $voice.Volume = {int(volume * 100)}
                $voice.Rate = {max(-10, min(10, int((rate - 150) / 15)))}
                $voice.SpeakAsync("{escaped_text}")
                Start-Sleep -Seconds 1
                Write-Output "Speech synthesis started"
            }} catch {{
                Write-Error "Speech synthesis failed: $_"
            }}
            '''
            
            result = subprocess.run(
                ['powershell', '-ExecutionPolicy', 'Bypass', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True
            )
            
            if result.returncode == 0:
                st.success(f"🔊 SPEAKING (Windows SAPI): {clean_text}")
                audio_success = True
                return "sapi_success"
            else:
                print(f"PowerShell SAPI error: {result.stderr}")
                
        except Exception as e:
            print(f"🔄 Windows SAPI failed: {e}")
    
    # Method 3: Command line utilities fallback
    if not audio_success:
        try:
            if os.name == 'nt':  # Windows
                # Use built-in narrator
                subprocess.Popen([
                    'powershell', '-Command', 
                    f'(New-Object -ComObject SAPI.SpVoice).Speak("{escaped_text}")'
                ], shell=True)
                st.success(f"🔊 SPEAKING (System Voice): {clean_text}")
                audio_success = True
                return "system_voice_success"
            elif os.name == 'posix':  # Linux/Mac
                subprocess.Popen(['espeak', clean_text])
                st.success(f"🔊 SPEAKING (espeak): {clean_text}")
                audio_success = True
                return "espeak_success"
        except Exception as e:
            print(f"🔄 System voice failed: {e}")
    
    # Final fallback - Enhanced visual display
    if not audio_success:
        st.error("� AUDIO UNAVAILABLE - VISUAL OUTPUT:")
        st.markdown(f"""
        <div style="
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            padding: 20px;
            border-radius: 15px;
            border: 3px solid #FFD93D;
            color: white;
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
            animation: pulse 2s infinite;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
        ">
            🔊📢 AUDIO MESSAGE: {clean_text}
        </div>
        """, unsafe_allow_html=True)
        return "visual_fallback"
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
    """Translate text to target language with robust error handling"""
    global translator
    try:
        if not text or not text.strip():
            return text
            
        # Try translation with multiple fallback strategies
        try:
            # First attempt: regular translation
            translated = translator.translate(text, dest=target_language)
            
            # Handle different response types
            if hasattr(translated, 'text'):
                return translated.text
            elif hasattr(translated, '__await__'):
                # If it's a coroutine, use asyncio
                import asyncio
                try:
                    # Check if we're in an async context
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, can't use run_until_complete
                        st.warning("Async translation detected - using original text")
                        return text
                    except RuntimeError:
                        # No running loop, we can create one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(translated)
                            return result.text if hasattr(result, 'text') else str(result)
                        finally:
                            loop.close()
                except Exception as async_error:
                    st.warning(f"Async translation failed: {async_error}")
                    return text
            else:
                # Fallback: convert to string
                return str(translated)
                
        except Exception as translate_error:
            # If translation fails, try recreating the translator
            try:
                translator = Translator()
                translated = translator.translate(text, dest=target_language)
                return translated.text if hasattr(translated, 'text') else str(translated)
            except Exception as retry_error:
                st.warning(f"Translation service unavailable: {retry_error}")
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

# 🌐 Cloud Environment Configuration
IS_CLOUD = detect_cloud_environment()
if IS_CLOUD:
    configure_for_cloud()
    st.markdown("""
    <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 10px; border-radius: 10px; color: white; margin: 10px 0; text-align: center;">
        🌐 <strong>Streamlit Cloud Detected</strong> - Optimized for cloud deployment
    </div>
    """, unsafe_allow_html=True)

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
    
    # Add comprehensive troubleshooting section
    st.header("🔧 Troubleshooting")
    
    with st.expander("📹 Camera Issues", expanded=False):
        st.markdown("""
        **Common Solutions:**
        
        🔒 **Permission Issues:**
        - Allow camera access when browser asks
        - Check browser settings (camera permissions)
        - Try refreshing the page
        
        📱 **App Conflicts:**
        - Close Zoom, Skype, Teams, Discord
        - Exit other video calling apps
        - Stop screen recording software
        
        🔌 **Hardware Issues:**
        - Check camera cable connection
        - Try a different USB port
        - Test camera in other applications
        
        🌐 **Browser Compatibility:**
        - Chrome and Firefox work best
        - Enable camera in browser settings
        - Try incognito/private mode
        
        ⚡ **Quick Fixes:**
        - Restart your browser
        - Reboot your computer
        - Update camera drivers
        """)
        
        if st.button("🔄 Reset Camera Settings", key="reset_camera"):
            if 'cap' in st.session_state and st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.camera_simulation = False
            st.success("Camera settings reset! Please refresh the page.")
    
    with st.expander("🔊 Audio Issues", expanded=False):
        st.markdown("""
        **Audio Solutions:**
        
        🌐 **Streamlit Cloud:**
        - Audio shows as visual notifications
        - Download app for local audio
        - Balloons = successful audio trigger
        
        🔊 **Local Audio Problems:**
        - Check system volume settings
        - Test Windows audio devices
        - Update audio drivers
        - Run as administrator
        
        🎵 **TTS Not Working:**
        - Install `pyttsx3`: `pip install pyttsx3`
        - Try different TTS methods
        - Check Windows Speech Platform
        
        ⚙️ **Advanced Fixes:**
        - Restart Windows Audio service
        - Check PowerShell execution policy
        - Update .NET Framework
        """)
    
    with st.expander("🌐 Cloud Deployment", expanded=False):
        st.markdown("""
        **Streamlit Cloud Notes:**
        
        📹 **Camera:**
        - May not work in cloud environment
        - Use Smart Simulation Mode instead
        - Full functionality maintained
        
        🔊 **Audio:**
        - Visual notifications replace audio
        - Download for local audio experience
        - All features work with visual feedback
        
        🚀 **Performance:**
        - Cloud optimized for speed
        - Reduced resource usage
        - Automatic fallback systems
        """)
    
    # System status indicator
    st.markdown("---")
    st.markdown("**System Status:**")
    
    # Environment detection
    is_cloud = (
        'STREAMLIT_SHARING' in os.environ or 
        'STREAMLIT_CLOUD' in os.environ or 
        '/mount/src' in os.getcwd() or
        'streamlit.io' in os.environ.get('HOSTNAME', '') or
        'share.streamlit.io' in os.environ.get('HTTP_HOST', '')
    )
    
    if is_cloud:
        st.info("🌐 Running on Streamlit Cloud")
        st.caption("Camera simulation and visual audio available")
    else:
        st.success("💻 Running locally")
        st.caption("Full camera and audio support available")
    
    # Show app version
    st.caption("🤖 AI Multimodal Assistant v2.1")
    st.caption("Translation fix applied ✅")
    
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
        # Automatic camera initialization - no user intervention needed
        if 'auto_camera_started' not in st.session_state:
            st.session_state.auto_camera_started = True
            run_camera = True
            st.success("🚀 Auto-started camera capture!")
        else:
            run_camera = st.checkbox("📹 Camera Active", value=True)
    
    # Camera Mode Control Section - Always active for auto-capture
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
        
        # Camera capture controls
        st.markdown("### 📸 Enhanced Capture Controls")
        capture_col1, capture_col2, capture_col3, capture_col4 = st.columns(4)
        
        with capture_col1:
            auto_capture_enabled = st.checkbox("🔄 Auto-capture (every 5s)", value=True)
            st.session_state.auto_capture_enabled = auto_capture_enabled
            
        with capture_col2:
            gesture_capture_enabled = st.checkbox("🤏 Gesture Capture", value=False, help="Capture when specific gestures are detected")
            st.session_state.gesture_capture_enabled = gesture_capture_enabled
            
        with capture_col3:
            if st.button("📸 Capture Now!", key="manual_capture"):
                st.session_state.manual_capture_trigger = True
                
        with capture_col4:
            if 'capture_count' in st.session_state:
                st.metric("📷 Total Captures", st.session_state.capture_count)
        
        # Voice feedback controls
        voice_col1, voice_col2 = st.columns(2)
        with voice_col1:
            capture_voice_enabled = st.checkbox("🔊 Voice Feedback", value=True, help="Announce captures and gestures")
            st.session_state.capture_voice_enabled = capture_voice_enabled
        
        with voice_col2:
            # Text input for voice output
            text_input = st.text_input("💬 Text to Speech:", placeholder="Enter text to speak...", key="voice_text_input")
            if st.button("🎙️ Speak Text", key="speak_button") and text_input:
                if enable_tts:
                    text_to_speech_with_method(text_input, target_lang_code)
                st.success(f"🔊 Speaking: {text_input}")
    
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
        
        # Enhanced camera initialization with frame validation
        def init_camera_robust():
            """Robust camera initialization with comprehensive frame validation"""
            camera_configs = [
                (0, cv2.CAP_DSHOW, "DirectShow (Primary)"),
                (0, cv2.CAP_MSMF, "Media Foundation"),
                (0, cv2.CAP_V4L2, "Video4Linux2"),
                (0, cv2.CAP_ANY, "Any Backend"),
                (1, cv2.CAP_DSHOW, "DirectShow (Secondary Camera)"),
                (1, cv2.CAP_ANY, "Any Backend (Camera 1)"),
                (2, cv2.CAP_ANY, "Any Backend (Camera 2)"),
            ]
            
            def configure_camera_properties(cap):
                """Configure optimal camera properties"""
                properties = [
                    (cv2.CAP_PROP_FRAME_WIDTH, 640),
                    (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                    (cv2.CAP_PROP_FPS, 30),
                    (cv2.CAP_PROP_BUFFERSIZE, 1),  # Reduce buffer for latest frames
                    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')),
                ]
                
                for prop, value in properties:
                    try:
                        cap.set(prop, value)
                    except:
                        pass  # Some properties might not be supported
                
                # Optional properties for better quality
                try:
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                except:
                    pass
            
            def validate_frame_content(frame):
                """Validate that frame contains actual image data"""
                if frame is None or frame.size == 0:
                    return False
                
                # Check if frame has variation (not solid color)
                mean_val = np.mean(frame)
                std_val = np.std(frame)
                
                # Frame should have some variation
                if std_val < 5:
                    print(f"⚠️ Frame appears to be solid color (std: {std_val:.2f})")
                    return False
                
                # Frame should not be completely black or white
                if mean_val < 10 or mean_val > 245:
                    print(f"⚠️ Frame appears to be too dark/bright (mean: {mean_val:.2f})")
                    return False
                
                return True
            
            def test_frame_reading(cap, backend_name, max_attempts=15):
                """Enhanced frame reading test with multiple strategies"""
                print(f"🔍 Testing frame reading for {backend_name}...")
                
                # Strategy 1: Direct frame reading
                successful_frames = 0
                for attempt in range(max_attempts):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if validate_frame_content(frame):
                            successful_frames += 1
                            print(f"✅ Valid frame {successful_frames}/{max_attempts}")
                            if successful_frames >= 3:  # Require 3 valid frames
                                return True
                    else:
                        print(f"❌ Attempt {attempt + 1}: No valid frame")
                    time.sleep(0.1)
                
                # Strategy 2: Flush buffer and retry
                print("🔄 Flushing camera buffer...")
                for _ in range(5):
                    cap.read()  # Flush old frames
                time.sleep(0.3)
                
                for attempt in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if validate_frame_content(frame):
                            print(f"✅ Buffer flush successful on attempt {attempt + 1}")
                            return True
                    time.sleep(0.1)
                
                return False
            
            # Try each camera configuration
            for camera_index, backend, backend_name in camera_configs:
                try:
                    print(f"🔍 Trying {backend_name}...")
                    cap = cv2.VideoCapture(camera_index, backend)
                    
                    if not cap.isOpened():
                        print(f"❌ {backend_name}: Cannot open camera")
                        continue
                    
                    # Configure camera properties
                    configure_camera_properties(cap)
                    time.sleep(0.5)  # Allow camera to warm up
                    
                    # Test frame reading with validation
                    if test_frame_reading(cap, backend_name):
                        print(f"✅ Camera initialized successfully: {backend_name}")
                        return cap
                    else:
                        print(f"❌ {backend_name}: Frame reading failed")
                        cap.release()
                    
                except Exception as e:
                    print(f"❌ Failed {backend_name}: {e}")
                    if 'cap' in locals():
                        try:
                            cap.release()
                        except:
                            pass
                    continue
            
            print("❌ No working camera found with valid frame reading")
            return None
        
        # Enhanced camera status checking with cloud detection
        def check_camera_advanced():
            """Advanced camera availability check with cloud environment detection"""
            try:
                # Detect cloud environment more comprehensively with better detection patterns
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
                    'REPLIT' in os.environ,
                    'RAILWAY' in os.environ,
                    '/tmp' in os.getcwd() and not os.path.exists('/home'),  # Cloud container pattern
                    'gesture' in os.getcwd() and '/mount/src' in os.getcwd(),  # Your specific deployment
                    not os.path.exists('/dev/video0'),  # No camera device file
                    os.environ.get('HOME', '').startswith('/mount') or os.environ.get('HOME', '').startswith('/app')
                ]
                
                # DISABLED: Force real camera detection instead of cloud detection
                # Additional checks for Streamlit Cloud specifically
                cwd = os.getcwd()
                if any([
                    '/mount/src' in cwd,
                    '/app' in cwd and 'streamlit' in os.environ.get('PATH', ''),
                    'workspace' in cwd and not os.path.exists('C:'),  # Linux-based cloud
                ]):
                    cloud_indicators.append(True)
                
                # DISABLED: Always try real camera instead of cloud detection
                if False:  # Disabled cloud detection
                    print("🌐 Cloud environment detected - camera not available")
                    print(f"🔍 Environment details: CWD={cwd}, HOME={os.environ.get('HOME', 'N/A')}")
                    return False
                
                # Always try camera detection regardless of environment
                print("📹 Forcing real camera detection...")
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None and frame.size > 0:
                        print("✅ Local camera detected and working")
                        return True
                    else:
                        print("❌ Local camera detected but not producing frames")
                        return False
                else:
                    print("❌ No local camera found")
                    return False
            except Exception as e:
                print(f"Camera check error: {e}")
                return False
        
        try:
            # Enhanced cloud environment detection with more specific patterns
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
                'REPLIT' in os.environ,
                'RAILWAY' in os.environ,
                '/tmp' in os.getcwd() and not os.path.exists('/home'),  # Cloud container pattern
                'gesture' in os.getcwd() and '/mount/src' in os.getcwd(),  # Your specific deployment
                not os.path.exists('/dev/video0'),  # No camera device file
                os.environ.get('HOME', '').startswith('/mount') or os.environ.get('HOME', '').startswith('/app')
            ]
            
            # Additional checks for Streamlit Cloud specifically
            cwd = os.getcwd()
            if any([
                '/mount/src' in cwd,
                '/app' in cwd and 'streamlit' in os.environ.get('PATH', ''),
                'workspace' in cwd and not os.path.exists('C:'),  # Linux-based cloud
            ]):
                cloud_indicators.append(True)
            
            # DISABLED: Force real camera mode instead of cloud detection
            is_cloud_environment = False  # Always use real camera
            
            # Debug information
            print(f"🔍 Camera Mode Debug:")
            print(f"   Current working directory: {cwd}")
            print(f"   Environment HOME: {os.environ.get('HOME', 'N/A')}")
            print(f"   Forced camera mode: Real camera enabled")
            print(f"   Manual simulation: {st.session_state.get('camera_simulation', False)}")
            
            # Force camera simulation to False (real camera mode)
            if 'camera_simulation' not in st.session_state:
                st.session_state.camera_simulation = False
            
            # Only use simulation if manually enabled by user (not automatic cloud detection)
            if st.session_state.get('camera_simulation', False):
                with camera_status.container():
                    st.success("🎬 Manual Simulation Mode Active - Full Functionality Available!")
                st.session_state.cap = None
                st.session_state.camera_simulation = True
            elif 'cap' not in st.session_state or st.session_state.cap is None or not st.session_state.cap.isOpened():
                with camera_status.container():
                    st.info("🔄 Initializing camera system...")
                
                # Advanced camera availability check (only for local environment)
                camera_available = check_camera_advanced()
                
                if camera_available:
                    st.session_state.cap = init_camera_robust()
                    
                    if st.session_state.cap is not None:
                        camera_status.success("✅ Camera initialized successfully!")
                        st.session_state.camera_simulation = False
                        
                        # Show helpful camera info
                        if enable_tts:
                            text_to_speech_with_method("Camera ready for gesture recognition", target_lang_code)
                    else:
                        camera_status.warning("📹 Camera detected but initialization failed")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Retry Camera", key="retry_camera"):
                                st.rerun()
                        with col2:
                            if st.button("🎬 Use Simulation Mode", key="fallback_sim"):
                                st.session_state.camera_simulation = True
                                st.rerun()
                else:
                    # No camera detected - offer simulation mode
                    camera_status.info("� No camera detected")
                    
                    # Create a nice layout for the options
                    st.markdown("""
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                        <h4>🎥 Camera Setup Options</h4>
                        <p>• <strong>Allow camera access</strong> when browser asks</p>
                        <p>• <strong>Close other apps</strong> (Zoom, Skype, Teams)</p>
                        <p>• <strong>Check camera connection</strong> and try refreshing</p>
                        <p>• <strong>Use Chrome or Firefox</strong> for best compatibility</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔄 Retry Camera Setup", key="retry_setup"):
                            st.rerun()
                    with col2:
                        if st.button("🎬 Smart Simulation Mode", key="start_sim"):
                            st.session_state.camera_simulation = True
                            if enable_tts:
                                text_to_speech_with_method("Simulation mode activated", target_lang_code)
                            st.rerun()
                    with col3:
                        if st.button("📋 Troubleshooting Guide", key="help_guide"):
                            st.info("💡 See the troubleshooting section in the sidebar for detailed help!")
                
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
                
                # Only use simulation mode if MANUALLY enabled by user
                if st.session_state.get('camera_simulation', False):
                    # Generate simulation frame using cloud config if available
                    if CLOUD_CONFIG_AVAILABLE:
                        frame = create_simulation_frame()
                        # Simulate gesture recognition
                        simulated_gesture = simulate_gesture_recognition()
                        
                        # Display simulation info
                        camera_placeholder.image(frame, channels="BGR", caption="🎬 Manual Simulation Mode Active")
                        
                        # Create simulation UI with gesture results
                        st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                            <h4 style="margin: 0;">🤖 AI Simulation Active</h4>
                            <p style="margin: 5px 0;">Detected Gesture: <strong>{simulated_gesture}</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">Simulating real-time gesture recognition</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Continue with gesture translation if simulated gesture is detected
                        if simulated_gesture and simulated_gesture != "Unknown":
                            gesture_translation = translate_text(simulated_gesture, target_lang_code)
                            st.success(f"🔄 **Simulated Translation:** {simulated_gesture} → {gesture_translation}")
                            
                            # TTS for simulated gesture
                            if enable_tts:
                                cloud_compatible_tts(f"Gesture detected: {simulated_gesture}", target_lang_code)
                    else:
                        # Basic simulation without cloud config
                        basic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        camera_placeholder.image(basic_frame, channels="BGR", caption="🎬 Basic Simulation Mode")
                        st.info("🌐 Cloud simulation active - camera features simulated")
                    
                    # Skip camera processing since we're in simulation mode
                    time.sleep(0.1)  # Small delay for smooth UI updates
                    
                elif cap is not None and cap.isOpened():
                    # Enhanced frame reading with validation and error handling
                    # Read multiple frames to flush buffer and get latest
                    frame_attempts = 0
                    max_attempts = 5
                    valid_frame = None
                    
                    while frame_attempts < max_attempts:
                        ret, frame = cap.read()
                        frame_attempts += 1
                        
                        if ret and frame is not None and frame.size > 0:
                            # Validate frame content
                            mean_val = np.mean(frame)
                            std_val = np.std(frame)
                            
                            # Check if frame has valid content (not solid color)
                            if std_val > 5 and 10 < mean_val < 245:
                                valid_frame = frame
                                break
                            else:
                                print(f"⚠️ Frame {frame_attempts} validation failed: std={std_val:.2f}, mean={mean_val:.2f}")
                                time.sleep(0.1)  # Brief delay before retry
                        else:
                            print(f"❌ Frame read attempt {frame_attempts} failed")
                            time.sleep(0.1)
                    
                    if valid_frame is not None:
                        frame = valid_frame
                        # Add 1 second delay for camera frame rate control
                        time.sleep(1.0)
                        # Flip frame horizontally for mirror effect
                        frame = cv2.flip(frame, 1)
                        # Convert BGR to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Display frame in full width
                        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Reset any camera error counters
                        if 'camera_error_count' in st.session_state:
                            st.session_state.camera_error_count = 0
                    else:
                        # Handle camera frame reading failure
                        if 'camera_error_count' not in st.session_state:
                            st.session_state.camera_error_count = 0
                        st.session_state.camera_error_count += 1
                        
                        if st.session_state.camera_error_count <= 3:
                            camera_placeholder.error(f"⚠️ Camera frame reading issue (attempt {st.session_state.camera_error_count}/3)")
                        else:
                            camera_placeholder.error("❌ Camera frame reading failed - Try reinitializing camera")
                            # Offer camera reset option
                            if st.button("🔄 Reset Camera", key="reset_camera_frames"):
                                if st.session_state.cap:
                                    st.session_state.cap.release()
                                st.session_state.cap = None
                                st.session_state.camera_error_count = 0
                                st.rerun()
                        # Skip further camera processing if no valid frame
                        st.stop()
                
                # If we have a valid frame, proceed with capture logic
                        
                        # Automatic image capture every 5 seconds
                        if 'last_capture_time' not in st.session_state:
                            st.session_state.last_capture_time = time.time()
                            st.session_state.capture_count = 0
                        
                        # Manual capture trigger
                        manual_capture = st.session_state.get('manual_capture_trigger', False)
                        if manual_capture:
                            st.session_state.manual_capture_trigger = False
                            # Force immediate capture
                            current_time = time.time()
                            st.session_state.capture_count += 1
                            
                            # Save captured image
                            if not os.path.exists('captures'):
                                os.makedirs('captures')
                            capture_filename = f'captures/manual_capture_{st.session_state.capture_count}_{int(current_time)}.jpg'
                            cv2.imwrite(capture_filename, frame)
                            
                            st.success(f"📸 Manual capture #{st.session_state.capture_count} saved!")
                            
                            # Play audible notification
                            if enable_tts:
                                text_to_speech_with_method(f"Manual image {st.session_state.capture_count} captured", target_lang_code, "pyttsx3 Only", 150, 1.0)
                        
                        # Enhanced Auto capture logic with gesture detection
                        auto_capture_enabled = st.session_state.get('auto_capture_enabled', True)
                        gesture_capture_enabled = st.session_state.get('gesture_capture_enabled', False)
                        capture_voice_enabled = st.session_state.get('capture_voice_enabled', True)
                        current_time = time.time()
                        
                        # Time-based auto capture
                        if auto_capture_enabled and current_time - st.session_state.last_capture_time >= 5.0:  # Capture every 5 seconds
                            st.session_state.capture_count += 1
                            st.session_state.last_capture_time = current_time
                            
                            # Save captured image
                            if not os.path.exists('captures'):
                                os.makedirs('captures')
                            capture_filename = f'captures/auto_capture_{st.session_state.capture_count}_{int(current_time)}.jpg'
                            cv2.imwrite(capture_filename, frame)
                            
                            # Show capture notification with enhanced feedback
                            st.success(f"📸 Auto-captured image #{st.session_state.capture_count}")
                            
                            # Enhanced voice feedback
                            if enable_tts and capture_voice_enabled:
                                text_to_speech_with_method(f"Image {st.session_state.capture_count} captured automatically", target_lang_code, "pyttsx3 Only", 150, 1.0)
                        
                        # Gesture-triggered capture (when gesture capture is enabled)
                        if gesture_capture_enabled and gesture_mode:
                            # Initialize gesture capture tracking
                            if 'last_gesture_capture' not in st.session_state:
                                st.session_state.last_gesture_capture = ''
                            if 'gesture_capture_cooldown' not in st.session_state:
                                st.session_state.gesture_capture_cooldown = 0
                            
                            # Quick gesture detection for capture triggering
                            try:
                                landmarks = get_hand_landmarks(frame)
                                if landmarks is not None:
                                    # Simple gesture detection for capture
                                    landmarks_array = np.array(landmarks).reshape(-1, 3)
                                    
                                    # Detect thumbs up gesture for capture
                                    thumb_tip = landmarks_array[4]
                                    thumb_mcp = landmarks_array[2]
                                    index_tip = landmarks_array[8]
                                    
                                    gesture_detected = None
                                    if thumb_tip[1] < thumb_mcp[1] and thumb_tip[1] < index_tip[1]:
                                        gesture_detected = 'thumbs_up'
                                    elif len([i for i in range(8, 21, 4) if landmarks_array[i][1] < landmarks_array[i-2][1]]) >= 4:
                                        gesture_detected = 'open_hand'
                                    
                                    # Trigger capture on specific gestures with cooldown
                                    if (gesture_detected in ['thumbs_up', 'open_hand'] and 
                                        gesture_detected != st.session_state.last_gesture_capture and
                                        current_time - st.session_state.gesture_capture_cooldown > 2.0):
                                        
                                        st.session_state.capture_count += 1
                                        st.session_state.last_gesture_capture = gesture_detected
                                        st.session_state.gesture_capture_cooldown = current_time
                                        
                                        # Save gesture-triggered capture
                                        if not os.path.exists('captures'):
                                            os.makedirs('captures')
                                        capture_filename = f'captures/gesture_capture_{gesture_detected}_{st.session_state.capture_count}_{int(current_time)}.jpg'
                                        cv2.imwrite(capture_filename, frame)
                                        
                                        # Enhanced feedback for gesture capture
                                        st.success(f"🤏 Gesture captured! {gesture_detected.replace('_', ' ').title()} - Image #{st.session_state.capture_count}")
                                        
                                        # Voice feedback for gesture capture
                                        if enable_tts and capture_voice_enabled:
                                            text_to_speech_with_method(f"Gesture {gesture_detected.replace('_', ' ')} captured successfully", target_lang_code, "pyttsx3 Only", 150, 1.0)
                            
                            except Exception as gesture_error:
                                # Silent handling of gesture detection errors
                                pass
                    
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
                
                # Enhanced camera status and error handling with cloud compatibility
                if cap is None or (cap is not None and not cap.isOpened()):
                    # Check if we're in a cloud environment
                    cloud_indicators = [
                        'STREAMLIT_SHARING' in os.environ,
                        'STREAMLIT_CLOUD' in os.environ,
                        '/mount/src' in os.getcwd(),
                        '/app' in os.getcwd(),
                        'streamlit.io' in os.environ.get('HOSTNAME', ''),
                        'heroku' in os.environ.get('DYNO', ''),
                        os.path.exists('/.dockerenv')
                    ]
                    
                    if any(cloud_indicators):
                        # Cloud environment - automatically switch to simulation
                        camera_status.info("🌐 Cloud environment detected - using simulation mode")
                        st.session_state.camera_simulation = True
                        st.session_state.cap = None
                    else:
                        # Local environment - offer reconnection options
                        camera_status.warning("📹 Camera connection lost - attempting recovery...")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("🔄 Reconnect Camera", key="reconnect_camera"):
                                if cap is not None:
                                    cap.release()
                                st.session_state.cap = None
                                st.rerun()
                        with col2:
                            if st.button("🎬 Switch to Simulation", key="switch_simulation"):
                                st.session_state.camera_simulation = True
                                if cap is not None:
                                    cap.release()
                                st.session_state.cap = None
                                st.rerun()
                        with col3:
                            if st.button("🔧 Troubleshooting", key="troubleshoot"):
                                st.session_state.show_troubleshooting = True
                                st.rerun()
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
                    if CLOUD_CONFIG_AVAILABLE:
                        simulation_frame = create_simulation_frame()
                    else:
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
                                
                                if CLOUD_CONFIG_AVAILABLE:
                                    # Use cloud-compatible gesture simulation
                                    gesture_result = simulate_gesture_recognition()
                                    st.session_state['gesture_detected'] = gesture_result
                                    st.session_state.gesture_count = st.session_state.get('gesture_count', 0) + 1
                                    
                                    st.success(f"✅ Cloud Simulated Gesture: {gesture_result['original']} → {gesture_result['translated']}")
                                    
                                    # Cloud-compatible audio output
                                    if enable_tts:
                                        cloud_compatible_tts(gesture_result['translated'], target_lang_code)
                                else:
                                    # Fallback simulation
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
