# 🤖 Multimodal Communication Assistant

A comprehensive real-time gesture recognition and communication system built with Streamlit, MediaPipe, and advanced computer vision technologies.

## 🌟 Features

### 🎯 Core Capabilities
- **Real-time Gesture Recognition**: Advanced hand gesture detection and classification
- **Video-based Lip Reading**: Intelligent lip movement analysis for speech detection
- **Text-to-Speech (TTS)**: Multi-method audio output with maximum volume support
- **Multilingual Translation**: Tamil and English language support
- **Interactive UI**: Modern 3D-styled Streamlit interface

### 🔧 Technologies Used
- **Frontend**: Streamlit with custom CSS animations
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Pre-trained gesture classification models
- **Audio Processing**: Multiple TTS engines (Windows SAPI, pyttsx3, VBS)
- **Translation**: Google Translate API
- **Deployment**: Docker support included

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Windows OS (for optimal TTS support)
- Webcam for gesture and lip reading

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gesture-recognition.git
cd gesture-recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
project/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container deployment
├── README.md             # Project documentation
├── data/                 # Training data and datasets
├── models/               # Pre-trained ML models
│   └── sign_classifier.pkl
├── static/               # Static assets and animations
├── utils/                # Utility functions
│   └── mediapipe_helpers.py
└── clean_file.py         # Data preprocessing utilities
```

## 🎮 Usage

### 1. Gesture Recognition Mode
- Enable camera access
- Position your hand in front of the camera
- Perform gestures for real-time recognition
- Get instant translation and audio feedback

### 2. Lip Reading Mode
- Enable video-based lip reading
- Speak naturally in front of the camera
- System analyzes lip movements for speech detection
- Automatic translation and audio output

### 3. Text Input Mode
- Type text directly into the interface
- Select target language (Tamil/English)
- Get instant translation and TTS output
- Perfect for accessibility and communication aid

## ⚙️ Configuration

### Audio Settings
- Multiple TTS methods available
- Automatic volume maximization
- Fallback audio systems for reliability

### Language Support
- Tamil (ta)
- English (en)
- Easy extension for additional languages

### Camera Settings
- Adjustable frame rate
- Real-time processing optimization
- GPU acceleration when available

## 🔧 Technical Details

### Gesture Recognition Pipeline
1. **Frame Capture**: Real-time video input via OpenCV
2. **Hand Detection**: MediaPipe hand landmark detection
3. **Feature Extraction**: Landmark coordinate processing
4. **Classification**: ML model prediction
5. **Translation**: Multi-language output
6. **Audio Synthesis**: TTS with maximum volume

### Lip Reading System
1. **Facial Landmark Detection**: MediaPipe face mesh
2. **Lip Movement Analysis**: Real-time coordinate tracking
3. **Pattern Recognition**: Movement-based speech detection
4. **Speech Synthesis**: Text-to-speech conversion

## 🐳 Docker Deployment

```bash
# Build the container
docker build -t gesture-recognition .

# Run the application
docker run -p 8501:8501 gesture-recognition
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for computer vision frameworks
- Streamlit for the amazing web framework
- OpenCV community for image processing tools
- Google Translate for multilingual support

## 📞 Support

For support, email your-email@example.com or create an issue in this repository.

---

**Made with ❤️ for accessible communication technology**
