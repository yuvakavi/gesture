@echo off
echo 🚀 Starting AI Multimodal Assistant...
echo.
echo 📍 Changing to project directory...
cd /d "c:\Users\Yuva sri\project"

echo 🔍 Checking Python and Streamlit...
python -c "import streamlit; print('✅ Streamlit available')" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit not found. Installing...
    pip install streamlit
)

echo 🎬 Starting Streamlit app...
echo.
echo 📝 Note: Use Ctrl+C to stop the server
echo 🌐 App will open at: http://localhost:8513
echo.
python -m streamlit run app.py

pause
