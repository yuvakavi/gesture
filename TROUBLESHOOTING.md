# 🚀 Streamlit Connection Troubleshooting Guide

## Quick Fix Commands

### Method 1: Standard Start
```bash
cd "c:\Users\Yuva sri\project"
python -m streamlit run app.py
```

### Method 2: Specific Port
```bash
python -m streamlit run app.py --server.port 8513
```

### Method 3: Reset and Restart
```bash
# Kill any existing Streamlit processes
taskkill /f /im python.exe
# Start fresh
python -m streamlit run app.py
```

## Common Connection Errors & Fixes

### ❌ "streamlit command not found"
**Fix:** Use `python -m streamlit` instead of just `streamlit`

### ❌ "Connection refused" or "Can't connect"
**Fixes:**
1. Check if another process is using the port
2. Try a different port: `--server.port 8514`
3. Restart your browser
4. Clear browser cache

### ❌ "Module not found" errors
**Fix:** Install requirements
```bash
pip install -r requirements.txt
```

### ❌ Camera access issues
**Fixes:**
1. Close other camera apps (Zoom, Skype, Teams)
2. Grant camera permissions in browser
3. Try simulation mode if camera unavailable

## Streamlit Cloud Deployment

### Quick Deploy
1. Push to GitHub: `git push origin main`
2. Visit: [share.streamlit.io](https://share.streamlit.io)
3. Connect repository: `yuvakavi/AI-Mood-Translator`
4. Deploy from `main` branch

### Environment Variables (if needed)
```
OPENCV_LOG_LEVEL=SILENT
STREAMLIT_SERVER_ENABLE_CORS=false
```

## Useful Commands

### Check if Streamlit is running
```bash
netstat -an | findstr :8513
```

### Force kill Streamlit
```bash
taskkill /f /im streamlit.exe
taskkill /f /im python.exe
```

### Test app syntax
```bash
python -c "import app; print('App syntax OK!')"
```

## Port Information
- **Default:** http://localhost:8513
- **Alternative:** http://localhost:8514, 8515, 8516, etc.
- **Network:** Available on local network IP

## Browser Compatibility
- ✅ **Chrome** (recommended)
- ✅ **Firefox** 
- ✅ **Edge**
- ⚠️ **Safari** (limited camera support)

---

## 🎯 Quick Start Script

Just double-click `start_app.bat` for automatic startup!

---

**Need help?** Check the GitHub repository: https://github.com/yuvakavi/AI-Mood-Translator
