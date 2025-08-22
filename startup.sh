#!/bin/bash

# Streamlit Cloud Startup Script
echo "🚀 Starting AI Multimodal Assistant..."

# Set environment variables for cloud deployment
export OPENCV_LOG_LEVEL=SILENT
export STREAMLIT_SERVER_ENABLE_CORS=false

# Start the application
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
