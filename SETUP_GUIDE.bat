@echo off
REM ============================================================================
REM EvilEye — Setup & Run Guide
REM Windows batch file to help understand the deployment steps
REM ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║              EvilEye — Protect your sessions from evil doing           ║
echo ║                    For Jetson Nano Deployment                          ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.
echo NOTE: This project runs on JETSON NANO (Linux/ARM), not Windows.
echo This batch file is for informational purposes only.
echo.

echo ════════════════════════════════════════════════════════════════════════
echo STEP 1: SSH into your Jetson Nano
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   Windows PowerShell:
echo   ssh ubuntu@^<jetson-ip^>
echo.
echo   Example: ssh ubuntu@192.168.1.100
echo.
echo ════════════════════════════════════════════════════════════════════════
echo STEP 2: On Jetson, clone or copy the EvilEye repository
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   cd ~
echo   git clone ^<repo-url^> evileye
echo   cd classpulse-ai
echo.
echo ════════════════════════════════════════════════════════════════════════
echo STEP 3: Install Python dependencies
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   pip3 install -r requirements.txt
echo.
echo Note: TensorRT, CUDA, cuDNN come pre-installed with JetPack 4.6+
echo.
echo ════════════════════════════════════════════════════════════════════════
echo STEP 4: Prepare YOLOv8n TensorRT engine
echo ════════════════════════════════════════════════════════════════════════
echo.
echo Option A: Use pre-built engine (recommended)
echo   - Copy yolov8n.engine to root directory
echo.
echo Option B: Convert from ONNX
echo   yolo export model=yolov8n.pt format=onnx imgsz=320
echo   /usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx ^
echo     --saveEngine=yolov8n.engine --half
echo.
echo ════════════════════════════════════════════════════════════════════════
echo STEP 5: Run the system (open 2 SSH terminals)
echo ════════════════════════════════════════════════════════════════════════
echo.
echo TERMINAL 1 - Tracking Pipeline:
echo   python3 single_student_main.py
echo.
echo TERMINAL 2 - Dashboard Server:
echo   python3 server.py
echo.
echo Then access the dashboard from your browser:
echo   http://^<jetson-ip^>:5000
echo.
echo   Example: http://192.168.1.100:5000
echo.
echo ════════════════════════════════════════════════════════════════════════
echo SUPPORTED KEYBOARDS (Live Window)
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   q  = Quit
echo   r  = Reset session
echo   s  = Save screenshot
echo.
echo ════════════════════════════════════════════════════════════════════════
echo DASHBOARD FEATURES
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   - Real-time attention monitoring
echo   - Yawn detection and count
echo   - Head-down event tracking
echo   - Phone usage detection
echo   - Attention over time (chart)
echo   - Session summary and grade
echo   - Export data (JSON/CSV)
echo.
echo ════════════════════════════════════════════════════════════════════════
echo TROUBLESHOOTING
echo ════════════════════════════════════════════════════════════════════════
echo.
echo Q: Dashboard shows "Idle"
echo A: 1. Check both scripts are running
echo    2. Verify outputs/state.json exists: cat outputs/state.json
echo    3. Check camera is connected: ls /dev/video*
echo    4. Browser console (F12) for fetch errors
echo.
echo Q: "Cannot open webcam"
echo A: - Check camera: ls -la /dev/video*
echo    - Set permissions: sudo usermod -aG video $(whoami)
echo    - Reboot: sudo reboot
echo.
echo Q: Low FPS (less than 10)
echo A: - Check Jetson temp: sudo tegrastats
echo    - Lower INPUT_SIZE in single_student_main.py to 256
echo    - Reduce CONF_THRESH to 0.3
echo.
echo Q: "TensorRT engine not found"
echo A: - Verify yolov8n.engine in working directory
echo    - Engine must be built for Jetson (not x86)
echo.
echo ════════════════════════════════════════════════════════════════════════
echo SYSTEM REQUIREMENTS
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   - Jetson Nano 2GB or higher
echo   - JetPack 4.6+ (includes TensorRT 8.x, CUDA, cuDNN)
echo   - USB Webcam OR CSI Camera Module
echo   - Python 3.6+
echo   - 1.5+ GB free RAM
echo.
echo ════════════════════════════════════════════════════════════════════════

pause
