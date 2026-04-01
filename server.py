# =============================================================================
# server.py
# EvilEye — Protect your sessions from evil doing | Lightweight HTTP Server for Jetson Nano
#
# Serves the dashboard HTML and exposes the state.json endpoint.
# Compatible with: Python 3.6+ / Jetson Nano 2GB / Flask
#
# Run:
#   python3 server.py
#
# Then open browser to: http://<jetson_ip>:5000
#
# =============================================================================

import json
import os
import sys
import argparse

# Check Flask dependency
try:
    from flask import Flask, jsonify, send_from_directory
except ImportError:
    print("[ERROR] Flask is not installed")
    print("[ERROR] Install it with: pip install flask")
    print("[ERROR] On Jetson: pip install flask")
    sys.exit(1)

from threading import Lock

app = Flask(__name__)

# Configuration (Jetson Nano optimized)
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          'outputs', 'state.json')
DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'dashboard.html')

state_lock = Lock()
request_count = 0

# ============================================================================ #
# CORS & Headers for dashboard polling
# ============================================================================ #

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to allow cross-origin requests from localhost."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

# ============================================================================ #
# Routes
# ============================================================================ #

@app.route('/')
def serve_dashboard():
    """Serve the dashboard HTML."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 
                               'dashboard.html')

@app.route('/outputs/state.json')
def get_state():
    """
    Return the current state.json from the tracking system.
    This endpoint is polled by the dashboard every 500ms.
    Returns the file written by student_tracker.py.
    """
    global request_count
    request_count += 1
    
    try:
        with state_lock:
            if os.path.exists(STATE_FILE):
                # Check file size to prevent reading huge files
                file_size = os.path.getsize(STATE_FILE)
                if file_size > 10 * 1024 * 1024:  # > 10MB
                    return jsonify({'error': 'State file too large'}), 500
                
                try:
                    # Read with timeout to prevent hangs on slow storage
                    with open(STATE_FILE, 'r') as f:
                        data = json.load(f)
                    return jsonify(data)
                except json.JSONDecodeError as je:
                    # File is corrupted, return default state
                    print(f"[Server] State file corrupted: {je}")
                    return jsonify({'error': 'Corrupted state file', 'status': 'recovering'}), 200
                except IOError as ie:
                    # Disk read error
                    print(f"[Server] Disk error reading state: {ie}")
                    return jsonify({'error': 'Disk read error'}), 500
            else:
                # Return default/idle state if file not yet created
                return jsonify({
                    'ts': '00:00:00',
                    'session_secs': 0,
                    'student_name': 'Student',
                    'present': False,
                    'face_found': False,
                    'attention_score': 0.0,
                    'attention_label': 'IDLE',
                    'attention_pct': 0,
                    'eye_state': 'UNKNOWN',
                    'head_state': 'UNKNOWN',
                    'yawn_state': 'NO_YAWN',
                    'yawn_count': 0,
                    'yawn_markers': [],
                    'head_down_count': 0,
                    'head_down_total_secs': 0,
                    'head_down_current_secs': 0,
                    'head_currently_down': False,
                    'phone_active': False,
                    'phone_events_count': 0,
                    'phone_total_secs': 0,
                    'phone_current_secs': 0,
                    'attention_ts': [],
                })
    except json.JSONDecodeError as e:
        print(f"[Server] JSON decode error in state.json: {e}")
        return jsonify({'error': 'Invalid JSON'}), 500
    except Exception as e:
        print(f"[Server] Error reading state.json: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'requests': request_count
    })

@app.route('/api/version')
def version():
    """API version endpoint."""
    return jsonify({
        'app': 'EvilEye - Protect your sessions from evil doing',
        'version': '1.0.0',
        'platform': 'Jetson Nano'
    })

# ============================================================================ #
# Main
# ============================================================================ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvilEye Dashboard Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (not recommended on Jetson)')
    args = parser.parse_args()
    
    print("[Server] ╔════════════════════════════════════════════════════════════╗")
    print("[Server] ║       EvilEye — Protect your sessions from evil doing       ║")
    print("[Server] ╚════════════════════════════════════════════════════════════╝")
    print(f"[Server] Starting on {args.host}:{args.port}")
    print(f"[Server] Dashboard:  http://localhost:{args.port}")
    print(f"[Server] API:        http://localhost:{args.port}/outputs/state.json")
    print(f"[Server] Health:     http://localhost:{args.port}/health")
    print(f"[Server] Mode:       {'DEBUG' if args.debug else 'PRODUCTION (single-threaded)'}")
    print(f"[Server] Memory:     Optimized for Jetson Nano 2GB")
    print("[Server] ")
    print("[Server] Ensure single_student_main.py is running in another terminal")
    print("[Server] Press Ctrl+C to stop")
    print("[Server] ")
    
    # For Jetson Nano with low memory: single-threaded, no debug
    app.run(
        host=args.host,
        port=args.port,
        threaded=False,        # Single-threaded to save memory
        debug=args.debug,
        use_reloader=False     # Disable auto-reload
    )
