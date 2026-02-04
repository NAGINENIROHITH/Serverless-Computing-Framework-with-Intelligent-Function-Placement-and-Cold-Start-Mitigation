"""
Flask web server for Intelligent Serverless Framework UI.
Serves the dashboard and provides API proxy.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import os
from loguru import logger

app = Flask(__name__)
CORS(app)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')


@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/proxy/<path:endpoint>', methods=['GET', 'POST'])
def api_proxy(endpoint):
    """
    Proxy API requests to avoid CORS issues.
    Frontend calls /api/proxy/... which forwards to backend API.
    """
    try:
        # Forward GET requests
        if request.method == 'GET':
            response = requests.get(
                f"{API_BASE_URL}/{endpoint}",
                params=request.args,
                timeout=10
            )
        # Forward POST requests
        elif request.method == 'POST':
            response = requests.post(
                f"{API_BASE_URL}/{endpoint}",
                json=request.get_json(),
                timeout=10
            )
        
        return jsonify(response.json()), response.status_code
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API proxy error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ui_server': 'running',
        'api_endpoint': API_BASE_URL
    })


@app.route('/config')
def config():
    """Get UI configuration"""
    return jsonify({
        'api_base_url': API_BASE_URL,
        'refresh_interval': 30000,  # 30 seconds
        'features': {
            'realtime_updates': True,
            'notifications': True,
            'export_data': True
        }
    })


if __name__ == '__main__':
    port = int(os.getenv('UI_PORT', 3000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting UI server on port {port}")
    logger.info(f"API endpoint: {API_BASE_URL}")
    logger.info(f"Dashboard will be available at http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

