#!/usr/bin/env python3
"""
Discord Game Economy Analysis Tool - Web Interface
A simple web interface for running analysis and viewing results
"""

import os
import sys
import json
import subprocess
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utils if available
try:
    from gameanalytics.utils import logger, ensure_directory
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("discord_analysis_ui")
    
    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev_key_change_in_production')
app.config['PORT'] = int(os.getenv('PORT', '5000'))
app.config['DEBUG'] = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
app.config['THEME'] = os.getenv('THEME', 'dark')
app.config['IMAGE_SIZE_FACTOR'] = float(os.getenv('IMAGE_SIZE_FACTOR', '1.5'))

# Global variables for tracking task status
current_task = None
task_output = []
task_status = "idle"  # idle, running, completed, failed

def get_analysis_results():
    """Get list of available analysis results"""
    results = {
        'economy': [],
        'gambling': [],
        'category': [],
        'advanced': []
    }
    
    # Economy analysis results
    if os.path.exists('economy_analysis'):
        for file in os.listdir('economy_analysis'):
            if file.endswith('.html') or file.endswith('.png'):
                results['economy'].append(file)
    
    # Gambling analysis results
    if os.path.exists('gambling_analysis'):
        for file in os.listdir('gambling_analysis'):
            if file.endswith('.html') or file.endswith('.png'):
                results['gambling'].append(file)
    
    # Category analysis results
    if os.path.exists('category_analysis'):
        for file in os.listdir('category_analysis'):
            if file.endswith('.html') or file.endswith('.png'):
                results['category'].append(file)
    
    # Advanced analysis results
    if os.path.exists('advanced_analysis'):
        for file in os.listdir('advanced_analysis'):
            if file.endswith('.html') or file.endswith('.png'):
                results['advanced'].append(file)
    
    return results

def run_task(command, args=None):
    """Run a task in a separate thread and capture output"""
    global current_task, task_output, task_status
    
    task_output = []
    task_status = "running"
    
    # Build command
    if args:
        cmd = [command] + args
    else:
        cmd = command.split()
    
    logger.info(f"Running command: {cmd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture and store output
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            logger.info(f"Task output: {line}")
            task_output.append(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            task_status = "completed"
            logger.info("Task completed successfully")
        else:
            task_status = "failed"
            logger.error(f"Task failed with return code {return_code}")
    
    except Exception as e:
        task_status = "failed"
        logger.error(f"Error running task: {e}")
        task_output.append(f"Error: {str(e)}")
    
    finally:
        current_task = None

@app.route('/')
def index():
    """Home page"""
    results = get_analysis_results()
    
    # Check if data extraction has been done
    has_data = os.path.exists(os.path.join('balance_data', 'balance_updates.csv'))
    
    return render_template(
        'index.html', 
        results=results,
        has_data=has_data,
        task_status=task_status,
        theme=app.config['THEME'],
        image_size_factor=app.config['IMAGE_SIZE_FACTOR']
    )

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    """Fetch Discord data using the fetch_discord_data.py script"""
    global current_task, task_status
    
    if current_task:
        return jsonify({'status': 'error', 'message': 'Another task is currently running'})
    
    token = request.form.get('token', '')
    channel = request.form.get('channel', '')
    
    # Prepare arguments
    args = []
    if token:
        args.extend(['--token', token])
    if channel:
        args.extend(['--channel', channel])
    
    # Start task in thread
    current_task = threading.Thread(
        target=run_task,
        args=('./fetch_discord_data.py', args)
    )
    current_task.daemon = True
    current_task.start()
    
    return jsonify({'status': 'started', 'message': 'Data fetch started'})

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run the analysis pipeline"""
    global current_task, task_status
    
    if current_task:
        return jsonify({'status': 'error', 'message': 'Another task is currently running'})
    
    analysis_type = request.form.get('analysis_type', 'full')
    skip_extraction = request.form.get('skip_extraction', 'false') == 'true'
    
    # Prepare arguments
    args = []
    
    if skip_extraction:
        args.append('--skip-extraction')
    else:
        args.append('export.json')
    
    if analysis_type == 'economy':
        args.append('--economy-only')
    elif analysis_type == 'gambling':
        args.append('--gambling-only')
    elif analysis_type == 'category':
        args.append('--category-only')
    elif analysis_type == 'advanced':
        args.append('--advanced-only')
    
    # Start task in thread
    current_task = threading.Thread(
        target=run_task,
        args=('python run_analysis.py', args)
    )
    current_task.daemon = True
    current_task.start()
    
    return jsonify({'status': 'started', 'message': 'Analysis started'})

@app.route('/task_status')
def get_task_status():
    """Get the status of the current task"""
    return jsonify({
        'status': task_status,
        'output': task_output[-50:] if len(task_output) > 50 else task_output
    })

@app.route('/results/<path:filename>')
def get_result_file(filename):
    """Serve result files"""
    # Determine directory based on file path
    if filename.startswith('economy_analysis/'):
        return send_from_directory('.', filename)
    elif filename.startswith('gambling_analysis/'):
        return send_from_directory('.', filename)
    elif filename.startswith('category_analysis/'):
        return send_from_directory('.', filename)
    elif filename.startswith('advanced_analysis/'):
        return send_from_directory('.', filename)
    else:
        return "File not found", 404

@app.route('/view/<analysis_type>')
def view_analysis(analysis_type):
    """View a specific analysis report"""
    if analysis_type == 'economy':
        return send_from_directory('economy_analysis', 'economy_analysis_report.html')
    elif analysis_type == 'gambling':
        return send_from_directory('gambling_analysis', 'gambling_analysis_report.html')
    elif analysis_type == 'category':
        return send_from_directory('category_analysis', 'category_analysis_report.html')
    elif analysis_type == 'advanced':
        return send_from_directory('advanced_analysis', 'advanced_economy_report.html')
    else:
        return "Report not found", 404

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page"""
    if request.method == 'POST':
        app.config['THEME'] = request.form.get('theme', 'dark')
        app.config['IMAGE_SIZE_FACTOR'] = float(request.form.get('image_size_factor', '1.5'))
        # Could save to .env file here for persistence
        return redirect(url_for('index'))
    
    return render_template(
        'settings.html',
        theme=app.config['THEME'],
        image_size_factor=app.config['IMAGE_SIZE_FACTOR']
    )

def create_required_directories():
    """Create required directories if they don't exist"""
    ensure_directory('static')
    ensure_directory('templates')
    ensure_directory('balance_data')
    ensure_directory('economy_analysis')
    ensure_directory('gambling_analysis')
    ensure_directory('category_analysis')
    ensure_directory('advanced_analysis')

def main():
    """Run the Flask application"""
    create_required_directories()
    
    # Create default templates if they don't exist
    create_default_templates()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )

def create_default_templates():
    """Create default HTML templates if they don't exist"""
    # Create index.html if it doesn't exist
    index_template = os.path.join('templates', 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Discord Game Economy Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #121212;
            --text-color: #f0f0f0;
            --primary-color: #3f51b5;
            --secondary-color: #303f9f;
            --accent-color: #ff4081;
            --card-bg: #1e1e1e;
            --input-bg: #2a2a2a;
            --border-color: #333;
        }
        
        body.light {
            --bg-color: #f5f5f5;
            --text-color: #333;
            --primary-color: #3f51b5;
            --secondary-color: #303f9f;
            --accent-color: #ff4081;
            --card-bg: #fff;
            --input-bg: #f0f0f0;
            --border-color: #ddd;
        }
        
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        h1, h2, h3 {
            font-weight: 500;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        
        .btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .btn:hover {
            background-color: var(--secondary-color);
        }
        
        .btn-accent {
            background-color: var(--accent-color);
        }
        
        .btn-accent:hover {
            background-color: #e91e63;
        }
        
        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 8px 12px;
            color: var(--text-color);
            font-size: 14px;
            margin-bottom: 10px;
            width: 100%;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .results-container {
            margin-top: 30px;
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 10px;
            transition: transform 0.2s;
            cursor: zoom-in;
        }
        
        .result-image:hover {
            transform: scale(1.02);
        }
        
        .task-output {
            background-color: var(--input-bg);
            border-radius: 4px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            margin-top: 15px;
        }
        
        .status-running {
            color: orange;
        }
        
        .status-completed {
            color: green;
        }
        
        .status-failed {
            color: red;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }
        
        .tab.active {
            border-bottom: 3px solid var(--accent-color);
            font-weight: 500;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .image-container {
            width: calc(50% - 15px);
        }
        
        @media (max-width: 768px) {
            .image-container {
                width: 100%;
            }
        }
        
        .fullscreen-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }
        
        .fullscreen-image {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        }
        
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav-links {
            display: flex;
            gap: 15px;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
        }
    </style>
</head>
<body class="{{ theme }}">
    <div id="fullscreen-overlay" class="fullscreen-overlay">
        <img src="" id="fullscreen-image" class="fullscreen-image">
    </div>

    <header>
        <div class="navbar">
            <h1>Discord Game Economy Analysis</h1>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/settings">Settings</a>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="card">
            <h2>Fetch Discord Data</h2>
            <form id="fetch-form">
                <div class="form-group">
                    <label for="token">Bot Token (leave empty to use .env setting)</label>
                    <input type="password" id="token" name="token" placeholder="Discord Bot Token">
                </div>
                <div class="form-group">
                    <label for="channel">Channel ID</label>
                    <input type="text" id="channel" name="channel" value="1344274217969123381" placeholder="Discord Channel ID">
                </div>
                <button type="submit" class="btn btn-accent">Fetch Discord Data</button>
            </form>
        </div>

        <div class="card">
            <h2>Run Analysis</h2>
            <form id="analysis-form">
                <div class="form-group">
                    <label for="analysis-type">Analysis Type</label>
                    <select id="analysis-type" name="analysis_type">
                        <option value="full">Full Analysis</option>
                        <option value="economy">Economy Analysis Only</option>
                        <option value="gambling">Gambling Analysis Only</option>
                        <option value="category">Category Analysis Only</option>
                        <option value="advanced">Advanced Analysis Only</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="skip-extraction" name="skip_extraction" value="true" {% if not has_data %}disabled{% endif %}>
                        Skip Data Extraction (use existing data)
                    </label>
                </div>
                <button type="submit" class="btn">Run Analysis</button>
            </form>
        </div>

        <div class="card">
            <h2>Task Status: <span id="task-status-label" class="status-{{ task_status }}">{{ task_status }}</span></h2>
            <div id="task-output" class="task-output"></div>
        </div>

        <div class="results-container">
            <h2>Analysis Results</h2>
            
            <div class="tabs">
                <div class="tab active" data-tab="economy">Economy Analysis</div>
                <div class="tab" data-tab="gambling">Gambling Analysis</div>
                <div class="tab" data-tab="category">Category Analysis</div>
                <div class="tab" data-tab="advanced">Advanced Analysis</div>
            </div>
            
            <div id="economy-tab" class="tab-content active">
                <div class="card">
                    <h3>Economy Analysis Results</h3>
                    {% if results.economy %}
                        <a href="/view/economy" class="btn" target="_blank">Open Full Report</a>
                        <div class="image-gallery">
                            {% for file in results.economy %}
                                {% if file.endswith('.png') %}
                                <div class="image-container">
                                    <img src="/results/economy_analysis/{{ file }}" class="result-image" alt="{{ file }}" 
                                         style="width: calc(100% * {{ image_size_factor }});" data-src="/results/economy_analysis/{{ file }}">
                                </div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% else %}
                        <p>No economy analysis results available yet.</p>
                    {% endif %}
                </div>
            </div>
            
            <div id="gambling-tab" class="tab-content">
                <div class="card">
                    <h3>Gambling Analysis Results</h3>
                    {% if results.gambling %}
                        <a href="/view/gambling" class="btn" target="_blank">Open Full Report</a>
                        <div class="image-gallery">
                            {% for file in results.gambling %}
                                {% if file.endswith('.png') %}
                                <div class="image-container">
                                    <img src="/results/gambling_analysis/{{ file }}" class="result-image" alt="{{ file }}"
                                         style="width: calc(100% * {{ image_size_factor }});" data-src="/results/gambling_analysis/{{ file }}">
                                </div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% else %}
                        <p>No gambling analysis results available yet.</p>
                    {% endif %}
                </div>
            </div>
            
            <div id="category-tab" class="tab-content">
                <div class="card">
                    <h3>Category Analysis Results</h3>
                    {% if results.category %}
                        <a href="/view/category" class="btn" target="_blank">Open Full Report</a>
                        <div class="image-gallery">
                            {% for file in results.category %}
                                {% if file.endswith('.png') %}
                                <div class="image-container">
                                    <img src="/results/category_analysis/{{ file }}" class="result-image" alt="{{ file }}"
                                         style="width: calc(100% * {{ image_size_factor }});" data-src="/results/category_analysis/{{ file }}">
                                </div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% else %}
                        <p>No category analysis results available yet.</p>
                    {% endif %}
                </div>
            </div>
            
            <div id="advanced-tab" class="tab-content">
                <div class="card">
                    <h3>Advanced Analysis Results</h3>
                    {% if results.advanced %}
                        <a href="/view/advanced" class="btn" target="_blank">Open Full Report</a>
                        <div class="image-gallery">
                            {% for file in results.advanced %}
                                {% if file.endswith('.png') %}
                                <div class="image-container">
                                    <img src="/results/advanced_analysis/{{ file }}" class="result-image" alt="{{ file }}"
                                         style="width: calc(100% * {{ image_size_factor }});" data-src="/results/advanced_analysis/{{ file }}">
                                </div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    {% else %}
                        <p>No advanced analysis results available yet.</p>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Tab functionality
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Deactivate all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    
                    // Activate clicked tab
                    tab.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId + '-tab').classList.add('active');
                });
            });
            
            // Fullscreen image viewing
            const overlay = document.getElementById('fullscreen-overlay');
            const fullscreenImage = document.getElementById('fullscreen-image');
            
            document.querySelectorAll('.result-image').forEach(img => {
                img.addEventListener('click', () => {
                    fullscreenImage.src = img.getAttribute('data-src');
                    overlay.style.display = 'flex';
                });
            });
            
            overlay.addEventListener('click', () => {
                overlay.style.display = 'none';
            });
            
            // Form submission for fetching data
            document.getElementById('fetch-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                
                fetch('/fetch_data', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        updateTaskStatus();
                    } else {
                        alert(data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
            
            // Form submission for running analysis
            document.getElementById('analysis-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                
                fetch('/run_analysis', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        updateTaskStatus();
                    } else {
                        alert(data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
            
            // Function to update task status
            function updateTaskStatus() {
                const statusLabel = document.getElementById('task-status-label');
                const outputDiv = document.getElementById('task-output');
                
                fetch('/task_status')
                .then(response => response.json())
                .then(data => {
                    statusLabel.textContent = data.status;
                    statusLabel.className = 'status-' + data.status;
                    
                    outputDiv.innerHTML = '';
                    data.output.forEach(line => {
                        const lineDiv = document.createElement('div');
                        lineDiv.textContent = line;
                        outputDiv.appendChild(lineDiv);
                    });
                    
                    outputDiv.scrollTop = outputDiv.scrollHeight;
                    
                    if (data.status === 'running') {
                        setTimeout(updateTaskStatus, 1000);
                    } else if (data.status === 'completed') {
                        setTimeout(() => window.location.reload(), 3000);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
            
            // If a task is running, start polling
            if (statusLabel.textContent === 'running') {
                updateTaskStatus();
            }
        });
    </script>
</body>
</html>
""")
    
    # Create settings.html if it doesn't exist
    settings_template = os.path.join('templates', 'settings.html')
    if not os.path.exists(settings_template):
        with open(settings_template, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Settings - Discord Game Economy Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #121212;
            --text-color: #f0f0f0;
            --primary-color: #3f51b5;
            --secondary-color: #303f9f;
            --accent-color: #ff4081;
            --card-bg: #1e1e1e;
            --input-bg: #2a2a2a;
            --border-color: #333;
        }
        
        body.light {
            --bg-color: #f5f5f5;
            --text-color: #333;
            --primary-color: #3f51b5;
            --secondary-color: #303f9f;
            --accent-color: #ff4081;
            --card-bg: #fff;
            --input-bg: #f0f0f0;
            --border-color: #ddd;
        }
        
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        h1, h2, h3 {
            font-weight: 500;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        
        .btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .btn:hover {
            background-color: var(--secondary-color);
        }
        
        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 8px 12px;
            color: var(--text-color);
            font-size: 14px;
            margin-bottom: 10px;
            width: 100%;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav-links {
            display: flex;
            gap: 15px;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
        }
    </style>
</head>
<body class="{{ theme }}">
    <header>
        <div class="navbar">
            <h1>Settings</h1>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/settings">Settings</a>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="card">
            <h2>UI Settings</h2>
            <form method="POST">
                <div class="form-group">
                    <label for="theme">Theme</label>
                    <select id="theme" name="theme">
                        <option value="dark" {% if theme == 'dark' %}selected{% endif %}>Dark</option>
                        <option value="light" {% if theme == 'light' %}selected{% endif %}>Light</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="image_size_factor">Image Size Factor</label>
                    <input type="number" id="image_size_factor" name="image_size_factor" min="0.5" max="3" step="0.1" value="{{ image_size_factor }}">
                    <small>Adjust the size of images in the results (1.0 = 100% original size)</small>
                </div>
                <button type="submit" class="btn">Save Settings</button>
            </form>
        </div>

        <div class="card">
            <h2>About</h2>
            <p>Discord Game Economy Analysis Tool</p>
            <p>This tool helps analyze economy data from Discord games, providing insights for game developers.</p>
        </div>
    </div>
</body>
</html>
""")

if __name__ == "__main__":
    main() 