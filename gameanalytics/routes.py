#!/usr/bin/env python3
"""
Routes module for the Discord Game Analysis application.
Handles all HTTP routes for the Flask web interface.
"""

import os
import json
import requests
import sys
import uuid
import logging
from datetime import datetime
from flask import render_template, request, redirect, url_for, jsonify, send_from_directory, flash
from flask_cors import CORS  # Add CORS support

from gameanalytics.utils import logger, ensure_directory, is_docker_running
from gameanalytics.task_manager import (
    get_task_status, run_task, start_task, run_script, 
    TaskManager, register_task, TASK_STATUS  # Add TASK_STATUS import
)
from gameanalytics.database import (
    get_extraction_status, get_balance_updates_df, store_analysis_result, get_connection, init_tables
)
from gameanalytics.errors import handle_error, APIError

# Create task manager instance
task_manager = TaskManager()

@handle_error
def init_routes(app):
    """Initialize routes for the Flask application"""
    # Add CORS support
    CORS(app)
    
    # Define API error handler
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    # API routes
    @app.route('/api/status', methods=['GET'])
    def api_status():
        """API endpoint to check if the service is running"""
        return jsonify({
            "status": "ok",
            "version": "1.0.0",
            "message": "Game Analysis API is running"
        })
    
    @app.route('/')
    def index():
        """Home page"""
        # For backward compatibility - check if there are any extracted data
        has_data = False
        db_status = "Not connected"
        
        try:
            extraction_status = get_extraction_status()
            has_data = extraction_status is not None and extraction_status.get('total_balance_updates', 0) > 0
            db_status = "Connected" if has_data else "Empty"
        except Exception as e:
            logger.error(f"Error checking database: {e}")
        
        # Get docker status for info display
        docker_running = is_docker_running()
        
        # Get task status - modified to handle the new task_status parameter requirement
        # Use a default "latest" status for the index page
        current_status = {"status": "idle", "output": []}
        try:
            # If no specific task is requested, show the default status
            tasks = list(TASK_STATUS.values()) if hasattr(sys.modules['gameanalytics.task_manager'], 'TASK_STATUS') else []
            if tasks:
                # Get the most recent task
                current_status = sorted(tasks, key=lambda t: t.get('created_at', 0), reverse=True)[0]
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            
        # For compatibility - redirect to frontend or serve the template
        if os.path.exists(os.path.join(app.static_folder, 'index.html')):
            return send_from_directory(app.static_folder, 'index.html')
        else:
            # Fallback to the old template
            return render_template(
                'index.html', 
                has_data=has_data,
                task_status=current_status['status'],
                db_status=db_status,
                theme=app.config['THEME'],
                page="home",
                docker_running=docker_running
            )
    
    @app.route('/results')
    def results():
        """Results page"""
        results = get_analysis_results()
        
        # Check if data extraction has been done
        has_data = os.path.exists(os.path.join('balance_data', 'balance_updates.csv'))
        
        # Get available analysis dates based on the timestamp in filenames
        analysis_dates = []
        if has_data:
            # Check for timestamped directories or use current if none found
            timestamp = datetime.now().strftime('%Y-%m-%d')
            analysis_dates.append(timestamp)
        
        current_status = get_task_status()
        
        return render_template(
            'results.html', 
            results=results,
            has_data=has_data,
            analysis_dates=analysis_dates,
            task_status=current_status['status'],
            theme=app.config['THEME'],
            image_size_factor=app.config['IMAGE_SIZE_FACTOR'],
            page="results"
        )
    
    @app.route('/fetch_data', methods=['POST'])
    def fetch_data():
        """Fetch Discord data using the fetch_discord_data.py script"""
        current_status = get_task_status()
        if current_status['status'] == 'running':
            return jsonify({'status': 'error', 'message': 'Another task is currently running'})
        
        # Check if Docker is running
        if not is_docker_running():
            return jsonify({
                'status': 'error', 
                'message': 'Docker is not running. Please start Docker and try again.',
                'error_type': 'docker_not_running'
            })
        
        token = request.form.get('token', '')
        channel = request.form.get('channel', '')
        
        # Prepare arguments
        args = []
        if token:
            args.extend(['--token', token])
        if channel:
            args.extend(['--channel', channel])
        
        # Start task in thread
        success = run_script('./fetch_discord_data.py', args)
        
        if success:
            return jsonify({'status': 'started', 'message': 'Data fetch started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start data fetch task'})
    
    @app.route('/run_analysis', methods=['POST'])
    def run_analysis():
        """Run the analysis pipeline"""
        current_status = get_task_status()
        if current_status['status'] == 'running':
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
        success = run_script('./run_analysis.py', args)
        
        if success:
            return jsonify({'status': 'started', 'message': 'Analysis started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start analysis task'})
    
    @app.route('/generate_recommendations', methods=['POST'])
    def generate_recommendations():
        """Generate AI recommendations based on analysis results"""
        current_status = get_task_status()
        if current_status['status'] == 'running':
            return jsonify({'status': 'error', 'message': 'Another task is currently running'})
        
        def generate_ai_recommendations_task():
            try:
                result = generate_ai_recommendations(app.config)
                return result is not None
            except Exception as e:
                logger.error(f"Error generating AI recommendations: {e}")
                return False
        
        success = start_task(generate_ai_recommendations_task)
        
        if success:
            return jsonify({'status': 'started', 'message': 'Generating AI recommendations'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start recommendations task'})
    
    @app.route('/task_status')
    def task_status():
        """Get the status of the current task"""
        return jsonify(get_task_status())
    
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
        elif filename.startswith('ai_recommendations/'):
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
        elif analysis_type.startswith('recommendations_'):
            return send_from_directory('ai_recommendations', analysis_type)
        else:
            return "Report not found", 404
    
    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        """Settings page"""
        if request.method == 'POST':
            # Update app settings
            app.config['THEME'] = request.form.get('theme', 'dark')
            app.config['IMAGE_SIZE_FACTOR'] = float(request.form.get('image_size_factor', '1.5'))
            
            # Update Claude API settings
            app.config['CLAUDE_API_KEY'] = request.form.get('claude_api_key', '')
            app.config['CLAUDE_MODEL'] = request.form.get('claude_model', 'claude-3-opus-20240229')
            app.config['CLAUDE_PROMPT'] = request.form.get('claude_prompt', app.config['CLAUDE_PROMPT'])
            
            # Save to .env file
            update_env_file({
                'THEME': app.config['THEME'],
                'IMAGE_SIZE_FACTOR': str(app.config['IMAGE_SIZE_FACTOR']),
                'CLAUDE_API_KEY': app.config['CLAUDE_API_KEY'],
                'CLAUDE_MODEL': app.config['CLAUDE_MODEL'],
                'CLAUDE_PROMPT': app.config['CLAUDE_PROMPT'].replace('\n', '\\n')
            })
            
            flash('Settings updated successfully', 'success')
            return redirect(url_for('settings'))
        
        return render_template(
            'settings.html',
            theme=app.config['THEME'],
            image_size_factor=app.config['IMAGE_SIZE_FACTOR'],
            claude_api_key=app.config['CLAUDE_API_KEY'],
            claude_model=app.config['CLAUDE_MODEL'],
            claude_prompt=app.config['CLAUDE_PROMPT'],
            page="settings"
        )

    @app.route('/check_docker_status')
    def check_docker_status():
        """Check if Docker is running and return status as JSON"""
        docker_running = is_docker_running()
        return jsonify({
            'running': docker_running,
            'message': 'Docker is running' if docker_running else 'Docker is not running'
        })

    @app.route('/reset_task_status', methods=['POST'])
    def reset_task():
        """Reset a stuck task status"""
        from gameanalytics.task_manager import reset_task_status
        reset_task_status()
        return jsonify({
            'status': 'success',
            'message': 'Task status has been reset'
        })

    @app.route('/api/run_analysis', methods=['POST'])
    def api_run_analysis():
        """API endpoint to run analysis"""
        try:
            data = request.json
            # Validate required parameters
            if not data.get('file_path'):
                raise APIError("Missing file_path parameter", status_code=400)
                
            task_id = str(uuid.uuid4())
            analysis_type = data.get('analysis_type', 'all')
            
            # Register the task
            register_task(task_id, "Running analysis: " + analysis_type)
            
            # Start the task in the background
            task_manager.run_task(
                task_id=task_id,
                task_type='analysis',
                file_path=data.get('file_path'),
                options=data
            )
            
            return jsonify({
                "status": "success",
                "task_id": task_id,
                "message": "Analysis started"
            })
        except Exception as e:
            logger.error(f"Error starting analysis: {e}")
            raise APIError(str(e), status_code=500)
    
    @app.route('/api/task_status/<task_id>', methods=['GET'])
    def api_task_status(task_id):
        """API endpoint to get task status"""
        status = get_task_status(task_id)
        
        if not status:
            return jsonify({
                "status": "error",
                "message": "Task not found"
            }), 404
            
        return jsonify(status)

# Helper functions 

def update_env_file(new_values):
    """Update values in .env file
    
    Args:
        new_values: Dictionary of key-value pairs to update in .env file
    """
    # Read existing .env file
    env_vars = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    # Update with new values
    env_vars.update(new_values)
    
    # Write back to .env file
    with open('.env', 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

def get_analysis_results():
    """Get list of available analysis results
    
    Returns:
        dict: Dictionary of analysis results by category
    """
    results = {
        'economy': [],
        'gambling': [],
        'category': [],
        'advanced': [],
        'ai_recommendations': []
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
                
    # AI Recommendations
    if os.path.exists('ai_recommendations'):
        for file in os.listdir('ai_recommendations'):
            if file.endswith('.html') or file.endswith('.txt'):
                results['ai_recommendations'].append(file)
    
    return results

def extract_analysis_data():
    """Extract key data from analysis results for AI recommendations
    
    Returns:
        dict: Analysis data for AI recommendations
    """
    analysis_data = {
        "economy": {},
        "gambling": {},
        "category": {},
        "advanced": {}
    }
    
    # Extract data from HTML reports
    report_paths = {
        "economy": os.path.join("economy_analysis", "economy_analysis_report.html"),
        "gambling": os.path.join("gambling_analysis", "gambling_analysis_report.html"),
        "category": os.path.join("category_analysis", "category_analysis_report.html"),
        "advanced": os.path.join("advanced_analysis", "advanced_economy_report.html")
    }
    
    for report_type, path in report_paths.items():
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract data between <div class="insights"> and </div>
                    insight_start = content.find('<div class="insight">')
                    if insight_start > 0:
                        insight_end = content.find('</div>', insight_start)
                        if insight_end > 0:
                            insights = content[insight_start:insight_end + 6]
                            # Strip HTML tags
                            import re
                            insights = re.sub('<[^<]+?>', '', insights)
                            analysis_data[report_type]["insights"] = insights.strip()
            except Exception as e:
                logger.error(f"Error extracting insights from {path}: {e}")
    
    # Extract key metrics
    try:
        # Economy metrics
        if os.path.exists(os.path.join("economy_analysis", "economy_metrics.json")):
            with open(os.path.join("economy_analysis", "economy_metrics.json"), 'r') as f:
                analysis_data["economy"]["metrics"] = json.load(f)
        
        # Gambling metrics
        if os.path.exists(os.path.join("gambling_analysis", "gambling_metrics.json")):
            with open(os.path.join("gambling_analysis", "gambling_metrics.json"), 'r') as f:
                analysis_data["gambling"]["metrics"] = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
    
    return analysis_data

def generate_ai_recommendations(config):
    """Generate AI recommendations using Claude API
    
    Args:
        config: Flask app configuration
        
    Returns:
        str: Generated recommendations or None if failed
    """
    # Check if API key is configured
    if not config['CLAUDE_API_KEY']:
        logger.error("Claude API key not configured. Please add CLAUDE_API_KEY to .env file.")
        return None
    
    # Extract analysis data
    analysis_data = extract_analysis_data()
    
    # Format data for Claude
    formatted_data = json.dumps(analysis_data, indent=2)
    
    # Fill in the prompt template
    prompt = config['CLAUDE_PROMPT'].format(analysis_data=formatted_data)
    
    try:
        # Call Claude API
        headers = {
            "x-api-key": config['CLAUDE_API_KEY'],
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config['CLAUDE_MODEL'],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            recommendations = result["content"][0]["text"]
            
            # Save recommendations to file
            ensure_directory("ai_recommendations")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Recommendations</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    h3 {{ color: #7f8c8d; margin-top: 20px; }}
                    ul {{ margin-bottom: 20px; }}
                    .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-top: 40px; }}
                    .recommendations {{ white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>AI-Generated Recommendations</h1>
                <div class="recommendations">
                {recommendations}
                </div>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            with open(os.path.join("ai_recommendations", f"recommendations_{timestamp}.html"), "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Also save raw text
            with open(os.path.join("ai_recommendations", f"recommendations_{timestamp}.txt"), "w", encoding="utf-8") as f:
                f.write(recommendations)
            
            # Store the result in the database if available
            try:
                store_analysis_result(
                    "ai_recommendations",
                    {"model": config['CLAUDE_MODEL']},
                    {"recommendations": recommendations},
                    os.path.join("ai_recommendations", f"recommendations_{timestamp}.html")
                )
            except Exception as e:
                logger.warning(f"Could not store recommendations in database: {e}")
            
            return recommendations
        else:
            error_msg = f"Error calling Claude API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return None
    
    except Exception as e:
        error_msg = f"Error generating AI recommendations: {str(e)}"
        logger.error(error_msg)
        return None 