#!/usr/bin/env python3
"""
Task Manager module for the Discord Game Analysis application.
Handles background tasks and task status tracking.
"""

import subprocess
import threading
import time
from datetime import datetime
import os
import sys
import traceback
import uuid
import queue
import logging
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor

from gameanalytics.utils import logger

# Global variables for tracking task status
current_task = None
task_output = []
task_status = "idle"  # idle, running, completed, failed
task_lock = threading.Lock()
task_start_time = None
# Task timeout in seconds (5 minutes)
TASK_TIMEOUT = 300

# Task status storage
TASK_STATUS = {}
TASK_QUEUE = queue.Queue()
MAX_WORKERS = 4

def register_task(task_id: str, description: str) -> None:
    """Register a new task
    
    Args:
        task_id: Unique identifier for the task
        description: Description of the task
    """
    TASK_STATUS[task_id] = {
        'task_id': task_id,
        'status': 'pending',
        'description': description,
        'progress': 0,
        'message': 'Task registered',
        'output': [],
        'created_at': time.time(),
        'started_at': None,
        'completed_at': None,
    }

def update_task_status(task_id: str, 
                       status: Optional[str] = None, 
                       progress: Optional[int] = None,
                       message: Optional[str] = None,
                       output: Optional[str] = None) -> None:
    """Update task status
    
    Args:
        task_id: Task ID to update
        status: New status value (running, completed, failed)
        progress: Progress percentage (0-100)
        message: Status message
        output: Output text to append
    """
    if task_id not in TASK_STATUS:
        logger.warning(f"Attempted to update unknown task: {task_id}")
        return
        
    if status:
        TASK_STATUS[task_id]['status'] = status
        
        # Update timestamps
        if status == 'running' and not TASK_STATUS[task_id].get('started_at'):
            TASK_STATUS[task_id]['started_at'] = time.time()
        elif status in ['completed', 'failed']:
            TASK_STATUS[task_id]['completed_at'] = time.time()
            
    if progress is not None:
        TASK_STATUS[task_id]['progress'] = progress
        
    if message:
        TASK_STATUS[task_id]['message'] = message
        
    if output:
        TASK_STATUS[task_id]['output'].append(output)
        logger.info(f"Task output: {output}")
        
    # Calculate estimated time remaining if we have progress
    if progress and progress > 0 and status == 'running' and TASK_STATUS[task_id].get('started_at'):
        elapsed = time.time() - TASK_STATUS[task_id]['started_at']
        estimated_total = elapsed * 100 / progress
        remaining = estimated_total - elapsed
        
        if remaining > 0:
            TASK_STATUS[task_id]['eta'] = max(int(remaining), 0)

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a task
    
    Args:
        task_id: Task ID to get status for
        
    Returns:
        Dictionary with task status information
    """
    return TASK_STATUS.get(task_id, {})

class TaskManager:
    """Task Manager for handling background tasks"""
    
    def __init__(self):
        """Initialize the task manager"""
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info(f"Task manager initialized with {MAX_WORKERS} workers")
        
    def _worker_loop(self):
        """Worker loop to process tasks from the queue"""
        while True:
            try:
                task = TASK_QUEUE.get()
                if task:
                    self.executor.submit(self._execute_task, **task)
                TASK_QUEUE.task_done()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
            time.sleep(0.1)
    
    def _execute_task(self, **kwargs):
        """Execute a task based on its type"""
        task_id = kwargs.get('task_id')
        task_type = kwargs.get('task_type')
        
        if not task_id or not task_type:
            logger.error("Missing task_id or task_type")
            return
            
        update_task_status(task_id, status='running', message=f"Starting {task_type} task")
        
        try:
            if task_type == 'analysis':
                self._run_analysis_task(**kwargs)
            elif task_type == 'extraction':
                self._run_extraction_task(**kwargs)
            elif task_type == 'script':
                self._run_script_task(**kwargs)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            update_task_status(task_id, status='completed', progress=100, 
                              message=f"{task_type.capitalize()} task completed")
                              
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            update_task_status(task_id, status='failed', 
                              message=f"Task failed: {str(e)}")
    
    def _run_analysis_task(self, **kwargs):
        """Run an analysis task"""
        task_id = kwargs.get('task_id')
        file_path = kwargs.get('file_path')
        options = kwargs.get('options', {})
        
        # Import here to avoid circular imports
        from gameanalytics.analyzers.runner import run_full_analysis
        
        # Run the analysis with progress callback
        def progress_callback(message, progress):
            update_task_status(task_id, progress=progress, output=message)
            
        run_full_analysis(
            file_path=file_path,
            skip_extraction=options.get('skip_extraction', False),
            economy_only=options.get('economy_only', False),
            gambling_only=options.get('gambling_only', False),
            category_only=options.get('category_only', False),
            progress_callback=progress_callback
        )
    
    def _run_extraction_task(self, **kwargs):
        """Run a data extraction task"""
        task_id = kwargs.get('task_id')
        file_path = kwargs.get('file_path')
        
        # Import here to avoid circular imports
        from gameanalytics.extractors.discord_extractor import extract_data
        
        def progress_callback(message, progress):
            update_task_status(task_id, progress=progress, output=message)
            
        extract_data(file_path, progress_callback=progress_callback)
    
    def _run_script_task(self, **kwargs):
        """Run a script as a subprocess"""
        task_id = kwargs.get('task_id')
        script_path = kwargs.get('script_path')
        args = kwargs.get('args', [])
        
        if not script_path or not os.path.exists(script_path):
            raise ValueError(f"Script not found: {script_path}")
            
        cmd = [sys.executable, script_path] + args
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read output and update task status
        for line in iter(process.stdout.readline, ''):
            if line:
                update_task_status(task_id, output=line.strip())
                
                # Try to parse progress information
                if 'progress:' in line.lower():
                    try:
                        progress_part = line.lower().split('progress:')[1].strip()
                        progress = int(progress_part.split('%')[0].strip())
                        update_task_status(task_id, progress=progress)
                    except (ValueError, IndexError):
                        pass
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Script exited with code {process.returncode}")
    
    def run_task(self, **kwargs):
        """Add a task to the queue
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task (analysis, extraction, script)
            **kwargs: Additional arguments for the task
        """
        TASK_QUEUE.put(kwargs)
        logger.info(f"Task added to queue: {kwargs.get('task_id')} - {kwargs.get('task_type')}")
        
# Legacy functions for backward compatibility
def run_task(*args, **kwargs):
    """Legacy function for backward compatibility"""
    task_manager = TaskManager()
    return task_manager.run_task(*args, **kwargs)

def start_task(*args, **kwargs):
    """Legacy function for backward compatibility"""
    task_id = str(uuid.uuid4())
    register_task(task_id, kwargs.get('description', 'Task'))
    kwargs['task_id'] = task_id
    run_task(**kwargs)
    return task_id
    
def run_script(*args, **kwargs):
    """Legacy function for backward compatibility"""
    task_id = str(uuid.uuid4())
    register_task(task_id, f"Running script: {kwargs.get('script_path')}")
    
    task_manager = TaskManager()
    task_manager.run_task(
        task_id=task_id,
        task_type='script',
        **kwargs
    )
    
    return task_id

def reset_task_status():
    """Reset the task status to idle
    
    Returns:
        None
    """
    global task_status, task_output, task_start_time
    with task_lock:
        task_status = "idle"
        task_output = []
        task_start_time = None
        logger.info("Task status has been manually reset") 