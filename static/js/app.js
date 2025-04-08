// Main App Component
const App = () => {
  const [theme, setTheme] = React.useState('dark');
  
  return (
    <div className={theme === 'light' ? 'light-theme' : ''}>
      <AppHeader theme={theme} setTheme={setTheme} />
      <main className="app-container">
        <TaskStatus taskId="dd3aa114-6de0-4de2-a78c-6c7635665d5e" />
      </main>
    </div>
  );
};

// Header Component
const AppHeader = ({ theme, setTheme }) => {
  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };
  
  return (
    <header className="app-header">
      <div className="header-content">
        <h1>Discord Game Economy Analysis</h1>
        <div className="header-actions">
          <button onClick={toggleTheme} className="theme-toggle">
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </div>
    </header>
  );
};

// Task Status Component
const TaskStatus = ({ taskId }) => {
  const [status, setStatus] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  
  const fetchStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/task_status/${taskId}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch task status on component mount
  React.useEffect(() => {
    fetchStatus();
    
    // Refresh status every 2 seconds if task is running
    const interval = setInterval(() => {
      if (status?.status === 'running' || !status) {
        fetchStatus();
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [taskId, status?.status]);
  
  if (loading && !status) {
    return (
      <div className="task-status-card">
        <h2>Task Status</h2>
        <div className="loading-spinner"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="task-status-card error">
        <h2>Task Status</h2>
        <p className="error-message">Error: {error}</p>
        <button onClick={fetchStatus} className="retry-button">Retry</button>
      </div>
    );
  }
  
  if (!status) {
    return (
      <div className="task-status-card">
        <h2>Task Status</h2>
        <p>Task not found</p>
      </div>
    );
  }
  
  const getStatusColor = (statusType) => {
    switch (statusType) {
      case 'running': return 'var(--warning-color)';
      case 'completed': return 'var(--success-color)';
      case 'failed': return 'var(--error-color)';
      default: return 'var(--info-color)';
    }
  };
  
  return (
    <div className="task-status-card">
      <h2>Task Status: {taskId}</h2>
      <div className="status-indicator" style={{ backgroundColor: getStatusColor(status.status) }}></div>
      <div className="status-details">
        <p><strong>Status:</strong> {status.status}</p>
        <p><strong>Description:</strong> {status.description || 'N/A'}</p>
        <p><strong>Progress:</strong> {status.progress || 0}%</p>
        {status.message && <p><strong>Message:</strong> {status.message}</p>}
        {status.eta && <p><strong>Estimated time remaining:</strong> {status.eta} seconds</p>}
      </div>
      
      {status.output && status.output.length > 0 && (
        <div className="output-section">
          <h3>Output:</h3>
          <div className="task-output">
            {status.output.map((line, index) => (
              <div key={index} className="output-line">{line}</div>
            ))}
          </div>
        </div>
      )}
      
      <button onClick={fetchStatus} className="refresh-button">Refresh</button>
    </div>
  );
};

// Add CSS styles
const styleElement = document.createElement('style');
styleElement.textContent = `
  .app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }
  
  .app-header {
    background-color: var(--primary-color);
    color: white;
    padding: 15px 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  }
  
  .header-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .header-actions {
    display: flex;
    gap: 15px;
  }
  
  .theme-toggle {
    background-color: transparent;
    border: 2px solid white;
    color: white;
    padding: 8px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s;
  }
  
  .theme-toggle:hover {
    background-color: rgba(255,255,255,0.1);
  }
  
  .task-status-card {
    background-color: var(--card-bg);
    border-radius: 8px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: 1px solid var(--border-color);
  }
  
  .task-status-card h2 {
    margin-top: 0;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 10px;
    font-weight: 500;
  }
  
  .status-indicator {
    display: inline-block;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
  }
  
  .status-details {
    margin: 20px 0;
  }
  
  .output-section {
    margin-top: 20px;
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
  }
  
  .output-line {
    padding: 2px 0;
  }
  
  .refresh-button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 15px;
    font-size: 14px;
    cursor: pointer;
    margin-top: 15px;
    transition: background-color 0.2s;
  }
  
  .refresh-button:hover {
    background-color: var(--primary-dark);
  }
  
  .error-message {
    color: var(--error-color);
    background-color: rgba(207, 102, 121, 0.1);
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
  }
  
  .retry-button {
    background-color: var(--error-color);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 15px;
    font-size: 14px;
    cursor: pointer;
  }
  
  .retry-button:hover {
    background-color: #b00020;
  }
`;
document.head.appendChild(styleElement);

// Render the App
ReactDOM.render(<App />, document.getElementById('root')); 