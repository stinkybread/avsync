import { useEffect, useRef } from 'react'
import './LogViewer.css'

interface LogViewerProps {
  logs: { type: string; data: string }[];
  isProcessing: boolean;
}

export default function LogViewer({ logs, isProcessing }: LogViewerProps) {
  const logEndRef = useRef<HTMLDivElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const getLogClassName = (type: string) => {
    if (type === 'stderr') return 'log-line log-stderr';
    if (type === 'error') return 'log-line log-error';
    if (type === 'success') return 'log-line log-success';
    return 'log-line';
  };

  const handleExportLogs = () => {
    const logText = logs.map((log) => `[${log.type}] ${log.data}`).join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `avsync_log_${new Date().toISOString().replace(/:/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="log-viewer">
      <div className="log-header">
        <h2 className="section-title">Processing Log</h2>
        <div className="log-actions">
          {isProcessing && (
            <div className="log-status">
              <div className="spinner" />
              <span>Processing...</span>
            </div>
          )}
          {logs.length > 0 && (
            <>
              <button className="btn-log-action" onClick={handleExportLogs} title="Export logs">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 2v8m0 0l-3-3m3 3l3-3M3 14h10" stroke="currentColor" strokeWidth="1.5" fill="none" />
                </svg>
              </button>
            </>
          )}
        </div>
      </div>

      <div className="log-content" ref={logContainerRef}>
        {logs.length === 0 ? (
          <div className="log-empty">
            {isProcessing
              ? 'Waiting for output...'
              : 'No logs yet. Click "Run AVSync" to start processing.'}
          </div>
        ) : (
          <div className="log-lines">
            {logs.map((log, index) => (
              <div key={index} className={getLogClassName(log.type)}>
                <span className="log-timestamp">
                  {new Date().toLocaleTimeString('en-US', { hour12: false })}
                </span>
                <span className="log-message">{log.data}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}
      </div>
    </div>
  )
}
