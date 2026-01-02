import './JobQueue.css'

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'aborted';

export type Job = {
  id: string;
  refVideo: string;
  foreignVideo: string;
  outputVideo: string;
  status: JobStatus;
  progress?: number;
  logs: { type: string; data: string }[];
  error?: string;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  params: any;
};

interface JobQueueProps {
  jobs: Job[];
  onRemoveJob: (jobId: string) => void;
  onRetryJob: (jobId: string) => void;
  onAbortJob: (jobId: string) => void;
  onViewLogs: (jobId: string) => void;
  onClearAll: () => void;
  onStartQueue: () => void;
  currentJobId?: string;
}

export default function JobQueue({
  jobs,
  onRemoveJob,
  onRetryJob,
  onAbortJob,
  onViewLogs,
  onClearAll,
  onStartQueue,
  currentJobId
}: JobQueueProps) {
  const pending = jobs.filter(j => j.status === 'pending').length;
  const processing = jobs.filter(j => j.status === 'processing').length;
  const completed = jobs.filter(j => j.status === 'completed').length;
  const failed = jobs.filter(j => j.status === 'failed').length;
  const aborted = jobs.filter(j => j.status === 'aborted').length;

  const canStartQueue = pending > 0 && processing === 0;

  const getFileName = (path: string) => {
    return path.split(/[/\\]/).pop() || path;
  };

  const formatDuration = (start?: Date, end?: Date) => {
    if (!start) return '-';
    if (!end) {
      const seconds = Math.floor((Date.now() - start.getTime()) / 1000);
      return `${seconds}s`;
    }
    const seconds = Math.floor((end.getTime() - start.getTime()) / 1000);
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  return (
    <div className="job-queue">
      <div className="queue-header">
        <div className="queue-header-left">
          <h2 className="queue-title">Job Queue</h2>
          <div className="queue-stats">
            <span className="stat stat-processing">{processing}</span>
            <span className="stat stat-pending">{pending}</span>
            <span className="stat stat-completed">{completed}</span>
            <span className="stat stat-failed">{failed}</span>
            {aborted > 0 && <span className="stat stat-aborted">{aborted}</span>}
          </div>
        </div>
        <div className="queue-header-actions">
          {canStartQueue && (
            <button className="btn-queue-action btn-start-queue" onClick={onStartQueue} title="Start Processing Queue">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            </button>
          )}
          {jobs.length > 0 && (
            <button className="btn-queue-action btn-clear-all" onClick={onClearAll} title="Clear All Jobs">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                <line x1="10" y1="11" x2="10" y2="17"/>
                <line x1="14" y1="11" x2="14" y2="17"/>
              </svg>
            </button>
          )}
        </div>
      </div>

      <div className="queue-list">
        {jobs.length === 0 ? (
          <div className="queue-empty">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
              <line x1="9" y1="9" x2="15" y2="9" strokeWidth="2"/>
              <line x1="9" y1="15" x2="15" y2="15" strokeWidth="2"/>
            </svg>
            <p>No jobs in queue</p>
            <span>Add a job to start processing</span>
          </div>
        ) : (
          jobs.map((job) => (
            <div
              key={job.id}
              className={`job-item ${job.status} ${job.id === currentJobId ? 'active' : ''}`}
            >
              <div className="job-status-indicator">
                {job.status === 'processing' && (
                  <div className="spinner-small" />
                )}
                {job.status === 'completed' && (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M20 6L9 17l-5-5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
                {job.status === 'failed' && (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                    <line x1="15" y1="9" x2="9" y2="15" strokeWidth="2" strokeLinecap="round"/>
                    <line x1="9" y1="9" x2="15" y2="15" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                )}
                {job.status === 'aborted' && (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <rect x="6" y="6" width="12" height="12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
                {job.status === 'pending' && (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                    <polyline points="12 6 12 12 16 14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
              </div>

              <div className="job-content">
                <div className="job-main">
                  <div className="job-file-name">{getFileName(job.foreignVideo)}</div>
                  <div className="job-meta">
                    {job.status === 'processing' && job.progress !== undefined && (
                      <span className="job-progress">{job.progress}%</span>
                    )}
                    {(job.status === 'completed' || job.status === 'failed') && (
                      <span className="job-duration">{formatDuration(job.startedAt, job.completedAt)}</span>
                    )}
                    {job.status === 'processing' && (
                      <span className="job-duration">{formatDuration(job.startedAt)}</span>
                    )}
                  </div>
                </div>

                {job.status === 'processing' && job.progress !== undefined && (
                  <div className="job-progress-bar">
                    <div className="job-progress-fill" style={{ width: `${job.progress}%` }} />
                  </div>
                )}

                {job.status === 'failed' && job.error && (
                  <div className="job-error">{job.error}</div>
                )}
              </div>

              <div className="job-actions">
                {job.status === 'processing' && (
                  <button
                    className="job-action-btn abort"
                    onClick={() => onAbortJob(job.id)}
                    title="Abort Job"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <rect x="6" y="6" width="12" height="12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                )}
                {(job.status === 'failed' || job.status === 'aborted') && (
                  <button
                    className="job-action-btn retry"
                    onClick={() => onRetryJob(job.id)}
                    title="Retry"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M1 4v6h6M23 20v-6h-6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                )}
                {(job.status === 'completed' || job.status === 'failed' || job.status === 'processing' || job.status === 'aborted') && (
                  <button
                    className="job-action-btn logs"
                    onClick={() => onViewLogs(job.id)}
                    title="View Logs"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <polyline points="14 2 14 8 20 8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <line x1="12" y1="18" x2="12" y2="12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <line x1="9" y1="15" x2="15" y2="15" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                )}
                {(job.status === 'pending' || job.status === 'completed' || job.status === 'failed' || job.status === 'aborted') && (
                  <button
                    className="job-action-btn remove"
                    onClick={() => onRemoveJob(job.id)}
                    title="Remove"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <polyline points="3 6 5 6 21 6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
