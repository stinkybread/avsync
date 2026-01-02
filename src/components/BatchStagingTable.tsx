import { BatchJobItem } from './BatchJob'
import './BatchStagingTable.css'

interface BatchStagingTableProps {
  jobs: BatchJobItem[]
  editingId: string | null
  onEditJob: (id: string, field: keyof BatchJobItem, value: any) => void
  onRemoveJob: (id: string) => void
  onSetEditingId: (id: string | null) => void
  onAddToQueue: () => void
}

export default function BatchStagingTable({
  jobs,
  editingId,
  onEditJob,
  onRemoveJob,
  onSetEditingId,
  onAddToQueue,
}: BatchStagingTableProps) {
  const getFileName = (path: string) => {
    return path.split(/[/\\]/).pop() || path
  }

  if (jobs.length === 0) {
    return (
      <div className="batch-staging-table">
        <div className="staging-table-header">
          <h2 className="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="8" y1="6" x2="21" y2="6"/>
              <line x1="8" y1="12" x2="21" y2="12"/>
              <line x1="8" y1="18" x2="21" y2="18"/>
              <line x1="3" y1="6" x2="3.01" y2="6"/>
              <line x1="3" y1="12" x2="3.01" y2="12"/>
              <line x1="3" y1="18" x2="3.01" y2="18"/>
            </svg>
            Batch Job Staging
          </h2>
        </div>
        <div className="staging-table-empty">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
          <p>No batch jobs loaded yet</p>
          <span>Select folders in the Batch tab and click "Load Matched Files" to begin</span>
        </div>
      </div>
    )
  }

  return (
    <div className="batch-staging-table">
      <div className="staging-table-header">
        <h2 className="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="8" y1="6" x2="21" y2="6"/>
            <line x1="8" y1="12" x2="21" y2="12"/>
            <line x1="8" y1="18" x2="21" y2="18"/>
            <line x1="3" y1="6" x2="3.01" y2="6"/>
            <line x1="3" y1="12" x2="3.01" y2="12"/>
            <line x1="3" y1="18" x2="3.01" y2="18"/>
          </svg>
          Staged Batch Jobs ({jobs.length})
        </h2>
        <button className="btn-add-to-queue-large" onClick={onAddToQueue}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2"/>
          </svg>
          Add {jobs.length} Job{jobs.length !== 1 ? 's' : ''} to Queue
        </button>
      </div>

      <div className="staging-table-container">
        <table className="staging-table">
          <thead>
            <tr>
              <th>Reference</th>
              <th>Foreign</th>
              <th>Output</th>
              <th>First Seg</th>
              <th>Last Seg</th>
              <th>Skip Subs</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(job => (
              <tr key={job.id} className={editingId === job.id ? 'editing' : ''}>
                <td className="file-cell" title={job.refVideo}>
                  {getFileName(job.refVideo)}
                </td>
                <td className="file-cell" title={job.foreignVideo}>
                  {getFileName(job.foreignVideo)}
                </td>
                <td className="file-cell editable" title={job.outputVideo}>
                  {editingId === job.id ? (
                    <input
                      type="text"
                      value={getFileName(job.outputVideo)}
                      onChange={(e) => {
                        const dir = job.outputVideo.substring(0, job.outputVideo.lastIndexOf('/'))
                        onEditJob(job.id, 'outputVideo', `${dir}/${e.target.value}`)
                      }}
                      className="cell-input"
                    />
                  ) : (
                    getFileName(job.outputVideo)
                  )}
                </td>
                <td className="num-cell">
                  {editingId === job.id ? (
                    <input
                      type="number"
                      value={job.firstSegmentAdjust}
                      onChange={(e) => onEditJob(job.id, 'firstSegmentAdjust', parseFloat(e.target.value))}
                      className="cell-input-num"
                    />
                  ) : (
                    job.firstSegmentAdjust
                  )}
                </td>
                <td className="num-cell">
                  {editingId === job.id ? (
                    <input
                      type="number"
                      value={job.lastSegmentAdjust}
                      onChange={(e) => onEditJob(job.id, 'lastSegmentAdjust', parseFloat(e.target.value))}
                      className="cell-input-num"
                    />
                  ) : (
                    job.lastSegmentAdjust
                  )}
                </td>
                <td className="checkbox-cell">
                  <input
                    type="checkbox"
                    checked={job.skipSubtitles}
                    onChange={(e) => onEditJob(job.id, 'skipSubtitles', e.target.checked)}
                    disabled={editingId !== job.id && editingId !== null}
                  />
                </td>
                <td className="actions-cell">
                  {editingId === job.id ? (
                    <button className="btn-action btn-done" onClick={() => onSetEditingId(null)}>
                      Done
                    </button>
                  ) : (
                    <>
                      <button className="btn-action btn-edit" onClick={() => onSetEditingId(job.id)}>
                        Edit
                      </button>
                      <button className="btn-action btn-remove" onClick={() => onRemoveJob(job.id)}>
                        Remove
                      </button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
