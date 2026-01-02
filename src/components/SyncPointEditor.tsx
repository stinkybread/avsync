import { SyncPoint } from '../App'
import './SyncPointEditor.css'

interface SyncPointEditorProps {
  syncPoints: SyncPoint[];
  onSyncPointsChange: (syncPoints: SyncPoint[]) => void;
}

export default function SyncPointEditor({ syncPoints, onSyncPointsChange }: SyncPointEditorProps) {
  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = (seconds % 60).toFixed(2);
    return h > 0
      ? `${h}:${m.toString().padStart(2, '0')}:${s.padStart(5, '0')}`
      : `${m}:${s.padStart(5, '0')}`;
  };

  const handleRemove = (index: number) => {
    const newSyncPoints = syncPoints.filter((_, i) => i !== index);
    onSyncPointsChange(newSyncPoints);
  };

  const handleClear = () => {
    if (confirm('Clear all sync points?')) {
      onSyncPointsChange([]);
    }
  };

  const handleEdit = (index: number, field: 'refTime' | 'foreignTime', value: string) => {
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return;

    const newSyncPoints = [...syncPoints];
    newSyncPoints[index] = {
      ...newSyncPoints[index],
      [field]: numValue,
    };
    onSyncPointsChange(newSyncPoints);
  };

  return (
    <div className="sync-point-editor">
      <div className="sync-point-header">
        <h2 className="section-title">Manual Sync Points</h2>
        {syncPoints.length > 0 && (
          <button className="btn-clear" onClick={handleClear}>
            Clear All
          </button>
        )}
      </div>

      {syncPoints.length === 0 ? (
        <div className="sync-point-empty">
          No sync points defined. Add sync points from the video preview above to manually
          specify matching timestamps.
        </div>
      ) : (
        <div className="sync-point-list">
          <div className="sync-point-list-header">
            <span className="sync-col-index">#</span>
            <span className="sync-col-ref">Reference Time</span>
            <span className="sync-col-foreign">Foreign Time</span>
            <span className="sync-col-actions"></span>
          </div>

          {syncPoints.map((point, index) => (
            <div key={index} className="sync-point-row">
              <span className="sync-col-index">{index + 1}</span>
              <div className="sync-col-ref">
                <input
                  type="number"
                  step="0.01"
                  value={point.refTime.toFixed(2)}
                  onChange={(e) => handleEdit(index, 'refTime', e.target.value)}
                  className="sync-time-input"
                />
                <span className="sync-time-display">{formatTime(point.refTime)}</span>
              </div>
              <div className="sync-col-foreign">
                <input
                  type="number"
                  step="0.01"
                  value={point.foreignTime.toFixed(2)}
                  onChange={(e) => handleEdit(index, 'foreignTime', e.target.value)}
                  className="sync-time-input"
                />
                <span className="sync-time-display">{formatTime(point.foreignTime)}</span>
              </div>
              <div className="sync-col-actions">
                <button
                  className="btn-remove"
                  onClick={() => handleRemove(index)}
                  title="Remove sync point"
                >
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="sync-point-info">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="var(--accent-blue)">
          <circle cx="8" cy="8" r="7" stroke="var(--accent-blue)" strokeWidth="1.5" fill="none" />
          <text x="8" y="12" fontSize="12" textAnchor="middle" fill="var(--accent-blue)">i</text>
        </svg>
        <span>
          Sync points manually specify matching timestamps. Format: refTime:foreignTime (e.g., 10.5:12.3).
          Leave empty for automatic detection.
        </span>
      </div>
    </div>
  )
}
