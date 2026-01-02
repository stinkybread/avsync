import { useState } from 'react'
import { SyncPoint } from '../App'
import './SyncPointPills.css'

interface SyncPointPillsProps {
  syncPoints: SyncPoint[];
  onSyncPointsChange: (syncPoints: SyncPoint[]) => void;
}

export default function SyncPointPills({ syncPoints, onSyncPointsChange }: SyncPointPillsProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editValues, setEditValues] = useState({ refTime: '', foreignTime: '' });

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);

    const hh = h.toString().padStart(2, '0');
    const mm = m.toString().padStart(2, '0');
    const ss = s.toString().padStart(2, '0');
    const mss = ms.toString().padStart(3, '0');

    return `${hh}:${mm}:${ss}:${mss}`;
  };

  const handleRemove = (index: number) => {
    const newSyncPoints = syncPoints.filter((_, i) => i !== index);
    onSyncPointsChange(newSyncPoints);
    if (editingIndex === index) {
      setEditingIndex(null);
    }
  };

  const handleClear = () => {
    if (confirm('Clear all sync points?')) {
      onSyncPointsChange([]);
      setEditingIndex(null);
    }
  };

  const handleEdit = (index: number) => {
    setEditingIndex(index);
    setEditValues({
      refTime: syncPoints[index].refTime.toString(),
      foreignTime: syncPoints[index].foreignTime.toString(),
    });
  };

  const handleSaveEdit = () => {
    if (editingIndex === null) return;

    const refTime = parseFloat(editValues.refTime);
    const foreignTime = parseFloat(editValues.foreignTime);

    if (isNaN(refTime) || isNaN(foreignTime)) {
      alert('Invalid time values');
      return;
    }

    const newSyncPoints = [...syncPoints];
    newSyncPoints[editingIndex] = { refTime, foreignTime };
    onSyncPointsChange(newSyncPoints);
    setEditingIndex(null);
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
  };

  return (
    <div className="sync-point-pills">
      <div className="sync-pills-header">
        <div className="sync-pills-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="12" cy="12" r="3" strokeWidth="2"/>
            <path d="M12 1v6M12 17v6M4.22 4.22l4.24 4.24M15.54 15.54l4.24 4.24M1 12h6M17 12h6M4.22 19.78l4.24-4.24M15.54 8.46l4.24-4.24" strokeWidth="2"/>
          </svg>
          <span>Manual Sync Points</span>
          <span className="sync-count">{syncPoints.length}</span>
          <div className="info-tooltip" title="Navigate both videos to matching frames, then click 'Add Sync Point'">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--accent-blue)">
              <circle cx="12" cy="12" r="10" stroke="var(--accent-blue)" strokeWidth="1.5" fill="none"/>
              <path d="M12 16v-4M12 8h.01" stroke="var(--accent-blue)" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
        </div>
        <button
          className="btn-clear-all"
          onClick={handleClear}
          disabled={syncPoints.length === 0}
        >
          Clear All
        </button>
      </div>

      <div className="sync-pills-container">
        {syncPoints.length === 0 ? (
          <div className="sync-pills-empty">
            <p>No sync points defined</p>
            <span>Add matching frames from the video preview above</span>
          </div>
        ) : (
          <div className="sync-pills-list">
            {syncPoints.map((point, index) => (
              <div key={index} className={`sync-pill ${editingIndex === index ? 'editing' : ''}`}>
                {editingIndex === index ? (
                  <div className="sync-pill-edit">
                    <div className="edit-inputs">
                      <div className="edit-field">
                        <label>Ref</label>
                        <input
                          type="number"
                          step="0.001"
                          value={editValues.refTime}
                          onChange={(e) => setEditValues({ ...editValues, refTime: e.target.value })}
                          className="edit-input"
                          autoFocus
                        />
                      </div>
                      <span className="edit-separator">→</span>
                      <div className="edit-field">
                        <label>For</label>
                        <input
                          type="number"
                          step="0.001"
                          value={editValues.foreignTime}
                          onChange={(e) => setEditValues({ ...editValues, foreignTime: e.target.value })}
                          className="edit-input"
                        />
                      </div>
                    </div>
                    <div className="edit-actions">
                      <button className="edit-btn save" onClick={handleSaveEdit}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <polyline points="20 6 9 17 4 12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </button>
                      <button className="edit-btn cancel" onClick={handleCancelEdit}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
                          <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <span className="pill-number">{index + 1}</span>
                    <span className="pill-times">
                      {formatTime(point.refTime)} → {formatTime(point.foreignTime)}
                    </span>
                    <div className="pill-actions">
                      <button
                        className="pill-action-btn edit"
                        onClick={() => handleEdit(index)}
                        title="Edit"
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </button>
                      <button
                        className="pill-action-btn remove"
                        onClick={() => handleRemove(index)}
                        title="Remove"
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
                          <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
