import { useState, useEffect } from 'react'
import { VideoFile, SyncPoint } from '../App'
import './VideoPreview.css'

interface VideoPreviewProps {
  refVideo: VideoFile | null;
  foreignVideo: VideoFile | null;
  syncPoints: SyncPoint[];
  onAddSyncPoint: (point: SyncPoint) => void;
}

export default function VideoPreview({
  refVideo,
  foreignVideo,
  onAddSyncPoint,
}: VideoPreviewProps) {
  const [refTime, setRefTime] = useState(0);
  const [foreignTime, setForeignTime] = useState(0);
  const [refFrame, setRefFrame] = useState<string | null>(null);
  const [foreignFrame, setForeignFrame] = useState<string | null>(null);
  const [isLoadingRef, setIsLoadingRef] = useState(false);
  const [isLoadingForeign, setIsLoadingForeign] = useState(false);

  useEffect(() => {
    if (refVideo) {
      loadRefFrame(refTime);
    }
  }, [refVideo]);

  useEffect(() => {
    if (foreignVideo) {
      loadForeignFrame(foreignTime);
    }
  }, [foreignVideo]);

  const loadRefFrame = async (time: number) => {
    if (!refVideo) return;
    setIsLoadingRef(true);
    try {
      const frame = await window.electronAPI.extractFrame(refVideo.path, time);
      setRefFrame(frame);
    } catch (error) {
      console.error('Failed to extract reference frame:', error);
    }
    setIsLoadingRef(false);
  };

  const loadForeignFrame = async (time: number) => {
    if (!foreignVideo) return;
    setIsLoadingForeign(true);
    try {
      const frame = await window.electronAPI.extractFrame(foreignVideo.path, time);
      setForeignFrame(frame);
    } catch (error) {
      console.error('Failed to extract foreign frame:', error);
    }
    setIsLoadingForeign(false);
  };

  const handleRefTimeChange = (newTime: number) => {
    if (!refVideo) return;
    const clampedTime = Math.max(0, Math.min(refVideo.duration, newTime));
    setRefTime(clampedTime);
    loadRefFrame(clampedTime);
  };

  const handleForeignTimeChange = (newTime: number) => {
    if (!foreignVideo) return;
    const clampedTime = Math.max(0, Math.min(foreignVideo.duration, newTime));
    setForeignTime(clampedTime);
    loadForeignFrame(clampedTime);
  };

  const handleAddSyncPoint = () => {
    if (!refVideo || !foreignVideo) return;
    onAddSyncPoint({ refTime, foreignTime });
  };

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = (seconds % 60).toFixed(2);
    return h > 0
      ? `${h}:${m.toString().padStart(2, '0')}:${s.padStart(5, '0')}`
      : `${m}:${s.padStart(5, '0')}`;
  };

  const formatDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const stepFrame = (video: 'ref' | 'foreign', direction: number) => {
    const frameRate = 24; // Default, could be extracted from metadata
    const frameDuration = 1 / frameRate;
    const step = direction * frameDuration;

    if (video === 'ref') {
      handleRefTimeChange(refTime + step);
    } else {
      handleForeignTimeChange(foreignTime + step);
    }
  };

  const jumpTime = (video: 'ref' | 'foreign', seconds: number) => {
    if (video === 'ref') {
      handleRefTimeChange(refTime + seconds);
    } else {
      handleForeignTimeChange(foreignTime + seconds);
    }
  };

  return (
    <div className="video-preview">
      <h2 className="section-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polygon points="23 7 16 12 23 17 23 7"/>
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
        </svg>
        Video Preview & Frame Navigation
      </h2>

      <div className="preview-container">
        <div className="preview-panel">
          <div className="preview-header">
            <span className="preview-label">
              Reference Video
              {refVideo && (
                <span className="preview-metadata">
                  {refVideo.path.split(/[/\\]/).pop()?.replace(/\.[^/.]+$/, '')} •{' '}
                  {formatDuration(refVideo.duration)} • {refVideo.videoStreams.length}V{' '}
                  {refVideo.audioStreams.length}A
                </span>
              )}
            </span>
            <span className="preview-time">{formatTime(refTime)}</span>
          </div>
          <div className="preview-frame">
            {isLoadingRef ? (
              <div className="preview-loading">Loading frame...</div>
            ) : refFrame ? (
              <img src={refFrame} alt="Reference frame" />
            ) : (
              <div className="preview-placeholder">
                {refVideo ? 'Click timeline to load frame' : 'No video loaded'}
              </div>
            )}
          </div>
          <div className="preview-controls">
            <button onClick={() => jumpTime('ref', -10)} disabled={!refVideo}>
              -10s
            </button>
            <button onClick={() => jumpTime('ref', -1)} disabled={!refVideo}>
              -1s
            </button>
            <button onClick={() => stepFrame('ref', -1)} disabled={!refVideo}>
              &lt; Frame
            </button>
            <button onClick={() => stepFrame('ref', 1)} disabled={!refVideo}>
              Frame &gt;
            </button>
            <button onClick={() => jumpTime('ref', 1)} disabled={!refVideo}>
              +1s
            </button>
            <button onClick={() => jumpTime('ref', 10)} disabled={!refVideo}>
              +10s
            </button>
          </div>
          <input
            type="range"
            className="timeline-slider"
            min="0"
            max={refVideo?.duration || 0}
            step="0.01"
            value={refTime}
            onChange={(e) => handleRefTimeChange(parseFloat(e.target.value))}
            disabled={!refVideo}
          />
        </div>

        <div className="preview-panel">
          <div className="preview-header">
            <span className="preview-label">
              Foreign Video
              {foreignVideo && (
                <span className="preview-metadata">
                  {foreignVideo.path.split(/[/\\]/).pop()?.replace(/\.[^/.]+$/, '')} •{' '}
                  {formatDuration(foreignVideo.duration)} • {foreignVideo.videoStreams.length}V{' '}
                  {foreignVideo.audioStreams.length}A
                </span>
              )}
            </span>
            <span className="preview-time">{formatTime(foreignTime)}</span>
          </div>
          <div className="preview-frame">
            {isLoadingForeign ? (
              <div className="preview-loading">Loading frame...</div>
            ) : foreignFrame ? (
              <img src={foreignFrame} alt="Foreign frame" />
            ) : (
              <div className="preview-placeholder">
                {foreignVideo ? 'Click timeline to load frame' : 'No video loaded'}
              </div>
            )}
          </div>
          <div className="preview-controls">
            <button onClick={() => jumpTime('foreign', -10)} disabled={!foreignVideo}>
              -10s
            </button>
            <button onClick={() => jumpTime('foreign', -1)} disabled={!foreignVideo}>
              -1s
            </button>
            <button onClick={() => stepFrame('foreign', -1)} disabled={!foreignVideo}>
              &lt; Frame
            </button>
            <button onClick={() => stepFrame('foreign', 1)} disabled={!foreignVideo}>
              Frame &gt;
            </button>
            <button onClick={() => jumpTime('foreign', 1)} disabled={!foreignVideo}>
              +1s
            </button>
            <button onClick={() => jumpTime('foreign', 10)} disabled={!foreignVideo}>
              +10s
            </button>
          </div>
          <input
            type="range"
            className="timeline-slider"
            min="0"
            max={foreignVideo?.duration || 0}
            step="0.01"
            value={foreignTime}
            onChange={(e) => handleForeignTimeChange(parseFloat(e.target.value))}
            disabled={!foreignVideo}
          />
        </div>
      </div>

      <div className="sync-point-action">
        <span className="sync-point-hint">
          Manual sync points are not necessary, but use if you find some specific scenes not in perfect sync.
        </span>
        <button
          className="btn-add-sync-point"
          onClick={handleAddSyncPoint}
          disabled={!refVideo || !foreignVideo}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="2" />
          </svg>
          Add Sync Point at Current Positions
        </button>
      </div>
    </div>
  )
}
