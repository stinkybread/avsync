import { VideoFile } from '../App'
import './FileSelector.css'

interface FileSelectorProps {
  refVideo: VideoFile | null;
  foreignVideo: VideoFile | null;
  outputPath: string;
  onRefVideoChange: (video: VideoFile | null) => void;
  onForeignVideoChange: (video: VideoFile | null) => void;
  onOutputPathChange: (path: string) => void;
}

export default function FileSelector({
  refVideo,
  foreignVideo,
  outputPath,
  onRefVideoChange,
  onForeignVideoChange,
  onOutputPathChange,
}: FileSelectorProps) {
  const handleSelectRefVideo = async () => {
    const path = await window.electronAPI.selectFile({
      title: 'Select Reference Video',
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (path) {
      try {
        const metadata = await window.electronAPI.getVideoMetadata(path);
        const duration = parseFloat(metadata.format.duration || 0);
        const audioStreams = metadata.streams.filter((s: any) => s.codec_type === 'audio');
        const videoStreams = metadata.streams.filter((s: any) => s.codec_type === 'video');

        onRefVideoChange({
          path,
          duration,
          audioStreams,
          videoStreams,
        });
      } catch (error) {
        alert(`Failed to load video metadata: ${error}`);
      }
    }
  };

  const handleSelectForeignVideo = async () => {
    const path = await window.electronAPI.selectFile({
      title: 'Select Foreign Video',
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (path) {
      try {
        const metadata = await window.electronAPI.getVideoMetadata(path);
        const duration = parseFloat(metadata.format.duration || 0);
        const audioStreams = metadata.streams.filter((s: any) => s.codec_type === 'audio');
        const videoStreams = metadata.streams.filter((s: any) => s.codec_type === 'video');

        onForeignVideoChange({
          path,
          duration,
          audioStreams,
          videoStreams,
        });
      } catch (error) {
        alert(`Failed to load video metadata: ${error}`);
      }
    }
  };

  const handleSelectOutput = async () => {
    const suggestedName = refVideo
      ? refVideo.path.replace(/\.[^/.]+$/, '_synced.mkv')
      : 'output_synced.mkv';

    const path = await window.electronAPI.selectSaveFile({
      title: 'Select Output Path',
      defaultPath: suggestedName,
      filters: [
        { name: 'Matroska Video', extensions: ['mkv'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (path) {
      onOutputPathChange(path);
    }
  };

  return (
    <div className="file-selector">
      <h2 className="section-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
          <polyline points="13 2 13 9 20 9"/>
        </svg>
        Input/Output Files
      </h2>

      <div className="file-input-group">
        <label className="file-label">Reference Video (Original)</label>
        <div className="file-input-row">
          <button className="btn-select" onClick={handleSelectRefVideo}>
            Browse...
          </button>
          <div className="file-path">
            {refVideo ? (
              <span className="file-name">{refVideo.path.split(/[/\\]/).pop()}</span>
            ) : (
              <span className="file-placeholder">No file selected</span>
            )}
          </div>
        </div>
      </div>

      <div className="file-input-group">
        <label className="file-label">Foreign Video (To Sync)</label>
        <div className="file-input-row">
          <button className="btn-select" onClick={handleSelectForeignVideo}>
            Browse...
          </button>
          <div className="file-path">
            {foreignVideo ? (
              <span className="file-name">{foreignVideo.path.split(/[/\\]/).pop()}</span>
            ) : (
              <span className="file-placeholder">No file selected</span>
            )}
          </div>
        </div>
      </div>

      <div className="file-input-group">
        <label className="file-label">Output File</label>
        <div className="file-input-row">
          <button className="btn-select" onClick={handleSelectOutput}>
            Browse...
          </button>
          <div className="file-path">
            {outputPath ? (
              <span className="file-name">{outputPath.split(/[/\\]/).pop()}</span>
            ) : (
              <span className="file-placeholder">No file selected</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
