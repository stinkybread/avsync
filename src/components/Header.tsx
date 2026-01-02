import './Header.css'

interface HeaderProps {
  onStop: () => void;
  isProcessing: boolean;
}

export default function Header({ onStop, isProcessing }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title">AVSync Desktop</h1>
        <span className="header-subtitle">Audio/Video Synchronization Tool</span>
      </div>

      <div className="header-right">
        {isProcessing && (
          <button className="btn btn-danger" onClick={onStop}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <rect x="4" y="4" width="8" height="8" />
            </svg>
            Stop Processing
          </button>
        )}
      </div>
    </header>
  )
}
