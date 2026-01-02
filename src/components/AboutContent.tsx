import './AboutContent.css'

export default function AboutContent() {
  return (
    <div className="about-content-panel">
      <div className="about-scroll">
        <h2 className="about-title">About AVSync</h2>

        <section className="about-section">
          <h3>What is AVSync?</h3>
          <p>
            AVSync is an automated audio/video synchronization tool designed to align foreign language audio tracks
            with reference video content. It intelligently analyzes both video and audio streams to create perfectly
            synchronized output files.
          </p>
        </section>

        <section className="about-section">
          <h3>Key Features</h3>
          <ul>
            <li><strong>Automated Scene Detection:</strong> Analyzes video frames to identify scene changes and key moments</li>
            <li><strong>Audio Fingerprinting:</strong> Uses advanced audio analysis to match and align audio tracks</li>
            <li><strong>Manual Sync Points:</strong> Define precise synchronization points by navigating to matching frames</li>
            <li><strong>Batch Processing:</strong> Queue multiple jobs and process them sequentially</li>
            <li><strong>Stream Auto-Detection:</strong> Automatically identifies and selects the correct audio/video streams</li>
            <li><strong>Flexible Output:</strong> Customize audio codec, bitrate, and muxing options</li>
            <li><strong>Subtitle Support:</strong> Optionally synchronize subtitle tracks along with audio</li>
          </ul>
        </section>

        <section className="about-section">
          <h3>How It Works</h3>
          <ol>
            <li><strong>Load Videos:</strong> Select your reference video and the foreign video to synchronize</li>
            <li><strong>Define Sync Points:</strong> Add manual sync points or let the tool auto-detect alignments</li>
            <li><strong>Configure Parameters:</strong> Adjust scene detection, audio matching, and output settings</li>
            <li><strong>Process:</strong> Add jobs to the queue and start processing</li>
            <li><strong>Review:</strong> Monitor progress through detailed logs and status indicators</li>
          </ol>
        </section>

        <section className="about-section">
          <h3>Use Cases</h3>
          <ul>
            <li>Synchronizing dubbed audio tracks with original video content</li>
            <li>Aligning foreign language releases with reference versions</li>
            <li>Matching audio from different sources to the same video</li>
            <li>Creating multi-language video releases with precise timing</li>
          </ul>
        </section>

        <section className="about-section">
          <h3>Technical Details</h3>
          <p>
            AVSync leverages FFmpeg for video/audio processing and uses sophisticated algorithms for scene detection,
            audio fingerprinting, and temporal alignment. The tool supports various video containers (MKV, MP4, AVI)
            and audio codecs (AAC, AC3, and more).
          </p>
        </section>
      </div>
    </div>
  )
}
