import { useState } from 'react'
import './ParametersPanel.css'

interface ParametersPanelProps {
  parameters: any;
  onParametersChange: (params: any) => void;
}

export default function ParametersPanel({ parameters, onParametersChange }: ParametersPanelProps) {
  const [expandedSections, setExpandedSections] = useState({
    core: true,
    advanced: false,
    image: false,
    audio: false,
    muxing: false,
    cache: false,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleChange = (key: string, value: any) => {
    onParametersChange({ ...parameters, [key]: value });
  };

  return (
    <div className="parameters-panel">
      <h2 className="section-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m4.24-4.24l4.24-4.24"/>
        </svg>
        AVSync Parameters
      </h2>

      <div className="params-scroll">
        {/* Core Parameters */}
        <div className="param-section">
          <button
            className="param-section-header"
            onClick={() => toggleSection('core')}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="currentColor"
              style={{
                transform: expandedSections.core ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            >
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <span>Core Parameters</span>
          </button>

          {expandedSections.core && (
            <div className="param-section-content">
              <div className="param-item">
                <label className="param-label">
                  First Segment Adjust
                  <span className="param-hint">Milliseconds (+pad/-trim)</span>
                </label>
                <input
                  type="number"
                  step="1"
                  value={parameters.firstSegmentAdjust || 0}
                  onChange={(e) => handleChange('firstSegmentAdjust', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item">
                <label className="param-label">
                  Last Segment Adjust
                  <span className="param-hint">Milliseconds (+pad/-trim)</span>
                </label>
                <input
                  type="number"
                  step="1"
                  value={parameters.lastSegmentAdjust || 0}
                  onChange={(e) => handleChange('lastSegmentAdjust', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item param-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={parameters.noSubtitles || false}
                    onChange={(e) => handleChange('noSubtitles', e.target.checked)}
                  />
                  <span>Skip subtitle synchronization</span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Advanced Parameters */}
        <div className="param-section">
          <button
            className="param-section-header"
            onClick={() => toggleSection('advanced')}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="currentColor"
              style={{
                transform: expandedSections.advanced ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            >
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <span>Advanced Parameters</span>
          </button>

          {expandedSections.advanced && (
            <div className="param-section-content">
              <div className="param-item">
                <label className="param-label">Reference Language (ISO 639-2)</label>
                <input
                  type="text"
                  maxLength={3}
                  value={parameters.refLang}
                  onChange={(e) => handleChange('refLang', e.target.value)}
                  className="param-input"
                  placeholder="eng"
                />
              </div>

              <div className="param-item">
                <label className="param-label">Foreign Language (ISO 639-2)</label>
                <input
                  type="text"
                  maxLength={3}
                  value={parameters.foreignLang}
                  onChange={(e) => handleChange('foreignLang', e.target.value)}
                  className="param-input"
                  placeholder="spa"
                />
              </div>

              <div className="param-item">
                <label className="param-label">
                  dB Threshold
                  <span className="param-hint">Audio detection threshold (dBFS)</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={parameters.dbThreshold}
                  onChange={(e) => handleChange('dbThreshold', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item">
                <label className="param-label">
                  Min Segment Duration
                  <span className="param-hint">Seconds</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={parameters.minSegmentDuration}
                  onChange={(e) => handleChange('minSegmentDuration', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item param-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={parameters.autoDetect || false}
                    onChange={(e) => handleChange('autoDetect', e.target.checked)}
                  />
                  <span>Auto-detect streams (skip prompts)</span>
                </label>
              </div>

              {!parameters.autoDetect && (
                <>
                  <div className="param-item">
                    <label className="param-label">
                      Reference Audio Stream
                      <span className="param-hint">Audio stream index (0-based)</span>
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="0"
                      value={parameters.refAudioStream || 0}
                      onChange={(e) => handleChange('refAudioStream', parseInt(e.target.value))}
                      className="param-input"
                      placeholder="0"
                    />
                  </div>

                  <div className="param-item">
                    <label className="param-label">
                      Foreign Audio Stream
                      <span className="param-hint">Audio stream index (0-based)</span>
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="0"
                      value={parameters.foreignAudioStream || 0}
                      onChange={(e) => handleChange('foreignAudioStream', parseInt(e.target.value))}
                      className="param-input"
                      placeholder="0"
                    />
                  </div>
                </>
              )}

              <div className="param-item">
                <label className="param-label">Foreign Audio Codec</label>
                <select
                  value={parameters.muxForeignCodec}
                  onChange={(e) => handleChange('muxForeignCodec', e.target.value)}
                  className="param-input"
                >
                  <option value="aac">AAC</option>
                  <option value="ac3">AC3</option>
                  <option value="copy">Copy (no re-encode)</option>
                </select>
              </div>

              <div className="param-item">
                <label className="param-label">Foreign Audio Bitrate</label>
                <select
                  value={parameters.muxForeignBitrate}
                  onChange={(e) => handleChange('muxForeignBitrate', e.target.value)}
                  className="param-input"
                >
                  <option value="128k">128 kbps</option>
                  <option value="192k">192 kbps</option>
                  <option value="256k">256 kbps</option>
                  <option value="320k">320 kbps</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Image Pairing */}
        <div className="param-section">
          <button
            className="param-section-header"
            onClick={() => toggleSection('image')}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="currentColor"
              style={{
                transform: expandedSections.image ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            >
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <span>Image Pairing</span>
          </button>

          {expandedSections.image && (
            <div className="param-section-content">
              <div className="param-item">
                <label className="param-label">
                  Scene Threshold
                  <span className="param-hint">0.0 - 1.0 (lower = more changes)</span>
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={parameters.sceneThreshold}
                  onChange={(e) => handleChange('sceneThreshold', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item">
                <label className="param-label">
                  Match Threshold
                  <span className="param-hint">0.0 - 1.0 (template matching score)</span>
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={parameters.matchThreshold}
                  onChange={(e) => handleChange('matchThreshold', parseFloat(e.target.value))}
                  className="param-input"
                />
              </div>

              <div className="param-item">
                <label className="param-label">
                  Similarity Threshold
                  <span className="param-hint">pHash difference (-1 to disable)</span>
                </label>
                <input
                  type="number"
                  step="1"
                  min="-1"
                  value={parameters.similarityThreshold}
                  onChange={(e) => handleChange('similarityThreshold', parseInt(e.target.value))}
                  className="param-input"
                />
              </div>
            </div>
          )}
        </div>

        {/* Cache & Logging */}
        <div className="param-section">
          <button
            className="param-section-header"
            onClick={() => toggleSection('cache')}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="currentColor"
              style={{
                transform: expandedSections.cache ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            >
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <span>Cache & Logging</span>
          </button>

          {expandedSections.cache && (
            <div className="param-section-content">
              <div className="param-item param-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={parameters.useCache !== false}
                    onChange={(e) => handleChange('useCache', e.target.checked)}
                  />
                  <span>Use checkpoint cache</span>
                </label>
              </div>

              <div className="param-item param-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={parameters.verbose || false}
                    onChange={(e) => handleChange('verbose', e.target.checked)}
                  />
                  <span>Verbose logging</span>
                </label>
              </div>

              <div className="param-item param-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={parameters.showWarnings || false}
                    onChange={(e) => handleChange('showWarnings', e.target.checked)}
                  />
                  <span>Show warnings</span>
                </label>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
