import { useState } from 'react'
import './BatchJob.css'

export interface BatchJobItem {
  id: string
  refVideo: string
  foreignVideo: string
  outputVideo: string
  firstSegmentAdjust: number
  lastSegmentAdjust: number
  skipSubtitles: boolean
  status: 'pending' | 'processing' | 'completed' | 'failed'
}

interface BatchJobProps {
  parameters: any
  onAddToQueue: (jobs: BatchJobItem[]) => void
  onJobsChange?: (jobs: BatchJobItem[]) => void
  onEditingIdChange?: (id: string | null) => void
}

export default function BatchJob({ parameters, onAddToQueue, onJobsChange }: BatchJobProps) {
  const [refFolder, setRefFolder] = useState<string>('')
  const [foreignFolder, setForeignFolder] = useState<string>('')
  const [outputFolder, setOutputFolder] = useState<string>('')
  const [jobs, setJobs] = useState<BatchJobItem[]>([])

  const handleSelectFolder = async (type: 'ref' | 'foreign' | 'output') => {
    const path = await window.electronAPI.selectDirectory()
    if (!path) return

    if (type === 'ref') setRefFolder(path)
    else if (type === 'foreign') setForeignFolder(path)
    else setOutputFolder(path)
  }

  const handleLoadJobs = async () => {
    if (!refFolder || !foreignFolder || !outputFolder) {
      alert('Please select all three folders')
      return
    }

    try {
      const refFiles = await window.electronAPI.listDirectory(refFolder)
      const foreignFiles = await window.electronAPI.listDirectory(foreignFolder)

      // Filter video files
      const videoExtensions = ['.mkv', '.mp4', '.avi', '.mov', '.m4v']
      const refVideos = refFiles.filter(f =>
        videoExtensions.some(ext => f.toLowerCase().endsWith(ext))
      ).sort()
      const foreignVideos = foreignFiles.filter(f =>
        videoExtensions.some(ext => f.toLowerCase().endsWith(ext))
      ).sort()

      // Match by exact filename
      const matched: BatchJobItem[] = []
      for (const refFile of refVideos) {
        const matchingForeign = foreignVideos.find(f => f === refFile)
        if (matchingForeign) {
          matched.push({
            id: `batch-${Date.now()}-${Math.random()}`,
            refVideo: `${refFolder}/${refFile}`,
            foreignVideo: `${foreignFolder}/${matchingForeign}`,
            outputVideo: `${outputFolder}/${refFile}`,
            firstSegmentAdjust: parameters.firstSegmentAdjust || 0,
            lastSegmentAdjust: parameters.lastSegmentAdjust || 0,
            skipSubtitles: parameters.noSubtitles || false,
            status: 'pending'
          })
        }
      }

      if (matched.length === 0) {
        alert('No matching files found between reference and foreign folders')
      } else {
        setJobs(matched)
        onJobsChange?.(matched)
      }
    } catch (error) {
      console.error('Error loading jobs:', error)
      alert('Failed to load directory contents')
    }
  }

  const handleAddToQueue = () => {
    if (jobs.length === 0) {
      alert('No jobs to add to queue')
      return
    }
    onAddToQueue(jobs)
    setJobs([])
    onJobsChange?.([])
    setRefFolder('')
    setForeignFolder('')
    setOutputFolder('')
  }

  return (
    <div className="batch-job">
      <div className="batch-folders">
        <h2 className="section-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
          Folder Selection
        </h2>

        <div className="folder-inputs">
          <div className="folder-item">
            <label className="folder-label">Reference Folder</label>
            <div className="folder-input-group">
              <input
                type="text"
                value={refFolder}
                readOnly
                placeholder="No folder selected"
                className="folder-input"
              />
              <button className="btn-browse" onClick={() => handleSelectFolder('ref')}>
                Browse
              </button>
            </div>
          </div>

          <div className="folder-item">
            <label className="folder-label">Foreign Folder</label>
            <div className="folder-input-group">
              <input
                type="text"
                value={foreignFolder}
                readOnly
                placeholder="No folder selected"
                className="folder-input"
              />
              <button className="btn-browse" onClick={() => handleSelectFolder('foreign')}>
                Browse
              </button>
            </div>
          </div>

          <div className="folder-item">
            <label className="folder-label">Output Folder</label>
            <div className="folder-input-group">
              <input
                type="text"
                value={outputFolder}
                readOnly
                placeholder="No folder selected"
                className="folder-input"
              />
              <button className="btn-browse" onClick={() => handleSelectFolder('output')}>
                Browse
              </button>
            </div>
          </div>
        </div>

        <button
          className="btn-load-jobs"
          onClick={handleLoadJobs}
          disabled={!refFolder || !foreignFolder || !outputFolder}
        >
          Load Matched Files
        </button>
      </div>

      {jobs.length > 0 && (
        <div className="batch-info">
          <p className="batch-info-text">
            {jobs.length} job{jobs.length !== 1 ? 's' : ''} loaded. View and edit the staging table in the left panel, then add to queue.
          </p>
          <button className="btn-add-to-queue-bottom" onClick={handleAddToQueue}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Add {jobs.length} Job{jobs.length !== 1 ? 's' : ''} to Queue
          </button>
        </div>
      )}
    </div>
  )
}
