import { useState, useEffect } from 'react'
import FileSelector from './components/FileSelector'
import VideoPreview from './components/VideoPreview'
import ParametersPanel from './components/ParametersPanel'
import LogViewer from './components/LogViewer'
import JobQueue, { Job, JobStatus } from './components/JobQueue'
import SyncPointPills from './components/SyncPointPills'
import TabPanel from './components/TabPanel'
import BatchJob, { BatchJobItem } from './components/BatchJob'
import BatchStagingTable from './components/BatchStagingTable'
import AboutContent from './components/AboutContent'
import './App.css'

export type SyncPoint = {
  refTime: number;
  foreignTime: number;
};

export type VideoFile = {
  path: string;
  duration: number;
  audioStreams: any[];
  videoStreams: any[];
};

function App() {
  // Theme state
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Current job configuration
  const [refVideo, setRefVideo] = useState<VideoFile | null>(null);
  const [foreignVideo, setForeignVideo] = useState<VideoFile | null>(null);
  const [outputPath, setOutputPath] = useState<string>('');
  const [syncPoints, setSyncPoints] = useState<SyncPoint[]>([]);
  const [parameters, setParameters] = useState<any>({
    sceneThreshold: 0.25,
    matchThreshold: 0.7,
    similarityThreshold: 4,
    refLang: 'eng',
    foreignLang: 'spa',
    dbThreshold: -40.0,
    minSegmentDuration: 0.5,
    muxForeignCodec: 'aac',
    muxForeignBitrate: '192k',
    useCache: true,
    autoDetect: true,
  });

  // Job queue state
  const [jobs, setJobs] = useState<Job[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);

  // Batch staging state
  const [batchStagingJobs, setBatchStagingJobs] = useState<BatchJobItem[]>([]);
  const [batchEditingId, setBatchEditingId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('new-job');

  // Get current job
  const currentJob = jobs.find(j => j.id === currentJobId);
  const currentLogs = currentJob?.logs || [];

  // Set up IPC listeners
  useEffect(() => {
    window.electronAPI.onAVSyncLog((logData) => {
      if (currentJobId) {
        setJobs(prevJobs =>
          prevJobs.map(job =>
            job.id === currentJobId
              ? { ...job, logs: [...job.logs, logData] }
              : job
          )
        );
      }
    });

    window.electronAPI.onAVSyncComplete((data) => {
      if (currentJobId) {
        setJobs(prevJobs =>
          prevJobs.map(job =>
            job.id === currentJobId
              ? {
                  ...job,
                  status: data.code === 0 ? 'completed' : 'failed',
                  completedAt: new Date(),
                  error: data.code !== 0 ? `Process failed with exit code ${data.code}` : undefined,
                  logs: [
                    ...job.logs,
                    {
                      type: data.code === 0 ? 'success' : 'error',
                      data: `Process completed with exit code: ${data.code}`,
                    },
                  ],
                }
              : job
          )
        );

        // Process next job in queue
        setTimeout(() => {
          const updatedJobs = jobs.map(job =>
            job.id === currentJobId
              ? {
                  ...job,
                  status: data.code === 0 ? 'completed' as JobStatus : 'failed' as JobStatus,
                  completedAt: new Date(),
                }
              : job
          );
          processNextJob(updatedJobs);
        }, 100);
      }
    });
  }, [currentJobId, jobs]);

  const processNextJob = (jobList: Job[]) => {
    const nextJob = jobList.find(j => j.status === 'pending');
    if (!nextJob) {
      setCurrentJobId(null);
      return;
    }

    setCurrentJobId(nextJob.id);
    setJobs(prevJobs =>
      prevJobs.map(job =>
        job.id === nextJob.id
          ? { ...job, status: 'processing', startedAt: new Date() }
          : job
      )
    );

    // Execute the job
    setTimeout(async () => {
      try {
        await window.electronAPI.runAVSync({
          refVideo: nextJob.refVideo,
          foreignVideo: nextJob.foreignVideo,
          outputVideo: nextJob.outputVideo,
          params: nextJob.params,
        });
      } catch (error) {
        setJobs(prevJobs =>
          prevJobs.map(job =>
            job.id === nextJob.id
              ? {
                  ...job,
                  status: 'failed',
                  completedAt: new Date(),
                  error: `Error: ${error}`,
                  logs: [...job.logs, { type: 'error', data: `Error: ${error}` }],
                }
              : job
          )
        );
        processNextJob(jobList);
      }
    }, 500);
  };

  const handleAddToQueue = () => {
    if (!refVideo || !foreignVideo || !outputPath) {
      alert('Please select reference video, foreign video, and output path');
      return;
    }

    const syncParams = { ...parameters };

    // Add sync points if manually specified
    if (syncPoints.length > 0) {
      const syncSpec = syncPoints
        .map((sp) => `${sp.refTime.toFixed(2)}:${sp.foreignTime.toFixed(2)}`)
        .join(',');
      syncParams.forceSyncPoints = syncSpec;
    }

    const newJob: Job = {
      id: `job-${Date.now()}`,
      refVideo: refVideo.path,
      foreignVideo: foreignVideo.path,
      outputVideo: outputPath,
      status: 'pending',
      logs: [],
      createdAt: new Date(),
      params: syncParams,
    };

    setJobs([...jobs, newJob]);

    // Don't auto-start - user must manually start the queue

    // Clear the form for next job (optional)
    // setRefVideo(null);
    // setForeignVideo(null);
    // setOutputPath('');
    // setSyncPoints([]);
  };

  const handleStartQueue = () => {
    if (!currentJobId && jobs.some(j => j.status === 'pending')) {
      processNextJob(jobs);
    }
  };

  const handleAbortJob = async (jobId: string) => {
    await window.electronAPI.stopAVSync();
    setJobs(prevJobs =>
      prevJobs.map(job =>
        job.id === jobId
          ? {
              ...job,
              status: 'aborted' as JobStatus,
              completedAt: new Date(),
              error: 'Job aborted by user',
            }
          : job
      )
    );
    setTimeout(() => processNextJob(jobs), 100);
  };

  const handleRemoveJob = (jobId: string) => {
    setJobs(prevJobs => prevJobs.filter(j => j.id !== jobId));
  };

  const handleRetryJob = (jobId: string) => {
    setJobs(prevJobs =>
      prevJobs.map(job =>
        job.id === jobId
          ? { ...job, status: 'pending', logs: [], error: undefined }
          : job
      )
    );

    if (!currentJobId) {
      const updatedJobs = jobs.map(job =>
        job.id === jobId
          ? { ...job, status: 'pending' as JobStatus, logs: [], error: undefined }
          : job
      );
      processNextJob(updatedJobs);
    }
  };

  const handleViewLogs = (jobId: string) => {
    setCurrentJobId(jobId);
  };

  const handleClearAll = () => {
    if (confirm('Clear all jobs from the queue?')) {
      setJobs([]);
      setCurrentJobId(null);
    }
  };

  const handleBatchStagingJobsChange = (stagingJobs: BatchJobItem[]) => {
    setBatchStagingJobs(stagingJobs);
  };

  const handleBatchEditingIdChange = (id: string | null) => {
    setBatchEditingId(id);
  };

  const handleBatchEditJob = (id: string, field: keyof BatchJobItem, value: any) => {
    const newJobs = batchStagingJobs.map(j => j.id === id ? { ...j, [field]: value } : j);
    setBatchStagingJobs(newJobs);
  };

  const handleBatchRemoveJob = (id: string) => {
    const newJobs = batchStagingJobs.filter(j => j.id !== id);
    setBatchStagingJobs(newJobs);
  };

  const handleBatchAddToQueue = (batchJobs: BatchJobItem[]) => {
    const newJobs: Job[] = batchJobs.map(batchJob => ({
      id: `job-${Date.now()}-${Math.random()}`,
      refVideo: batchJob.refVideo,
      foreignVideo: batchJob.foreignVideo,
      outputVideo: batchJob.outputVideo,
      status: 'pending' as JobStatus,
      logs: [],
      createdAt: new Date(),
      params: {
        ...parameters,
        firstSegmentAdjust: batchJob.firstSegmentAdjust,
        lastSegmentAdjust: batchJob.lastSegmentAdjust,
        noSubtitles: batchJob.skipSubtitles,
      },
    }));

    setJobs([...jobs, ...newJobs]);
    setBatchStagingJobs([]);
  };

  const isProcessing = currentJob?.status === 'processing';

  return (
    <div className="app">
      <div className="app-content">
        {/* Left Column: Tabbed Panel */}
        <div className="left-column">
          <TabPanel
            theme={theme}
            onToggleTheme={toggleTheme}
            activeTab={activeTab}
            onTabChange={setActiveTab}
            tabs={[
              {
                id: 'new-job',
                label: 'New Job',
                content: (
                  <div className="new-job-tab">
                    <div className="new-job-content">
                      <FileSelector
                        refVideo={refVideo}
                        foreignVideo={foreignVideo}
                        outputPath={outputPath}
                        onRefVideoChange={setRefVideo}
                        onForeignVideoChange={setForeignVideo}
                        onOutputPathChange={setOutputPath}
                      />
                      <ParametersPanel
                        parameters={parameters}
                        onParametersChange={setParameters}
                      />
                    </div>
                    <div className="new-job-footer">
                      <button className="btn btn-add-to-queue" onClick={handleAddToQueue}>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                          <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        Add to Queue
                      </button>
                    </div>
                  </div>
                ),
              },
              {
                id: 'batch',
                label: 'Batch',
                content: (
                  <BatchJob
                    parameters={parameters}
                    onAddToQueue={handleBatchAddToQueue}
                    onJobsChange={handleBatchStagingJobsChange}
                    onEditingIdChange={handleBatchEditingIdChange}
                  />
                ),
              },
              {
                id: 'queue',
                label: 'Queue',
                content: (
                  <JobQueue
                    jobs={jobs}
                    onRemoveJob={handleRemoveJob}
                    onRetryJob={handleRetryJob}
                    onAbortJob={handleAbortJob}
                    onViewLogs={handleViewLogs}
                    onClearAll={handleClearAll}
                    onStartQueue={handleStartQueue}
                    currentJobId={currentJobId || undefined}
                  />
                ),
              },
              {
                id: 'about',
                label: 'About',
                content: (
                  <div className="about-credits">
                    <div className="credits-content">
                      <h2 className="credits-title">Credits</h2>

                      <section className="credits-section">
                        <h3>Developer</h3>
                        <div className="credit-item">
                          <p className="credit-name">Vaibhav Bhat</p>
                          <a
                            href="https://github.com/stinkybread"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="credit-link"
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                            github.com/stinkybread
                          </a>
                        </div>
                      </section>

                      <section className="credits-section">
                        <h3>User Interface</h3>
                        <div className="credit-item">
                          <p className="credit-name">Claude (Anthropic)</p>
                          <p className="credit-role">Desktop Application UI Design & Implementation</p>
                        </div>
                      </section>
                    </div>
                  </div>
                ),
              },
            ]}
            defaultTab="new-job"
          />
        </div>

        {/* Right Column: Video Previews + Sync Points OR Batch Staging OR Queue Logs OR About Content */}
        <div className="right-column">
          {activeTab === 'batch' ? (
            <BatchStagingTable
              jobs={batchStagingJobs}
              editingId={batchEditingId}
              onEditJob={handleBatchEditJob}
              onRemoveJob={handleBatchRemoveJob}
              onSetEditingId={setBatchEditingId}
              onAddToQueue={() => handleBatchAddToQueue(batchStagingJobs)}
            />
          ) : activeTab === 'queue' ? (
            <LogViewer logs={currentLogs} isProcessing={isProcessing} />
          ) : activeTab === 'about' ? (
            <AboutContent />
          ) : (
            <>
              <div className="left-video">
                <VideoPreview
                  refVideo={refVideo}
                  foreignVideo={foreignVideo}
                  syncPoints={syncPoints}
                  onAddSyncPoint={(point) => setSyncPoints([...syncPoints, point])}
                />
              </div>
              <div className="left-sync">
                <SyncPointPills
                  syncPoints={syncPoints}
                  onSyncPointsChange={setSyncPoints}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
