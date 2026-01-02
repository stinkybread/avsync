import { contextBridge, ipcRenderer } from 'electron';

export type AVSyncParams = {
  outputAudio?: string;
  outputCsv?: string;
  noSubtitles?: boolean;
  qcOutputDir?: string;
  sceneThreshold?: number;
  matchThreshold?: number;
  similarityThreshold?: number;
  forceSyncPoints?: string;
  refLang?: string;
  foreignLang?: string;
  dbThreshold?: number;
  minSegmentDuration?: number;
  firstSegmentAdjust?: number;
  lastSegmentAdjust?: number;
  refStreamIdx?: number;
  foreignStreamIdx?: number;
  autoDetect?: boolean;
  muxForeignCodec?: string;
  muxForeignBitrate?: string;
  useCache?: boolean;
  verbose?: boolean;
  showWarnings?: boolean;
  logFile?: string;
  noLog?: boolean;
};

export type ElectronAPI = {
  selectFile: (options: {
    title: string;
    filters?: { name: string; extensions: string[] }[];
    properties?: ('openFile' | 'openDirectory')[];
  }) => Promise<string | null>;

  selectSaveFile: (options: {
    title: string;
    defaultPath?: string;
    filters?: { name: string; extensions: string[] }[];
  }) => Promise<string | null>;

  selectDirectory: () => Promise<string | null>;

  listDirectory: (dirPath: string) => Promise<string[]>;

  getVideoMetadata: (videoPath: string) => Promise<any>;

  extractFrame: (videoPath: string, timestamp: number) => Promise<string>;

  runAVSync: (args: {
    refVideo: string;
    foreignVideo: string;
    outputVideo: string;
    params: AVSyncParams;
  }) => Promise<{ success: boolean }>;

  stopAVSync: () => Promise<{ success: boolean; message?: string }>;

  onAVSyncLog: (callback: (data: { type: string; data: string }) => void) => void;

  onAVSyncComplete: (callback: (data: { code: number }) => void) => void;

  getAppVersion: () => Promise<string>;
};

const electronAPI: ElectronAPI = {
  selectFile: (options) => ipcRenderer.invoke('select-file', options),
  selectSaveFile: (options) => ipcRenderer.invoke('select-save-file', options),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  listDirectory: (dirPath) => ipcRenderer.invoke('list-directory', dirPath),
  getVideoMetadata: (videoPath) => ipcRenderer.invoke('get-video-metadata', videoPath),
  extractFrame: (videoPath, timestamp) => ipcRenderer.invoke('extract-frame', videoPath, timestamp),
  runAVSync: (args) => ipcRenderer.invoke('run-avsync', args),
  stopAVSync: () => ipcRenderer.invoke('stop-avsync'),
  onAVSyncLog: (callback) => {
    ipcRenderer.on('avsync-log', (_event, data) => callback(data));
  },
  onAVSyncComplete: (callback) => {
    ipcRenderer.on('avsync-complete', (_event, data) => callback(data));
  },
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);
