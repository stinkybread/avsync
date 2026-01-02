import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let avSyncProcess: ChildProcess | null = null;

const isDev = process.env.NODE_ENV === 'development';
const isWindows = process.platform === 'win32';

function getResourcePath(relativePath: string): string {
  if (isDev) {
    // In development, resources are in the project root
    return path.join(__dirname, '..', relativePath);
  }
  // In production, resources are bundled by electron-builder in process.resourcesPath
  // extraResources are placed directly in process.resourcesPath, not nested in 'resources'
  return path.join(process.resourcesPath, relativePath);
}

function getBinaryPath(binaryName: string): string {
  const ext = isWindows ? '.exe' : '';
  const binaryFile = `${binaryName}${ext}`;

  // AVSync uses its own folder structure (PyInstaller bundle)
  if (binaryName === 'avsync') {
    if (isDev) {
      return getResourcePath(path.join('resources', 'avsync', binaryFile));
    }
    // In production, extraResources are copied to process.resourcesPath/avsync
    return getResourcePath(path.join('avsync', binaryFile));
  }

  // Other binaries are in resources/bin in dev, bin in production
  if (isDev) {
    return getResourcePath(path.join('resources', 'bin', binaryFile));
  }
  return getResourcePath(path.join('bin', binaryFile));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1800,
    height: 900,
    minWidth: 1800,
    minHeight: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    autoHideMenuBar: true,
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (avSyncProcess) {
    avSyncProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers

ipcMain.handle('select-file', async (_event, options: {
  title: string;
  filters?: { name: string; extensions: string[] }[];
  properties?: ('openFile' | 'openDirectory')[];
}) => {
  if (!mainWindow) return null;

  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title,
    filters: options.filters || [],
    properties: options.properties || ['openFile'],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

ipcMain.handle('select-save-file', async (_event, options: {
  title: string;
  defaultPath?: string;
  filters?: { name: string; extensions: string[] }[];
}) => {
  if (!mainWindow) return null;

  const result = await dialog.showSaveDialog(mainWindow, {
    title: options.title,
    defaultPath: options.defaultPath,
    filters: options.filters || [],
  });

  if (result.canceled || !result.filePath) {
    return null;
  }

  return result.filePath;
});

ipcMain.handle('select-directory', async () => {
  if (!mainWindow) return null;

  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

ipcMain.handle('list-directory', async (_event, dirPath: string) => {
  try {
    const files = await fs.promises.readdir(dirPath);
    return files;
  } catch (error) {
    console.error('Error reading directory:', error);
    return [];
  }
});

ipcMain.handle('get-video-metadata', async (_event, videoPath: string) => {
  const ffprobePath = getBinaryPath('ffprobe');

  return new Promise((resolve, reject) => {
    const ffprobe = spawn(ffprobePath, [
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      videoPath,
    ]);

    let output = '';
    let errorOutput = '';

    ffprobe.stdout.on('data', (data) => {
      output += data.toString();
    });

    ffprobe.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    ffprobe.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (error) {
          reject(new Error('Failed to parse ffprobe output'));
        }
      } else {
        reject(new Error(`ffprobe failed: ${errorOutput}`));
      }
    });
  });
});

ipcMain.handle('extract-frame', async (_event, videoPath: string, timestamp: number) => {
  const ffmpegPath = getBinaryPath('ffmpeg');
  const tempDir = app.getPath('temp');
  const outputPath = path.join(tempDir, `frame_${Date.now()}.jpg`);

  return new Promise((resolve, reject) => {
    const ffmpeg = spawn(ffmpegPath, [
      '-ss', timestamp.toString(),
      '-i', videoPath,
      '-vframes', '1',
      '-q:v', '2',
      '-y',
      outputPath,
    ]);

    let errorOutput = '';

    ffmpeg.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    ffmpeg.on('close', (code) => {
      if (code === 0 && fs.existsSync(outputPath)) {
        const imageData = fs.readFileSync(outputPath, 'base64');
        fs.unlinkSync(outputPath);
        resolve(`data:image/jpeg;base64,${imageData}`);
      } else {
        reject(new Error(`Failed to extract frame: ${errorOutput}`));
      }
    });
  });
});

ipcMain.handle('run-avsync', async (_event, args: {
  refVideo: string;
  foreignVideo: string;
  outputVideo: string;
  params: Record<string, any>;
}) => {
  const avSyncPath = getBinaryPath('avsync');

  const cmdArgs = [
    args.refVideo,
    args.foreignVideo,
    args.outputVideo,
  ];

  // Add optional parameters
  if (args.params.outputAudio) cmdArgs.push('--output_audio', args.params.outputAudio);
  if (args.params.outputCsv) cmdArgs.push('--output_csv', args.params.outputCsv);
  if (args.params.noSubtitles) cmdArgs.push('--no_subtitles');
  if (args.params.qcOutputDir) cmdArgs.push('--qc_output_dir', args.params.qcOutputDir);

  // Image pairing parameters
  if (args.params.sceneThreshold !== undefined) cmdArgs.push('--scene_threshold', args.params.sceneThreshold.toString());
  if (args.params.matchThreshold !== undefined) cmdArgs.push('--match_threshold', args.params.matchThreshold.toString());
  if (args.params.similarityThreshold !== undefined) cmdArgs.push('--similarity_threshold', args.params.similarityThreshold.toString());
  if (args.params.forceSyncPoints) cmdArgs.push('--force_sync_points', args.params.forceSyncPoints);

  // Audio parameters
  if (args.params.refLang) cmdArgs.push('--ref_lang', args.params.refLang);
  if (args.params.foreignLang) cmdArgs.push('--foreign_lang', args.params.foreignLang);
  if (args.params.dbThreshold !== undefined) cmdArgs.push('--db_threshold', args.params.dbThreshold.toString());
  if (args.params.minSegmentDuration !== undefined) cmdArgs.push('--min_segment_duration', args.params.minSegmentDuration.toString());
  if (args.params.firstSegmentAdjust !== undefined) cmdArgs.push('--first_segment_adjust', args.params.firstSegmentAdjust.toString());
  if (args.params.lastSegmentAdjust !== undefined) cmdArgs.push('--last_segment_adjust', args.params.lastSegmentAdjust.toString());
  if (args.params.refStreamIdx !== undefined) cmdArgs.push('--ref_stream_idx', args.params.refStreamIdx.toString());
  if (args.params.foreignStreamIdx !== undefined) cmdArgs.push('--foreign_stream_idx', args.params.foreignStreamIdx.toString());
  if (args.params.autoDetect) cmdArgs.push('--auto_detect');

  // Muxing parameters
  if (args.params.muxForeignCodec) cmdArgs.push('--mux_foreign_codec', args.params.muxForeignCodec);
  if (args.params.muxForeignBitrate) cmdArgs.push('--mux_foreign_bitrate', args.params.muxForeignBitrate);

  // Cache parameters
  if (args.params.useCache === false) cmdArgs.push('--no-cache');

  // Logging parameters
  if (args.params.verbose) cmdArgs.push('--verbose');
  if (args.params.showWarnings) cmdArgs.push('--show-warnings');
  if (args.params.logFile) cmdArgs.push('--log-file', args.params.logFile);
  if (args.params.noLog) cmdArgs.push('--no-log');

  avSyncProcess = spawn(avSyncPath, cmdArgs);

  avSyncProcess.stdout?.on('data', (data) => {
    mainWindow?.webContents.send('avsync-log', {
      type: 'stdout',
      data: data.toString(),
    });
  });

  avSyncProcess.stderr?.on('data', (data) => {
    mainWindow?.webContents.send('avsync-log', {
      type: 'stderr',
      data: data.toString(),
    });
  });

  avSyncProcess.on('close', (code) => {
    mainWindow?.webContents.send('avsync-complete', { code });
    avSyncProcess = null;
  });

  return { success: true };
});

ipcMain.handle('stop-avsync', async () => {
  if (avSyncProcess) {
    avSyncProcess.kill('SIGTERM');
    avSyncProcess = null;
    return { success: true };
  }
  return { success: false, message: 'No process running' };
});

ipcMain.handle('get-app-version', async () => {
  return app.getVersion();
});
