# AVSync Desktop

A desktop application for audio/video synchronization built with Electron and React.

## Features

- **Intuitive UI**: Modern React-based interface with video preview and frame-by-frame navigation
- **Video Preview**: View and navigate both reference and foreign videos frame-by-frame
- **Manual Sync Points**: Define precise synchronization points with visual feedback
- **Real-time Logging**: Monitor processing progress with real-time log output
- **Comprehensive Parameters**: Full control over all AVSync parameters through the UI
- **Cross-platform**: Builds for Windows and macOS

## Prerequisites

### For Development

- Node.js 18+ and npm
- Python 3.8+ with pip
- PyInstaller
- FFmpeg binaries
- MKVToolNix binaries

### For Building

All the same as development, plus:
- electron-builder dependencies for your platform

## Installation

### 1. Install Node Dependencies

```bash
npm install
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file:
```
opencv-python
numpy
scipy
tqdm
```

### 3. Build the AVSync Executable

```bash
npm run build:pyinstaller
```

This will create the executable in `dist/avsync/`.

### 4. Download Required Binaries

Download and place the following binaries in `resources/bin/`:

#### Windows:
- `ffmpeg.exe` - Download from https://ffmpeg.org/download.html
- `ffprobe.exe` - Included with FFmpeg
- `mkvmerge.exe` - Download from https://mkvtoolnix.download/

#### macOS:
- `ffmpeg` - Install via Homebrew: `brew install ffmpeg`
- `ffprobe` - Included with FFmpeg
- `mkvmerge` - Install via Homebrew: `brew install mkvtoolnix`

Copy the binaries to `resources/bin/`:
```bash
mkdir -p resources/bin
# Copy your downloaded binaries here
```

### 5. Copy AVSync Executable

Copy the PyInstaller-built AVSync executable to the resources:

```bash
# Windows
copy dist\avsync\avsync.exe resources\bin\

# macOS/Linux
cp dist/avsync/avsync resources/bin/
```

## Development

Run the development server:

```bash
npm run dev
```

This will:
1. Start the Vite dev server on http://localhost:5173
2. Launch Electron in development mode with hot-reload

## Building for Distribution

### Windows

```bash
npm run package:win
```

Creates:
- NSIS installer in `release/`
- Portable executable in `release/`

### macOS

```bash
npm run package:mac
```

Creates:
- DMG installer in `release/`
- ZIP archive in `release/`

## Project Structure

```
AVSync/
├── electron/              # Electron main process
│   ├── main.ts           # Main process entry point
│   └── preload.ts        # Preload script for IPC
├── src/                  # React application
│   ├── components/       # React components
│   │   ├── Header.tsx
│   │   ├── FileSelector.tsx
│   │   ├── VideoPreview.tsx
│   │   ├── SyncPointEditor.tsx
│   │   ├── ParametersPanel.tsx
│   │   └── LogViewer.tsx
│   ├── App.tsx          # Main app component
│   ├── main.tsx         # React entry point
│   └── index.css        # Global styles
├── resources/           # Application resources
│   └── bin/            # Bundled binaries
│       ├── avsync.exe  # PyInstaller-built AVSync
│       ├── ffmpeg.exe
│       ├── ffprobe.exe
│       └── mkvmerge.exe
├── AVSync_v12.py       # Original Python script
├── avsync.spec         # PyInstaller spec file
├── package.json        # Node dependencies and scripts
├── tsconfig.json       # TypeScript config
└── vite.config.ts      # Vite config
```

## Usage

1. **Select Videos**: Choose your reference video (original) and foreign video (to sync)
2. **Choose Output**: Select where to save the synchronized output file
3. **Set Parameters**: Adjust AVSync parameters in the right panel (or use defaults)
4. **Add Sync Points** (Optional):
   - Navigate to matching frames in both videos
   - Click "Add Sync Point" to manually specify synchronization points
5. **Run AVSync**: Click the "Run AVSync" button to start processing
6. **Monitor Progress**: Watch real-time logs in the log viewer

## Keyboard Shortcuts

### Video Navigation
- Frame navigation buttons for precise control
- Timeline slider for quick seeking
- Time jump buttons (-10s, -1s, +1s, +10s)

## Troubleshooting

### Binaries Not Found

If you see errors about missing binaries:
1. Ensure all binaries are in `resources/bin/` directory
2. On macOS/Linux, ensure binaries have execute permissions: `chmod +x resources/bin/*`
3. Rebuild the application: `npm run build && npm run package`

### PyInstaller Build Fails

If PyInstaller fails to build:
1. Ensure all Python dependencies are installed: `pip install -r requirements.txt`
2. Try building with verbose output: `pyinstaller --clean avsync.spec`
3. Check that AVSync_v12.py has no syntax errors

### FFmpeg/FFprobe Not Working

1. Test binaries directly from command line
2. Ensure correct binaries for your platform (Windows .exe vs Unix binaries)
3. Check file permissions

## License

MIT

## Credits

AVSync algorithm and Python implementation by Vaibhav
Desktop UI built with Electron, React, and TypeScript
