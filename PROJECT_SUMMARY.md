# AVSync Desktop - Project Summary

## What Was Built

A complete Electron desktop application for AVSync with a modern React UI, providing:

### Core Features
1. **Video File Management**
   - Browse and select reference/foreign videos
   - Automatic metadata extraction (duration, streams)
   - Output file selection

2. **Video Preview & Navigation**
   - Side-by-side video frame preview
   - Frame-by-frame navigation
   - Timeline scrubbing
   - Time jump controls (-10s, -1s, +1s, +10s)
   - Real-time frame extraction using FFmpeg

3. **Manual Sync Point Editor**
   - Visual sync point creation
   - Editable sync point table
   - Time display in both seconds and HH:MM:SS format
   - Add/remove sync points dynamically

4. **Comprehensive Parameter Controls**
   - Image Pairing: scene threshold, match threshold, similarity threshold
   - Audio Processing: language codes, dB threshold, segment duration, adjustments
   - Muxing: codec selection, bitrate configuration
   - Advanced: caching, verbose logging, warnings

5. **Real-time Processing**
   - Execute AVSync with all parameters
   - Live log streaming (stdout/stderr)
   - Process control (start/stop)
   - Log export functionality
   - Visual processing indicators

## Project Structure

```
AVSync/
├── electron/                    # Electron main process
│   ├── main.ts                 # IPC handlers, window management
│   └── preload.ts              # Secure IPC bridge
│
├── src/                        # React application
│   ├── components/
│   │   ├── Header.tsx          # Top bar with Run/Stop buttons
│   │   ├── FileSelector.tsx    # File input controls
│   │   ├── VideoPreview.tsx    # Frame viewer + navigation
│   │   ├── SyncPointEditor.tsx # Sync point table editor
│   │   ├── ParametersPanel.tsx # Collapsible parameter groups
│   │   └── LogViewer.tsx       # Real-time log display
│   ├── App.tsx                 # Main app layout
│   ├── main.tsx                # React entry point
│   └── index.css               # Global dark theme
│
├── resources/                  # Bundled resources
│   └── bin/                    # Binary executables
│       ├── avsync[.exe]        # PyInstaller bundle
│       ├── ffmpeg[.exe]        # Video processing
│       ├── ffprobe[.exe]       # Metadata extraction
│       └── mkvmerge[.exe]      # Final muxing
│
├── Configuration Files
│   ├── package.json            # Node dependencies + build scripts
│   ├── tsconfig.json           # TypeScript config (React)
│   ├── tsconfig.electron.json  # TypeScript config (Electron)
│   ├── tsconfig.node.json      # TypeScript config (Vite)
│   ├── vite.config.ts          # Vite bundler config
│   ├── avsync.spec             # PyInstaller spec
│   └── .eslintrc.cjs           # Linting rules
│
└── Documentation
    ├── README.md               # Feature overview
    ├── BUILD_INSTRUCTIONS.md   # Detailed build guide
    ├── QUICKSTART.md           # 5-minute setup
    └── PROJECT_SUMMARY.md      # This file
```

## Technology Stack

### Frontend
- **React 18**: UI components
- **TypeScript**: Type safety
- **CSS Variables**: Dark theme system
- **Vite**: Fast development & bundling

### Backend
- **Electron 28**: Desktop framework
- **Node.js**: Process spawning, file I/O
- **IPC**: Secure renderer ↔ main communication

### Build Tools
- **PyInstaller**: Bundle Python script to executable
- **electron-builder**: Package for Windows/macOS
- **TypeScript Compiler**: Compile Electron main process

### External Dependencies
- **FFmpeg/FFprobe**: Video processing & metadata
- **MKVToolNix**: Final video muxing
- **AVSync (Python)**: Core synchronization algorithm

## Key Implementation Details

### IPC Architecture
```
React UI (Renderer)
    ↓ (IPC via preload.ts)
Electron Main Process
    ↓ (spawn/exec)
Binary Executables (FFmpeg, AVSync, etc.)
```

### Security
- Context isolation enabled
- Node integration disabled in renderer
- Preload script for safe IPC exposure
- Type-safe IPC definitions

### State Management
- React useState for UI state
- Parent → Child prop passing
- Event callbacks for state updates
- No external state library (keeps it simple)

### Real-time Features
- IPC event listeners for log streaming
- Auto-scrolling log viewer
- Process lifecycle management
- Graceful shutdown handling

## Build Process

### Development
```bash
npm run dev
```
- Vite dev server (React)
- Electron in dev mode
- Hot module reload
- DevTools enabled

### Production Build
```bash
npm run build          # Compile TS + bundle React
npm run package:win    # Windows installer/portable
npm run package:mac    # macOS DMG/ZIP
```

**Output:**
- Windows: NSIS installer + portable .exe
- macOS: DMG installer + ZIP archive

### PyInstaller Bundle
```bash
npm run build:pyinstaller
```
Creates standalone `avsync[.exe]` with all Python dependencies.

## Features by Component

### Header Component
- Application title
- Version display (future)
- Primary Run/Stop button
- Visual processing state

### FileSelector Component
- Three file pickers (ref, foreign, output)
- Metadata display (duration, streams)
- Auto-suggested output names
- File validation

### VideoPreview Component
- Dual video frame display
- FFmpeg frame extraction
- Navigation controls
- Timeline sliders
- Loading states
- Sync point addition

### SyncPointEditor Component
- Tabular sync point list
- Inline editing
- Add/remove operations
- Time format conversion
- Clear all functionality
- Usage hints

### ParametersPanel Component
- Collapsible sections
- All AVSync CLI parameters
- Type-appropriate inputs (number, select, checkbox)
- Helpful hints and ranges
- Default values

### LogViewer Component
- Auto-scrolling output
- Color-coded log levels
- Timestamp display
- Export to file
- Processing indicator
- Empty state messaging

## CLI Parameter Mapping

All AVSync parameters are supported:

**Positional:**
- ref_video, foreign_video, output_video

**Image Pairing:**
- --scene_threshold, --match_threshold, --similarity_threshold
- --force_sync_points (auto-generated from UI sync points)

**Audio:**
- --ref_lang, --foreign_lang, --db_threshold
- --min_segment_duration
- --first_segment_adjust, --last_segment_adjust
- --ref_stream_idx, --foreign_stream_idx, --auto_detect

**Muxing:**
- --mux_foreign_codec, --mux_foreign_bitrate
- --no_subtitles

**Advanced:**
- --use-cache / --no-cache
- --verbose, --show-warnings
- --log-file, --no-log
- --output_audio, --output_csv, --qc_output_dir

## UI/UX Design Decisions

### Dark Theme
- Reduces eye strain during long processing sessions
- Professional appearance
- High contrast for readability

### Layout
- Two-panel grid (main content | parameters + logs)
- Vertical sections for logical grouping
- Sticky headers for navigation context

### Color Coding
- Blue: Primary actions, selections
- Green: Success, sync points
- Red: Errors, destructive actions
- Orange: Warnings
- Gray scale: UI chrome

### Accessibility
- Labeled inputs
- Hover states
- Disabled states
- Clear visual hierarchy
- Monospace for code/logs/times

## Future Enhancements (Not Implemented)

Potential additions:
- [ ] Drag & drop file selection
- [ ] Progress bar with ETA
- [ ] Multiple output queue
- [ ] Preset parameter profiles
- [ ] Side-by-side video comparison
- [ ] Waveform audio visualization
- [ ] Auto-update functionality
- [ ] Crash reporting integration
- [ ] Recent files menu
- [ ] Keyboard shortcuts
- [ ] Multi-language support

## Testing Checklist

Before distribution:
- [ ] Install dependencies on clean system
- [ ] Build PyInstaller bundle
- [ ] Download all binaries
- [ ] Test file selection dialogs
- [ ] Test video metadata extraction
- [ ] Test frame preview loading
- [ ] Test sync point creation/editing
- [ ] Test parameter changes
- [ ] Test full AVSync run
- [ ] Test log export
- [ ] Test process cancellation
- [ ] Build Windows installer
- [ ] Build macOS DMG
- [ ] Test installed app (Windows)
- [ ] Test installed app (macOS)

## Known Limitations

1. **Frame Rate**: Currently hardcoded to 24 fps for frame stepping (could extract from metadata)
2. **Binary Paths**: Assumes binaries in resources/bin (works for packaged app)
3. **Platform-Specific**: File paths use platform-specific separators (handled by Node.js)
4. **No Undo**: Parameter/sync point changes are immediate (could add history)
5. **Single Instance**: No queue for multiple jobs (one at a time)

## Performance Notes

- Frame extraction is on-demand (not pre-cached)
- Large videos may take time for initial metadata
- Log viewer auto-scrolls (may slow with 10,000+ lines)
- PyInstaller bundle is ~200MB (includes OpenCV, NumPy, SciPy)
- Full app installer is ~300-400MB (includes FFmpeg)

## Credits & License

- **AVSync Algorithm**: Original Python implementation
- **UI Framework**: Electron (MIT), React (MIT)
- **Build Tools**: Vite (MIT), electron-builder (MIT)
- **Dependencies**: See package.json and requirements.txt

**License**: MIT (suggested)

---

Built with ❤️ for seamless audio/video synchronization
