# AVSync Desktop - Complete File Structure

This document shows the complete file structure of the AVSync Desktop project.

## Directory Tree

```
AVSync/
│
├── 📄 Configuration Files
│   ├── package.json                 # Node.js dependencies and npm scripts
│   ├── tsconfig.json                # TypeScript config for React/Vite
│   ├── tsconfig.node.json           # TypeScript config for Vite config file
│   ├── tsconfig.electron.json       # TypeScript config for Electron main process
│   ├── vite.config.ts               # Vite bundler configuration
│   ├── .eslintrc.cjs                # ESLint configuration
│   ├── .gitignore                   # Git ignore patterns
│   ├── requirements.txt             # Python dependencies for PyInstaller
│   └── avsync.spec                  # PyInstaller specification file
│
├── 📄 Documentation
│   ├── README.md                    # Main project documentation
│   ├── QUICKSTART.md                # 5-minute setup guide
│   ├── BUILD_INSTRUCTIONS.md        # Detailed build instructions
│   ├── PROJECT_SUMMARY.md           # Complete project overview
│   └── FILE_STRUCTURE.md            # This file
│
├── 📄 Setup Scripts
│   ├── setup.bat                    # Windows automated setup script
│   └── setup.sh                     # Unix (macOS/Linux) automated setup script
│
├── 📄 Source Files
│   ├── AVSync_v12.py                # Original Python AVSync implementation
│   └── index.html                   # HTML entry point for Electron renderer
│
├── 📁 electron/                     # Electron main process (Node.js)
│   ├── main.ts                      # Main process entry point
│   │                                # - Window management
│   │                                # - IPC handlers (file dialogs, ffmpeg, avsync)
│   │                                # - Resource path resolution
│   │                                # - Process lifecycle
│   │
│   └── preload.ts                   # Preload script for secure IPC
│                                    # - Context bridge API
│                                    # - Type-safe IPC methods
│                                    # - Event listeners
│
├── 📁 src/                          # React application (renderer process)
│   ├── main.tsx                     # React entry point
│   ├── App.tsx                      # Main application component
│   ├── App.css                      # App layout styles
│   ├── index.css                    # Global styles and theme
│   ├── vite-env.d.ts                # TypeScript ambient declarations
│   │
│   └── 📁 components/               # React UI components
│       │
│       ├── Header.tsx               # Top navigation bar
│       ├── Header.css               # - App title
│       │                            # - Run/Stop button
│       │                            # - Processing state
│       │
│       ├── FileSelector.tsx         # File input controls
│       ├── FileSelector.css         # - Reference video selection
│       │                            # - Foreign video selection
│       │                            # - Output path selection
│       │                            # - Metadata display
│       │
│       ├── VideoPreview.tsx         # Video frame viewer
│       ├── VideoPreview.css         # - Dual video display
│       │                            # - Frame extraction
│       │                            # - Navigation controls
│       │                            # - Timeline scrubbing
│       │                            # - Sync point creation
│       │
│       ├── SyncPointEditor.tsx      # Sync point management
│       ├── SyncPointEditor.css      # - Sync point table
│       │                            # - Add/remove/edit points
│       │                            # - Time formatting
│       │                            # - Clear all functionality
│       │
│       ├── ParametersPanel.tsx      # AVSync parameter controls
│       ├── ParametersPanel.css      # - Collapsible sections
│       │                            # - Image pairing params
│       │                            # - Audio processing params
│       │                            # - Muxing options
│       │                            # - Advanced settings
│       │
│       ├── LogViewer.tsx            # Processing log display
│       └── LogViewer.css            # - Real-time log streaming
│                                    # - Auto-scrolling
│                                    # - Color-coded output
│                                    # - Export functionality
│
├── 📁 resources/                    # Application resources
│   └── 📁 bin/                      # Binary executables (to be populated)
│       ├── .gitkeep                 # Placeholder to track directory
│       │
│       ├── avsync.exe               # ⚠️ TO BE ADDED: PyInstaller bundle
│       │   (or avsync on Unix)      # Built with: npm run build:pyinstaller
│       │
│       ├── ffmpeg.exe               # ⚠️ TO BE ADDED: Video processing
│       │   (or ffmpeg on Unix)      # Download from: https://ffmpeg.org/
│       │
│       ├── ffprobe.exe              # ⚠️ TO BE ADDED: Metadata extraction
│       │   (or ffprobe on Unix)     # Included with FFmpeg
│       │
│       └── mkvmerge.exe             # ⚠️ TO BE ADDED: Final muxing
│           (or mkvmerge on Unix)    # Download from: https://mkvtoolnix.download/
│
├── 📁 dist/                         # ⚙️ Generated by Vite build
│   └── [React app bundle]           # Created with: npm run build
│
├── 📁 dist-electron/                # ⚙️ Generated by TypeScript
│   ├── main.js                      # Compiled from electron/main.ts
│   └── preload.js                   # Compiled from electron/preload.ts
│
├── 📁 node_modules/                 # ⚙️ Generated by npm install
│   └── [all npm packages]           # Dependencies from package.json
│
└── 📁 release/                      # ⚙️ Generated by electron-builder
    ├── AVSync Desktop Setup 1.0.0.exe       # Windows NSIS installer
    ├── AVSync Desktop 1.0.0.exe             # Windows portable
    ├── AVSync Desktop-1.0.0.dmg             # macOS disk image
    └── AVSync Desktop-1.0.0-mac.zip         # macOS ZIP archive
```

## File Counts by Category

### Source Code (TypeScript/React)
- Electron: 2 files (main.ts, preload.ts)
- React: 14 files (components + app + styles)
- Config: 5 TypeScript config files

### Documentation
- 5 markdown files (README, guides, summaries)

### Configuration
- 6 config files (package.json, eslint, git, vite)

### Python
- 1 source file (AVSync_v12.py)
- 1 spec file (avsync.spec)
- 1 requirements file

### Total
- **35 files** created for the project structure
- **4 binaries** to be added to resources/bin/

## Component Dependency Graph

```
App.tsx
  ├─→ Header.tsx
  ├─→ FileSelector.tsx
  ├─→ VideoPreview.tsx
  │     └─→ SyncPointEditor.tsx
  └─→ ParametersPanel.tsx
        └─→ LogViewer.tsx
```

## Build Output Sizes (Approximate)

### Development
- `node_modules/`: ~200 MB
- `src/`: ~50 KB (uncompiled)

### Production Build
- `dist/`: ~500 KB (bundled React app)
- `dist-electron/`: ~10 KB (compiled main process)
- PyInstaller `dist/avsync/`: ~150-200 MB (Python + dependencies)
- `resources/bin/`: ~100-150 MB (ffmpeg, mkvmerge, avsync)

### Packaged Application
- Windows installer: ~300-400 MB
- macOS DMG: ~300-400 MB
- Includes: Electron, Chrome runtime, Node.js, all binaries

## Key File Purposes

### Configuration Files

**package.json**
- Defines all npm dependencies
- Contains build scripts
- Configures electron-builder for packaging

**tsconfig.json**
- TypeScript settings for React/Vite
- Module resolution
- Strict type checking

**tsconfig.electron.json**
- Separate TS config for main process
- CommonJS modules for Node.js compatibility

**vite.config.ts**
- Development server settings
- Build optimizations
- Path aliases

**avsync.spec**
- PyInstaller bundle configuration
- Hidden imports
- Binary exclusions
- Output settings

### Source Files

**electron/main.ts** (400+ lines)
- Window creation and lifecycle
- IPC handler implementations
- File system operations
- Child process spawning
- Resource path resolution

**electron/preload.ts** (100+ lines)
- Secure IPC bridge
- Type-safe API exposure
- Event listener setup

**src/App.tsx** (150+ lines)
- Main application layout
- State management
- IPC event handling
- Component orchestration

**src/components/*** (200-400 lines each)
- Self-contained UI components
- Local state management
- Event handling
- Prop-based communication

### Documentation

**README.md**
- Project overview
- Feature list
- Installation guide
- Usage instructions

**QUICKSTART.md**
- Streamlined setup steps
- Common commands
- Quick troubleshooting

**BUILD_INSTRUCTIONS.md**
- Detailed build process
- Platform-specific steps
- Binary acquisition
- Packaging instructions

**PROJECT_SUMMARY.md**
- Architecture overview
- Technology stack
- Design decisions
- Feature breakdown

## File Size Guidelines

### Keep Small (<50 lines)
- CSS files (component-specific)
- TypeScript config files
- Simple utilities

### Medium Size (50-200 lines)
- Simple React components
- Preload script
- App.tsx

### Larger Files (200-500 lines)
- Complex components (VideoPreview, ParametersPanel)
- Main process (main.ts)
- Documentation files

### Very Large Files (>1000 lines)
- AVSync_v12.py (original implementation)

## Customization Points

To customize the app, edit these files:

**Branding:**
- `src/components/Header.tsx` - App title
- `package.json` - App name, version, author
- `resources/icon.ico` - Windows icon (to be added)
- `resources/icon.icns` - macOS icon (to be added)

**Styling:**
- `src/index.css` - Global theme colors
- Component-specific `.css` files

**Functionality:**
- `electron/main.ts` - Add new IPC handlers
- `src/App.tsx` - Add new top-level features
- `src/components/*` - Modify UI behavior

**Build:**
- `package.json` → `build` section - Package settings
- `avsync.spec` - PyInstaller options
- `vite.config.ts` - Build optimizations

## Next Steps After File Creation

1. ✅ All files created
2. ⏳ Run setup script: `setup.bat` or `setup.sh`
3. ⏳ Download binaries to `resources/bin/`
4. ⏳ Test in dev mode: `npm run dev`
5. ⏳ Build for production: `npm run package:win` or `npm run package:mac`

---

Last updated: 2026-01-01
Version: 1.0.0
