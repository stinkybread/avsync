# AVSync Desktop - Audio-Video Synchronization Tool

![Electron](https://img.shields.io/badge/Electron-29+-blue.svg)
![React](https://img.shields.io/badge/React-18+-61dafb.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178c6.svg)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful desktop application for automatically synchronizing foreign audio tracks to reference videos. Built with Electron, React, and TypeScript, powered by the AVSync Python engine with advanced visual anchor detection and precise audio timing algorithms.

![AVSync Night](https://github.com/stinkybread/avsync/blob/main/Main_Night.png) ![AvSync Day](https://github.com/stinkybread/avsync/blob/main/Main_Day.png)

## ✨ Features

### Desktop Application
- 🎨 **Modern UI**: Beautiful, intuitive interface with dark/light theme support
- 📹 **Video Preview**: Frame-by-frame navigation with visual sync point definition
- 🔄 **Batch Processing**: Process multiple videos with automatic file matching
- 📊 **Job Queue**: Manage multiple processing jobs with real-time progress tracking
- 💾 **Persistent Settings**: All parameters saved between sessions
- 🎯 **Manual Sync Points**: Define precise synchronization points visually
- 📝 **Real-time Logs**: Monitor processing with live log output

### Processing Engine
- 🎯 **Visual Anchor Detection**: Scene change detection and template matching
- 🔊 **Precise Audio Timing**: Iterative audio processing with millisecond-level precision
- 🌍 **Multi-language Support**: Automatic audio stream detection by language codes
- 📊 **Quality Control**: Generate side-by-side comparison images and CSV reports
- ⚡ **Parallel Processing**: Multi-threaded frame matching for faster processing
- 🎛️ **Flexible Configuration**: Extensive customization options
- 💾 **Smart Caching**: Cache visual anchors for faster re-processing

## 🎬 How It Works

1. **Image Pairing Stage**: Extracts scene change frames and matches them between reference and foreign videos
2. **Audio Synchronization Stage**: Processes audio segments iteratively to match reference timing precisely
3. **Muxing Stage**: Combines reference video, original audio, and synchronized foreign audio into final output

## 📋 Requirements

### System Dependencies
- **Node.js** 18+ and npm (for building the desktop app)
- **Python** 3.8+ with pip
- **FFmpeg** (full build with SoxR support)
- **FFprobe**
- **MKVToolNix** (mkvmerge, mkvextract)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/stinkybread/avsync.git
cd avsync
```

### 2. Install Node Dependencies
```bash
npm install
```

### 3. Setup Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Download External Binaries

Download FFmpeg (full build with SoxR) and MKVToolNix binaries and place them in `resources/bin/`:

**FFmpeg (Recommended):**
- Download from: https://ffbinaries.com/downloads
- Get the latest FFmpeg build for your platform
- Extract and copy `ffmpeg.exe` and `ffprobe.exe` to `resources/bin/`

**MKVToolNix:**
- Download from: https://mkvtoolnix.download/
- Copy `mkvmerge.exe` and `mkvextract.exe` to `resources/bin/`

Your `resources/bin/` folder should contain:
```
resources/bin/
├── ffmpeg.exe
├── ffprobe.exe
├── mkvmerge.exe
└── mkvextract.exe
```

### 5. Build Python Engine
```bash
# Activate venv if not already active
# Build AVSync executable with PyInstaller
python -m PyInstaller avsync.spec -y

# Copy built executable to resources
# Windows PowerShell:
Remove-Item -Path resources\avsync -Recurse -Force
Copy-Item -Path dist\avsync -Destination resources\avsync -Recurse -Force
```

### 6. Build Desktop Application
```bash
npm run build
```

## 🎮 Usage

### Development Mode
```bash
npm run dev
```
This starts the Vite dev server and launches Electron in development mode with hot reload.

### Production Build
```bash
npm run build
npm run electron
```

### Creating Distributable Package
```bash
npm run package
```
This creates a distributable application in the `release` folder.

## 💡 Application Guide

### New Job Tab
- Select reference video (the video with correct timing)
- Select foreign video (the video with audio to sync)
- Choose output location
- Adjust parameters as needed
- Add manual sync points if desired (optional)
- Click "Add to Queue"

### Batch Tab
- Select folders containing reference and foreign videos
- Configure file matching patterns
- Review matched files in the staging table
- Adjust per-job settings if needed
- Add all to queue

### Queue Tab
- View all queued jobs
- Start processing
- Monitor real-time progress and logs
- Abort, retry, or remove jobs as needed

### Parameters

#### Image Pairing
- **Scene Threshold**: Scene change detection sensitivity (0.0-1.0, default: 0.25)
- **Match Threshold**: Template matching threshold (0.0-1.0, default: 0.7)
- **Similarity Threshold**: Perceptual hash difference (default: 4, -1 to disable)

#### Audio Processing
- **Reference Language**: Language code for reference audio (default: eng)
- **Foreign Language**: Language code for foreign audio (default: spa)
- **dB Threshold**: Audio detection threshold (default: -40.0 dB)
- **Min Segment Duration**: Minimum segment length (default: 0.5s)
- **Auto-detect**: Automatically detect audio streams

#### Muxing
- **Foreign Audio Codec**: Output codec (default: aac)
- **Foreign Audio Bitrate**: Output bitrate (default: 192k)

#### Advanced
- **Use Cache**: Cache visual anchors for faster re-processing
- **Skip Subtitles**: Don't include subtitles in output

## 🔧 Project Structure

```
avsync/
├── electron/           # Electron main and preload scripts
├── src/               # React frontend source
│   ├── components/    # React components
│   ├── App.tsx       # Main application component
│   └── main.tsx      # React entry point
├── resources/        # Application resources
│   ├── bin/         # External binaries (FFmpeg, etc.)
│   └── avsync/      # PyInstaller bundle
├── AVSync_v12.py    # Python processing engine
├── avsync.spec      # PyInstaller specification
└── package.json     # Node.js dependencies
```

## 🎯 Tips for Best Results

### Video Content
- ✅ Use videos with clear scene changes and visual landmarks
- ✅ Ensure good video quality for accurate frame matching
- ✅ Both videos should be essentially the same (same scenes, different audio)
- ❌ Avoid videos with different intros, extra ads, or missing scenes

### Audio Content
- ✅ Ensure clear content boundaries in audio tracks
- ✅ Use similar audio quality between reference and foreign tracks
- ✅ Define manual sync points for problematic sections

### Parameter Tuning
- **Lower scene threshold**: Detects more frames (more anchor points)
- **Higher match threshold**: Stricter frame matching (fewer false positives)
- **Longer min segment duration**: Fewer, longer segments (more stable sync)

## 🐛 Troubleshooting

### "FFmpeg/FFprobe not found"
- Ensure binaries are in `resources/bin/`
- Rebuild PyInstaller bundle: `python -m PyInstaller avsync.spec -y`
- Copy to resources: See Installation step 5

### "No matches found"
- Try lowering scene threshold (e.g., 0.15)
- Try lowering match threshold (e.g., 0.6)
- Verify videos actually correspond to each other
- Add manual sync points

### "SoxR resampler unavailable"
- Download FFmpeg **full build** (not essentials)
- Use builds from https://ffbinaries.com/downloads

### Build Errors
```bash
# Clean and rebuild
rm -rf node_modules dist dist-electron build
npm install
npm run build
```

## 🏗️ Development

### Prerequisites
- Node.js 18+
- Python 3.8+
- Git

### Setup Development Environment
```bash
# Clone and install
git clone https://github.com/stinkybread/avsync.git
cd avsync
npm install

# Setup Python
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download binaries (see Installation step 4)

# Build Python engine
python -m PyInstaller avsync.spec -y
# Copy dist/avsync to resources/avsync

# Run in dev mode
npm run dev
```

### Tech Stack
- **Frontend**: React 18, TypeScript, Vite
- **Desktop**: Electron 29
- **Processing**: Python 3.8+, OpenCV, SciPy, NumPy
- **Bundling**: PyInstaller, electron-builder

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

**Developer**: [Vaibhav Bhat](https://github.com/stinkybread)

**UI Design & Implementation**: Claude (Anthropic)

**Special Thanks**:
- FFmpeg team for the multimedia framework
- OpenCV community for computer vision tools
- SciPy contributors for audio processing capabilities
- Electron and React teams

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/stinkybread/avsync/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/stinkybread/avsync/discussions)

---

**Made with ❤️ by Vaibhav Bhat**
