# Quick Start Guide - AVSync Desktop

Get up and running in 5 minutes!

## Prerequisites

- Node.js 18+ installed
- Python 3.8+ installed
- Git (optional)

## Quick Setup

### 1. Install Dependencies

```bash
# Install Node.js packages
npm install

# Install Python packages
pip install -r requirements.txt
```

### 2. Build AVSync Executable

```bash
npm run build:pyinstaller
```

### 3. Download Binaries

Download and place these files in `resources/bin/`:

**Windows:**
- FFmpeg: https://www.gyan.dev/ffmpeg/builds/ → Get `ffmpeg.exe` and `ffprobe.exe`
- MKVToolNix: https://mkvtoolnix.download/ → Get `mkvmerge.exe`

**macOS:**
```bash
brew install ffmpeg mkvtoolnix
cp $(which ffmpeg) resources/bin/
cp $(which ffprobe) resources/bin/
cp $(which mkvmerge) resources/bin/
```

### 4. Copy AVSync Executable

```bash
# Windows
copy dist\avsync\avsync.exe resources\bin\

# macOS/Linux
cp dist/avsync/avsync resources/bin/
chmod +x resources/bin/*
```

### 5. Run Development Mode

```bash
npm run dev
```

The app will open automatically!

## Build for Distribution

### Windows
```bash
npm run package:win
```
Output: `release/AVSync Desktop Setup x.x.x.exe`

### macOS
```bash
npm run package:mac
```
Output: `release/AVSync Desktop-x.x.x.dmg`

## Using the Application

1. **Select Files**: Click Browse to select your reference video, foreign video, and output path
2. **Preview Frames**: Navigate through both videos to find matching points
3. **Add Sync Points** (optional): Click "Add Sync Point" when both videos show the same moment
4. **Adjust Parameters**: Expand sections in the right panel to customize settings
5. **Run**: Click "Run AVSync" button and monitor the logs
6. **Wait**: Processing may take several minutes depending on video length
7. **Done**: Your synchronized video will be saved to the output path

## Tips

- Use the timeline sliders for quick navigation
- Use frame buttons for precise positioning
- The first and last sync points are most important for accuracy
- Leave sync points empty for fully automatic synchronization
- Enable "Verbose logging" for detailed output
- Check the log file saved alongside your output for troubleshooting

## Common Issues

**"Module not found" error**
```bash
rm -rf node_modules package-lock.json
npm install
```

**PyInstaller fails**
```bash
pip install --upgrade pyinstaller
pyinstaller --clean avsync.spec
```

**Binaries not found**
- Verify all files are in `resources/bin/`
- On Unix: `chmod +x resources/bin/*`

**App won't open (macOS)**
- Right-click → Open (instead of double-click)

## Need Help?

See the full documentation:
- `README.md` - Complete feature documentation
- `BUILD_INSTRUCTIONS.md` - Detailed build guide

Enjoy using AVSync Desktop!
