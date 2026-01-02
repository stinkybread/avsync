# AVSync Desktop - Complete Build Instructions

This guide will walk you through building the AVSync Desktop application from scratch.

## Step 1: Prepare Your Environment

### Install Required Software

1. **Node.js and npm**
   - Download from https://nodejs.org/ (LTS version recommended)
   - Verify installation: `node --version` and `npm --version`

2. **Python 3.8+**
   - Download from https://www.python.org/downloads/
   - Verify installation: `python --version`

3. **Git** (optional, for version control)
   - Download from https://git-scm.com/

## Step 2: Install Dependencies

### Install Node.js Dependencies

```bash
cd AVSync
npm install
```

This will install all required packages including:
- Electron
- React and React DOM
- TypeScript
- Vite
- electron-builder
- And all development dependencies

### Install Python Dependencies

Create a `requirements.txt` file in the project root:

```txt
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.3
tqdm==4.66.1
```

Then install:

```bash
pip install -r requirements.txt
pip install pyinstaller
```

## Step 3: Build the AVSync Executable

### Build with PyInstaller

```bash
# Clean any previous builds
pyinstaller --clean avsync.spec

# Or use the npm script
npm run build:pyinstaller
```

This creates the AVSync executable in `dist/avsync/` directory.

### Verify the Build

Test the executable:

```bash
# Windows
.\dist\avsync\avsync.exe --help

# macOS/Linux
./dist/avsync/avsync --help
```

You should see the AVSync help message with all available parameters.

## Step 4: Download and Prepare Binaries

### Create Resources Directory

```bash
mkdir -p resources/bin
```

### Download FFmpeg and FFprobe

#### Windows:

1. Go to https://www.gyan.dev/ffmpeg/builds/
2. Download "ffmpeg-release-essentials.zip"
3. Extract the archive
4. Copy from `ffmpeg-x.x.x-essentials_build/bin/`:
   - `ffmpeg.exe` → `resources/bin/ffmpeg.exe`
   - `ffprobe.exe` → `resources/bin/ffprobe.exe`

#### macOS:

```bash
# Install via Homebrew
brew install ffmpeg

# Copy binaries
cp $(which ffmpeg) resources/bin/
cp $(which ffprobe) resources/bin/
```

#### Linux:

```bash
# Install via package manager
sudo apt install ffmpeg  # Debian/Ubuntu
# or
sudo yum install ffmpeg  # RHEL/CentOS

# Copy binaries
cp $(which ffmpeg) resources/bin/
cp $(which ffprobe) resources/bin/
```

### Download MKVToolNix

#### Windows:

1. Go to https://mkvtoolnix.download/downloads.html
2. Download the portable version (ZIP)
3. Extract and copy `mkvmerge.exe` → `resources/bin/mkvmerge.exe`

#### macOS:

```bash
brew install mkvtoolnix
cp $(which mkvmerge) resources/bin/
```

#### Linux:

```bash
sudo apt install mkvtoolnix  # Debian/Ubuntu
cp $(which mkvmerge) resources/bin/
```

### Copy AVSync Executable

```bash
# Windows
copy dist\avsync\avsync.exe resources\bin\avsync.exe

# macOS/Linux
cp dist/avsync/avsync resources/bin/avsync
chmod +x resources/bin/avsync
```

### Verify All Binaries

Your `resources/bin/` directory should now contain:

```
resources/bin/
├── avsync.exe (or avsync on Unix)
├── ffmpeg.exe (or ffmpeg on Unix)
├── ffprobe.exe (or ffprobe on Unix)
└── mkvmerge.exe (or mkvmerge on Unix)
```

## Step 5: Build the Electron Application

### Development Build (for testing)

```bash
# Build TypeScript files
npm run build

# This compiles:
# - React app → dist/
# - Electron main process → dist-electron/
```

### Test in Development Mode

```bash
npm run dev
```

This launches the app in development mode with hot-reload enabled.

## Step 6: Package for Distribution

### Windows

```bash
npm run package:win
```

**Output:**
- `release/AVSync Desktop Setup x.x.x.exe` - NSIS installer
- `release/AVSync Desktop x.x.x.exe` - Portable executable

**What's Included:**
- Electron app with React UI
- All resources including binaries
- Windows-specific configurations

### macOS

```bash
npm run package:mac
```

**Output:**
- `release/AVSync Desktop-x.x.x.dmg` - DMG installer
- `release/AVSync Desktop-x.x.x-mac.zip` - ZIP archive

**What's Included:**
- Electron app with React UI
- All resources including binaries
- Code signing placeholder (add your certificate for distribution)

### Build for Both Platforms

If you want to build for both platforms (requires being on each respective OS):

```bash
npm run package
```

## Step 7: Testing the Built Application

### Windows

1. Run the installer from `release/` folder
2. Install the application
3. Launch "AVSync Desktop" from Start Menu
4. Test with sample video files

### macOS

1. Mount the DMG from `release/` folder
2. Drag "AVSync Desktop" to Applications
3. Open the app (you may need to right-click → Open on first launch)
4. Test with sample video files

### Testing Checklist

- [ ] Application launches without errors
- [ ] File selection dialogs work
- [ ] Video metadata loads correctly
- [ ] Frame extraction and preview works
- [ ] Sync point creation works
- [ ] Parameter adjustments are reflected
- [ ] AVSync process runs and shows logs
- [ ] Output file is created successfully
- [ ] Application can be closed cleanly

## Step 8: Code Signing (Optional, for Distribution)

### Windows

For production distribution, you should sign your application:

1. Obtain a code signing certificate
2. Add to `package.json`:

```json
"build": {
  "win": {
    "certificateFile": "path/to/certificate.pfx",
    "certificatePassword": "your-password"
  }
}
```

### macOS

For App Store or notarized distribution:

1. Enroll in Apple Developer Program
2. Create signing certificates
3. Add to `package.json`:

```json
"build": {
  "mac": {
    "identity": "Developer ID Application: Your Name (TEAM_ID)"
  }
}
```

## Troubleshooting

### Build Fails with "Cannot find module"

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
```

### PyInstaller Build Fails

```bash
# Clean build
pyinstaller --clean --noconfirm avsync.spec
```

### Binaries Not Found at Runtime

1. Check `resources/bin/` contains all required files
2. Verify paths in `electron/main.ts` → `getResourcePath()` function
3. Rebuild: `npm run build && npm run package`

### FFmpeg Permission Denied (macOS/Linux)

```bash
chmod +x resources/bin/ffmpeg
chmod +x resources/bin/ffprobe
chmod +x resources/bin/mkvmerge
chmod +x resources/bin/avsync
```

### App Won't Open on macOS (Security Warning)

Right-click the app → Open (instead of double-clicking)

Or disable Gatekeeper temporarily:
```bash
sudo spctl --master-disable
```

## Advanced: Continuous Integration

For automated builds, you can use GitHub Actions:

```yaml
# .github/workflows/build.yml
name: Build

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: npm ci
      - run: pip install -r requirements.txt
      - run: npm run build:pyinstaller
      - run: npm run package:win
      - uses: actions/upload-artifact@v3
        with:
          name: windows-build
          path: release/*.exe

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: npm ci
      - run: pip install -r requirements.txt
      - run: npm run build:pyinstaller
      - run: npm run package:mac
      - uses: actions/upload-artifact@v3
        with:
          name: macos-build
          path: release/*.dmg
```

## Next Steps

1. Test the application thoroughly
2. Consider adding auto-update functionality
3. Set up crash reporting (e.g., Sentry)
4. Create user documentation
5. Set up distribution (website, app stores, etc.)

## Support

For issues or questions:
- Check existing issues in the GitHub repository
- Create a new issue with detailed information
- Include logs from the application
