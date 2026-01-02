# Building AVSync Desktop for Distribution

This guide explains how to create a fully installable Windows executable with all dependencies bundled.

## Prerequisites

Before building, ensure you have:

1. **Node.js 18+** and npm installed
2. **Python 3.8+** with pip
3. **All dependencies installed**:
   ```bash
   npm install
   pip install -r requirements.txt
   ```
4. **External binaries** in `resources/bin/`:
   - `ffmpeg.exe`
   - `ffprobe.exe`
   - `mkvmerge.exe`
   - `mkvextract.exe`

## Build Process

### Step 1: Build the Python Engine

First, build the AVSync Python engine using PyInstaller:

```bash
# Activate your Python virtual environment
venv\Scripts\activate

# Build with PyInstaller
python -m PyInstaller avsync.spec -y
```

This creates the bundled Python application in the `dist/avsync/` folder.

### Step 2: Copy Python Bundle to Resources

Copy the built PyInstaller bundle to the resources folder:

```powershell
# Remove old bundle
Remove-Item -Path resources\avsync -Recurse -Force -ErrorAction SilentlyContinue

# Copy new bundle
Copy-Item -Path dist\avsync -Destination resources\avsync -Recurse -Force
```

### Step 3: Build the Electron Application

Build the Electron frontend and backend:

```bash
npm run build
```

This command:
1. Compiles TypeScript files
2. Builds the React frontend with Vite
3. Compiles the Electron main process

### Step 4: Create Distribution Package

Create the installer using electron-builder:

```bash
npm run package:win
```

This creates:
- **NSIS Installer** (`.exe`): Full installer with install wizard
- **Portable Version** (`.exe`): Standalone executable that doesn't require installation

Both will be in the `release/` folder.

## Complete Build Script

For convenience, here's a complete build script (PowerShell):

```powershell
# 1. Activate Python environment
venv\Scripts\activate

# 2. Build Python engine
python -m PyInstaller avsync.spec -y

# 3. Copy to resources
Remove-Item -Path resources\avsync -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -Path dist\avsync -Destination resources\avsync -Recurse -Force

# 4. Build and package Electron app
npm run build
npm run package:win
```

## What Gets Bundled

The final installer includes:

### Application Files
- React frontend (built with Vite)
- Electron runtime
- Node.js runtime

### Python Engine
- `resources/avsync/` - Complete PyInstaller bundle
  - `avsync.exe` - Main processing executable
  - `_internal/` - Python runtime and all dependencies
    - OpenCV, NumPy, SciPy, PIL, imagehash, etc.

### External Binaries
- `resources/bin/` - External tools
  - `ffmpeg.exe` - Media processing
  - `ffprobe.exe` - Media analysis
  - `mkvmerge.exe` - MKV muxing
  - `mkvextract.exe` - MKV extraction

## Output Files

After building, you'll find in the `release/` folder:

```
release/
├── AVSync Desktop Setup 1.0.0.exe    # NSIS installer (recommended for distribution)
└── AVSync Desktop 1.0.0.exe          # Portable version (no installation required)
```

### NSIS Installer
- Full installer with install wizard
- Allows user to choose installation directory
- Creates Start Menu shortcuts
- Includes uninstaller
- File size: ~300-400 MB

### Portable Version
- Single executable
- No installation required
- Can run from any location (USB drive, etc.)
- File size: ~300-400 MB

## Testing the Build

Before distribution, test both versions:

### Test NSIS Installer
1. Run `AVSync Desktop Setup 1.0.0.exe`
2. Follow installation wizard
3. Launch from Start Menu
4. Test with sample videos

### Test Portable Version
1. Copy `AVSync Desktop 1.0.0.exe` to a different location
2. Run directly
3. Test with sample videos

## Troubleshooting

### "Application failed to start"
- Ensure all binaries are in `resources/bin/` before building
- Rebuild PyInstaller bundle: `python -m PyInstaller avsync.spec -y`
- Copy fresh bundle to resources

### "FFmpeg not found" in production
- Check that `resources/bin/` contains all 4 binaries
- Ensure binaries were included in build (check file size - should be 300+ MB)
- Rebuild with `npm run package:win`

### Large file size
- This is normal! The bundle includes:
  - Electron runtime (~100 MB)
  - Python runtime with OpenCV (~100 MB)
  - FFmpeg full build with SoxR (~130 MB)
  - MKVToolNix binaries (~40 MB)

### Build fails with "Cannot find module"
- Run `npm install` again
- Ensure `node_modules/` is present
- Check that `dist-electron/` was created by `npm run build`

## Distribution

### For Windows Users
Distribute the **NSIS Installer** (`AVSync Desktop Setup 1.0.0.exe`) for:
- Easy installation
- Automatic shortcuts
- Professional appearance
- Built-in uninstaller

### For Advanced Users
Distribute the **Portable Version** (`AVSync Desktop 1.0.0.exe`) for:
- No installation required
- Run from USB/external drives
- No admin rights needed
- Quick testing

## Updating the Version

To change the version number:

1. Update `package.json`:
   ```json
   {
     "version": "1.0.1"
   }
   ```

2. Rebuild and package:
   ```bash
   npm run package:win
   ```

The installer filename will automatically update to reflect the new version.

## Code Signing (Optional)

For production distribution, consider code signing:

1. Obtain a code signing certificate
2. Update `package.json`:
   ```json
   {
     "build": {
       "win": {
         "certificateFile": "path/to/cert.pfx",
         "certificatePassword": "password"
       }
     }
   }
   ```

3. Rebuild: `npm run package:win`

Code signing prevents Windows SmartScreen warnings.

## CI/CD Integration

For automated builds, use GitHub Actions or similar:

```yaml
# .github/workflows/build.yml
name: Build
on: [push]
jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - uses: actions/setup-python@v2
      - run: npm install
      - run: pip install -r requirements.txt
      - run: python -m PyInstaller avsync.spec -y
      - run: Copy-Item -Path dist\avsync -Destination resources\avsync -Recurse
      - run: npm run package:win
      - uses: actions/upload-artifact@v2
        with:
          name: installers
          path: release/*.exe
```

## Summary

**Quick Build Command:**
```bash
# Complete build in one go
venv\Scripts\activate && python -m PyInstaller avsync.spec -y && Remove-Item -Path resources\avsync -Recurse -Force -ErrorAction SilentlyContinue && Copy-Item -Path dist\avsync -Destination resources\avsync -Recurse -Force && npm run build && npm run package:win
```

**Output:** Ready-to-distribute installer in `release/` folder!
