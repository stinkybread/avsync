# AVSync Desktop - Setup Checklist

Use this checklist to complete the setup and build your AVSync Desktop application.

## ✅ Completed Steps

- [x] Project structure created (35 files)
- [x] All React components implemented
- [x] Electron main process configured
- [x] TypeScript compilation successful
- [x] Build system configured (Vite + electron-builder)
- [x] Documentation written (README, guides, etc.)
- [x] Requirements file updated with >= versions

## 📋 Next Steps to Complete

### 1. Install Dependencies

```bash
# Install Node.js packages
npm install

# Install Python packages
pip install -r requirements.txt
```

**Status:** ⏳ To do
**Estimated time:** 5-10 minutes

---

### 2. Build AVSync Python Executable

```bash
# Using PyInstaller
npm run build:pyinstaller

# Or manually:
pyinstaller --clean avsync.spec
```

**Status:** ⏳ To do
**What this does:** Bundles AVSync_v12.py into a standalone executable
**Output location:** `dist/avsync/avsync.exe` (Windows) or `dist/avsync/avsync` (Unix)
**Estimated time:** 2-5 minutes

**Troubleshooting:**
- If it fails, ensure all Python dependencies are installed
- Check that AVSync_v12.py is in the current directory
- Try: `pip install --upgrade pyinstaller`

---

### 3. Download Required Binaries

**📖 See detailed instructions in:** `DOWNLOAD_BINARIES.md`

#### Windows (Quick Summary):

**FFmpeg & FFprobe:**
1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract and copy from `ffmpeg-x.x.x-essentials_build/bin/`:
   - `ffmpeg.exe` → `resources/bin/ffmpeg.exe`
   - `ffprobe.exe` → `resources/bin/ffprobe.exe`

**MKVToolNix:**
1. Download: https://mkvtoolnix.download/downloads.html (Portable version)
2. Extract and copy: `mkvmerge.exe` → `resources/bin/mkvmerge.exe`

#### macOS:

```bash
# Install via Homebrew
brew install ffmpeg mkvtoolnix

# Copy binaries
cp $(which ffmpeg) resources/bin/
cp $(which ffprobe) resources/bin/
cp $(which mkvmerge) resources/bin/
```

#### Linux:

```bash
# Install via package manager
sudo apt install ffmpeg mkvtoolnix  # Debian/Ubuntu

# Copy binaries
cp $(which ffmpeg) resources/bin/
cp $(which ffprobe) resources/bin/
cp $(which mkvmerge) resources/bin/
```

**Status:** ⏳ To do
**Estimated time:** 5-10 minutes

---

### 4. Copy AVSync Bundle to Resources

PyInstaller creates a folder with the exe and dependencies. Copy the entire folder:

```bash
# Windows (PowerShell)
Copy-Item -Path "dist\avsync" -Destination "resources\avsync" -Recurse -Force

# macOS/Linux
cp -r dist/avsync resources/avsync
chmod +x resources/avsync/avsync
```

**Why the whole folder?** PyInstaller bundles Python DLLs and dependencies in an `_internal` subfolder. The executable needs this folder structure to run.

**Status:** ⏳ To do
**Estimated time:** 1 minute

---

### 5. Verify All Binaries

Check that you have the correct structure:

```bash
# List PyInstaller bundle
ls -la resources/avsync/
# Should show: avsync.exe (or avsync) and _internal/ folder

# List other binaries
ls -la resources/bin/
# Should show: ffmpeg, ffprobe, mkvmerge (with .exe on Windows)
```

**Windows checklist:**
- [ ] `resources/avsync/avsync.exe` exists
- [ ] `resources/avsync/_internal/` folder exists
- [ ] `resources/bin/ffmpeg.exe` exists
- [ ] `resources/bin/ffprobe.exe` exists
- [ ] `resources/bin/mkvmerge.exe` exists

**Unix checklist:**
- [ ] `resources/avsync/avsync` exists and is executable
- [ ] `resources/avsync/_internal/` folder exists
- [ ] `resources/bin/ffmpeg` exists and is executable
- [ ] `resources/bin/ffprobe` exists and is executable
- [ ] `resources/bin/mkvmerge` exists and is executable

**Status:** ⏳ To do

---

### 6. Test in Development Mode

```bash
npm run dev
```

**What happens:**
1. Vite dev server starts on http://localhost:5173
2. Electron window opens automatically
3. React app loads with hot-reload enabled
4. DevTools are available for debugging

**Test the following:**
- [ ] Application window opens
- [ ] File selection dialogs work
- [ ] Can select reference and foreign videos
- [ ] Video metadata displays correctly
- [ ] Frame preview loads (may take a moment)
- [ ] Navigation controls work
- [ ] Can add sync points
- [ ] Parameters can be adjusted
- [ ] Application closes cleanly

**Status:** ⏳ To do
**Estimated time:** 10-15 minutes

---

### 7. Run a Test Sync (Optional but Recommended)

Before building for distribution, test the full AVSync process:

1. Select two test videos (reference and foreign)
2. Choose an output location
3. Optionally add manual sync points
4. Click "Run AVSync"
5. Monitor the log output
6. Wait for completion
7. Verify output file exists and plays correctly

**Status:** ⏳ To do (optional)
**Estimated time:** 5-30 minutes (depending on video length)

---

### 8. Build for Distribution

#### Windows:

```bash
npm run package:win
```

**Output:**
- `release/AVSync Desktop Setup 1.0.0.exe` - NSIS installer
- `release/AVSync Desktop 1.0.0.exe` - Portable executable

#### macOS:

```bash
npm run package:mac
```

**Output:**
- `release/AVSync Desktop-1.0.0.dmg` - DMG installer
- `release/AVSync Desktop-1.0.0-mac.zip` - ZIP archive

**Status:** ⏳ To do
**Estimated time:** 5-10 minutes
**Note:** This will bundle everything including all binaries

---

### 9. Test the Built Application

#### Windows:
1. Navigate to `release/` folder
2. Run the portable .exe OR install using the Setup.exe
3. Launch "AVSync Desktop"
4. Test all features again (like step 6)
5. Run a full sync operation

#### macOS:
1. Mount the DMG
2. Drag to Applications folder
3. Right-click → Open (first launch only)
4. Test all features
5. Run a full sync operation

**Checklist:**
- [ ] Application installs/launches successfully
- [ ] All UI elements render correctly
- [ ] File dialogs work
- [ ] Video preview works
- [ ] Full sync operation completes successfully
- [ ] Output video is valid
- [ ] Application closes without errors

**Status:** ⏳ To do
**Estimated time:** 15-20 minutes

---

## 🎯 Quick Start Alternative

If you want to get started quickly, run the automated setup script:

**Windows:**
```bash
setup.bat
```

**Unix (macOS/Linux):**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install npm dependencies
- Install Python dependencies
- Build AVSync executable
- Create resources directory
- Copy AVSync to resources (Windows)
- Attempt to copy system binaries (Unix)

You'll still need to manually download binaries on Windows.

---

## 📝 Notes

### File Sizes

Expect these approximate sizes:

- `node_modules/`: ~200 MB
- `dist/avsync/`: ~150-200 MB (PyInstaller bundle)
- `resources/bin/`: ~150 MB (all binaries)
- Final installer: ~300-400 MB

### Build Time Estimates

- First npm install: 5-10 minutes
- PyInstaller build: 2-5 minutes
- Vite build: <1 minute
- electron-builder package: 5-10 minutes

**Total setup time:** 30-60 minutes (including downloads)

### Common Issues

**"Module not found" during npm install:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**PyInstaller fails:**
```bash
pip install --upgrade pyinstaller
pyinstaller --clean avsync.spec
```

**Binaries not found at runtime:**
- Check all files are in `resources/bin/`
- On Unix: `chmod +x resources/bin/*`
- Rebuild: `npm run build && npm run package`

**App won't open on macOS:**
- Right-click → Open (don't double-click)
- Or: System Preferences → Security & Privacy → Allow

---

## ✨ Success!

When you complete all steps, you'll have:

✅ A fully functional desktop application
✅ Production-ready installers
✅ Cross-platform compatibility
✅ Professional UI with all AVSync features

**What to do with the built application:**
- Share the installer with users
- Upload to a distribution platform
- Keep for personal use
- Continue development

**Next enhancements you could add:**
- Drag & drop file support
- Recent files list
- Parameter presets/profiles
- Auto-update functionality
- Crash reporting
- More detailed progress indicators

---

## 📚 Documentation Reference

- **README.md** - Feature overview and usage
- **QUICKSTART.md** - Fastest way to get started
- **BUILD_INSTRUCTIONS.md** - Detailed build guide
- **PROJECT_SUMMARY.md** - Architecture details
- **FILE_STRUCTURE.md** - Complete file tree

---

**Happy building!** 🚀

For questions or issues, refer to the documentation or check the AVSync_v12.py source code for parameter details.
