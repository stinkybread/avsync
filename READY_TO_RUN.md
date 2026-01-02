# ✅ AVSync Desktop - Ready to Run!

## Current Status

Your AVSync Desktop app is **ready to use**! All required binaries are in place.

### Verified Structure:

```
✅ resources/avsync/
   ├── avsync.exe          (PyInstaller executable)
   └── _internal/          (Python runtime & dependencies)

✅ resources/bin/
   ├── ffmpeg.exe          (Video processing)
   ├── ffprobe.exe         (Metadata extraction)
   ├── mkvmerge.exe        (Video muxing)
   └── mkvextract.exe      (Bonus: MKV extraction tool)
```

---

## How to Run

### Development Mode (with hot-reload):

```powershell
npm run dev
```

**What happens:**
- Vite dev server starts on http://localhost:5173
- Electron window opens automatically
- Changes to React code reload instantly
- DevTools available for debugging

### Test the Full Workflow:

1. **Load Videos:**
   - Click "Browse" for Reference Video
   - Click "Browse" for Foreign Video
   - Click "Browse" for Output File location

2. **Preview Frames (Optional):**
   - Use the timeline sliders to navigate
   - Use frame buttons for precise control
   - Add sync points when both videos show matching moments

3. **Adjust Parameters (Optional):**
   - Expand sections in the right panel
   - Modify thresholds, language codes, etc.
   - Or just use the defaults

4. **Run Synchronization:**
   - Click the "Run AVSync" button
   - Watch the logs in real-time
   - Wait for completion (can take several minutes for long videos)

5. **Check Output:**
   - Find your synced video at the output location
   - Play it to verify synchronization

---

## What Just Got Fixed

### The Python DLL Error:

**Problem:** You were getting:
```
Failed to load Python DLL 'python313.dll'
LoadLibrary: The specified module could not be found.
```

**Root Cause:** Only `avsync.exe` was copied, but PyInstaller bundles Python DLLs and dependencies in an `_internal/` subfolder. The exe can't run without that folder.

**Solution:** Changed the setup to copy the entire `dist/avsync/` folder to `resources/avsync/`, preserving the folder structure.

### What Changed:

1. **electron/main.ts:**
   - Updated `getBinaryPath()` to handle AVSync in `resources/avsync/`
   - Other binaries (ffmpeg, ffprobe, mkvmerge) stay in `resources/bin/`

2. **package.json:**
   - Added `resources/avsync` to electron-builder's extraResources
   - This ensures the entire PyInstaller bundle gets packaged

3. **Setup scripts:**
   - `setup.bat` and `setup.sh` now copy the entire folder
   - Documentation updated to reflect correct approach

---

## Building for Distribution

When you're ready to create installers:

### Windows:

```powershell
npm run package:win
```

**Output:** `release/AVSync Desktop Setup 1.0.0.exe` (installer) and portable .exe

### macOS:

```bash
npm run package:mac
```

**Output:** `release/AVSync Desktop-1.0.0.dmg` (installer) and .zip

---

## File Sizes

Your current setup:

- `resources/avsync/`: ~200 MB (Python runtime + OpenCV + NumPy + SciPy)
- `resources/bin/`: ~200 MB (FFmpeg + MKVToolNix)
- **Total resources:** ~400 MB

Final packaged app will be ~500-600 MB (includes Electron + Chrome runtime).

---

## Tips for Using the App

### Frame Navigation:
- **-10s / +10s** buttons: Quick navigation
- **-1s / +1s** buttons: Fine navigation
- **< Frame / Frame >** buttons: Frame-by-frame (24 fps assumed)
- **Timeline slider**: Scrub through entire video

### Manual Sync Points:
- Navigate both videos to a matching moment (same frame/scene)
- Click "Add Sync Point at Current Positions"
- Repeat for 2-3 points (start, middle, end recommended)
- Or leave empty for fully automatic synchronization

### Parameters:
- **Image Pairing:** Controls how frames are matched
  - Lower scene_threshold = more scene changes detected
  - Higher match_threshold = stricter frame matching

- **Audio Processing:** Controls audio synchronization
  - db_threshold: Silence detection level (-40 is good default)
  - Language codes: Use ISO 639-2 (eng, spa, fra, etc.)

- **Muxing Options:** Output encoding
  - AAC recommended for compatibility
  - Use "copy" to avoid re-encoding (faster, larger file)

### Processing Time:
- Expect 5-30 minutes for typical movies
- Longer videos = more time
- Using cache speeds up re-runs

---

## Troubleshooting

### App won't start:
```powershell
# Rebuild
npm run build
npm run dev
```

### "Binary not found" errors:
```powershell
# Verify all files exist
ls resources/avsync
ls resources/bin
```

### Frame preview not loading:
- Check that ffmpeg.exe and ffprobe.exe are in resources/bin
- Try a different video file
- Check video codec is supported

### AVSync process fails:
- Check the log output for specific errors
- Verify input videos are valid
- Check output path is writable
- Try with --verbose enabled in parameters

---

## Next Steps

1. **Test with your videos** ✅ Ready now!
2. **Experiment with parameters** to optimize results
3. **Build installers** when satisfied: `npm run package:win`
4. **Share** with others or keep for personal use

---

## Performance Notes

### First Run:
- May be slow as cache is built
- Scene detection happens once per video
- Frame extraction is on-demand

### Subsequent Runs:
- Much faster with cache enabled
- Changing only audio params reuses image cache
- Changing only a few params reuses most cache

### RAM Usage:
- Expect 500MB-1GB RAM for the app
- Video processing uses additional RAM
- Large 4K videos may use 2-3GB temporarily

---

## Known Limitations

1. **Frame Rate:** Currently assumes 24 fps for frame stepping
   - Could be enhanced to extract actual fps from metadata

2. **Video Codecs:** Depends on FFmpeg's codec support
   - Most common formats work (MP4, MKV, AVI, MOV)
   - Some proprietary codecs may not work

3. **Single Job:** Can only run one sync operation at a time
   - Close current job before starting another
   - Or restart the app

4. **Large Files:** Very large videos (>10GB) may be slow
   - Consider breaking into segments
   - Or use a faster computer with SSD

---

## Enjoy Your Synced Videos! 🎬

Questions? Check the other documentation:
- `README.md` - Complete overview
- `SETUP_CHECKLIST.md` - Setup steps
- `BUILD_INSTRUCTIONS.md` - Detailed build guide
- `DOWNLOAD_BINARIES.md` - Where to get binaries
- `PROJECT_SUMMARY.md` - Architecture details

**Happy syncing!**
