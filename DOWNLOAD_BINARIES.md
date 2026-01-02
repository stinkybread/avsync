# Download Required Binaries - Quick Guide

You need to download 3 programs and place them in `resources/bin/`:

## 1. FFmpeg & FFprobe (Video Processing)

### Option A: Direct Download (Recommended)

**Download Link:** https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

**Steps:**
1. Click the link above to download `ffmpeg-release-essentials.zip` (~90 MB)
2. Extract the ZIP file
3. Navigate into the extracted folder: `ffmpeg-x.x.x-essentials_build/bin/`
4. Copy **two files**:
   - `ffmpeg.exe` → Copy to `C:\Users\Vaibhav\AVSync\resources\bin\ffmpeg.exe`
   - `ffprobe.exe` → Copy to `C:\Users\Vaibhav\AVSync\resources\bin\ffprobe.exe`

### Option B: Using Chocolatey (if you have it)

```powershell
choco install ffmpeg
# Then find the binaries and copy them to resources/bin/
```

---

## 2. MKVToolNix (mkvmerge - Video Muxing)

### Option A: Portable Version (Recommended for Development)

**Download Page:** https://mkvtoolnix.download/downloads.html

**Steps:**
1. Scroll to **"Windows (portable)"** section
2. Click: **"MKVToolNix (portable) 64-bit"** to download the ZIP (~30 MB)
3. Extract the ZIP file
4. Find `mkvmerge.exe` in the extracted folder
5. Copy to `C:\Users\Vaibhav\AVSync\resources\bin\mkvmerge.exe`

### Option B: Using Chocolatey

```powershell
choco install mkvtoolnix
# Then find mkvmerge.exe and copy to resources/bin/
```

---

## Quick Checklist

After downloading, verify you have these 3 files:

```
C:\Users\Vaibhav\AVSync\resources\bin\
├── ffmpeg.exe      ✅
├── ffprobe.exe     ✅
└── mkvmerge.exe    ✅
```

**To verify:**

```powershell
cd C:\Users\Vaibhav\AVSync
dir resources\bin
```

You should see all 3 .exe files listed.

---

## Test the Binaries

You can test each binary works correctly:

```powershell
# Test FFmpeg
.\resources\bin\ffmpeg.exe -version

# Test FFprobe
.\resources\bin\ffprobe.exe -version

# Test MKVMerge
.\resources\bin\mkvmerge.exe --version
```

Each should display version information without errors.

---

## File Sizes (Approximate)

- `ffmpeg.exe`: ~90 MB
- `ffprobe.exe`: ~90 MB
- `mkvmerge.exe`: ~15 MB

**Total:** ~195 MB

---

## After Downloading Binaries

Once you have all 3 binaries in place:

1. **You still need to build the AVSync executable:**
   ```powershell
   pip install -r requirements.txt
   npm run build:pyinstaller
   copy dist\avsync\avsync.exe resources\bin\avsync.exe
   ```

2. **Then test the app:**
   ```powershell
   npm run dev
   ```

---

## Troubleshooting

**"Access Denied" when copying files:**
- Make sure the files aren't open in another program
- Run PowerShell as Administrator if needed

**Files are missing after extracting:**
- Make sure you extracted the entire ZIP, not just opened it
- Look in the `bin/` subfolder of the extracted archive

**Can't find the download links:**
- FFmpeg: Go to https://ffmpeg.org/ → Download → Windows builds by gyan.dev
- MKVToolNix: Go to https://mkvtoolnix.download/ → Downloads → Windows portable

---

## Alternative: One-Line Download Script

If you have PowerShell 7+ and want to automate downloads:

```powershell
# This is advanced - only use if comfortable with PowerShell
# You'll still need to extract the archives manually

# Download FFmpeg
Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile "ffmpeg.zip"

# Download MKVToolNix (check website for latest version URL)
# Manual extraction still required
```

**Note:** Automated extraction and moving of files from these archives is complex. Manual download and copy is recommended.

---

**Ready?** Once you have all binaries, return to the main setup guide or run `npm run dev` to test!
