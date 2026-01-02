@echo off
echo ========================================
echo AVSync Desktop - Windows Setup Script
echo ========================================
echo.

echo Step 1: Installing Node.js dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)
echo.

echo Step 2: Installing Python dependencies...
call pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo.

echo Step 3: Building AVSync executable with PyInstaller...
call pyinstaller --clean avsync.spec
if %errorlevel% neq 0 (
    echo ERROR: Failed to build AVSync executable
    pause
    exit /b 1
)
echo.

echo Step 4: Creating resources directory...
if not exist "resources\bin" mkdir resources\bin
echo.

echo Step 5: Copying AVSync PyInstaller bundle to resources...
if exist "dist\avsync" (
    xcopy /E /I /Y dist\avsync resources\avsync > nul
    echo AVSync bundle copied successfully
) else (
    echo WARNING: AVSync folder not found at dist\avsync
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo NEXT STEPS:
echo.
echo 1. Download FFmpeg and FFprobe from:
echo    https://www.gyan.dev/ffmpeg/builds/
echo    Extract and copy ffmpeg.exe and ffprobe.exe to: resources\bin\
echo.
echo 2. Download MKVToolNix from:
echo    https://mkvtoolnix.download/
echo    Extract and copy mkvmerge.exe to: resources\bin\
echo.
echo 3. Run development mode:
echo    npm run dev
echo.
echo 4. Or build for distribution:
echo    npm run package:win
echo.
echo See QUICKSTART.md for more details.
echo.
pause
