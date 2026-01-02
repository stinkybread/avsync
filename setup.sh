#!/bin/bash

echo "========================================"
echo "AVSync Desktop - Unix Setup Script"
echo "========================================"
echo ""

echo "Step 1: Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Node.js dependencies"
    exit 1
fi
echo ""

echo "Step 2: Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi
echo ""

echo "Step 3: Building AVSync executable with PyInstaller..."
pyinstaller --clean avsync.spec
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build AVSync executable"
    exit 1
fi
echo ""

echo "Step 4: Creating resources directory..."
mkdir -p resources/bin
echo ""

echo "Step 5: Copying AVSync PyInstaller bundle to resources..."
if [ -d "dist/avsync" ]; then
    cp -r dist/avsync resources/avsync
    chmod +x resources/avsync/avsync 2>/dev/null
    echo "AVSync bundle copied successfully"
else
    echo "WARNING: AVSync folder not found at dist/avsync"
fi
echo ""

echo "Step 6: Detecting and copying system binaries..."

# Try to find and copy ffmpeg
if command -v ffmpeg &> /dev/null; then
    cp $(which ffmpeg) resources/bin/
    echo "FFmpeg copied from system"
else
    echo "WARNING: ffmpeg not found in PATH. Please install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
fi

# Try to find and copy ffprobe
if command -v ffprobe &> /dev/null; then
    cp $(which ffprobe) resources/bin/
    echo "FFprobe copied from system"
else
    echo "WARNING: ffprobe not found in PATH"
fi

# Try to find and copy mkvmerge
if command -v mkvmerge &> /dev/null; then
    cp $(which mkvmerge) resources/bin/
    echo "MKVMerge copied from system"
else
    echo "WARNING: mkvmerge not found in PATH. Please install: brew install mkvtoolnix (macOS) or apt install mkvtoolnix (Linux)"
fi

# Set execute permissions
chmod +x resources/bin/* 2>/dev/null

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo ""

if [ ! -f "resources/bin/ffmpeg" ] || [ ! -f "resources/bin/ffprobe" ]; then
    echo "1. Install missing binaries:"
    echo "   macOS: brew install ffmpeg mkvtoolnix"
    echo "   Linux: sudo apt install ffmpeg mkvtoolnix"
    echo ""
fi

echo "2. Run development mode:"
echo "   npm run dev"
echo ""
echo "3. Or build for distribution:"
echo "   npm run package:mac    (macOS)"
echo "   npm run package        (Current platform)"
echo ""
echo "See QUICKSTART.md for more details."
echo ""
