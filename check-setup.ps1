# AVSync Desktop Setup Verification Script
# Run this to check if all required files are in place

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "AVSync Desktop - Setup Verification" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$allGood = $true

# Check Node modules
Write-Host "Checking Node.js dependencies..." -ForegroundColor Yellow
if (Test-Path "node_modules") {
    Write-Host "  ✓ node_modules folder exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ node_modules folder missing - run: npm install" -ForegroundColor Red
    $allGood = $false
}

# Check build output
Write-Host "`nChecking build output..." -ForegroundColor Yellow
if (Test-Path "dist/index.html") {
    Write-Host "  ✓ React app built (dist/)" -ForegroundColor Green
} else {
    Write-Host "  ✗ React app not built - run: npm run build" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path "dist-electron/main.js") {
    Write-Host "  ✓ Electron main process built (dist-electron/)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Electron main process not built - run: npm run build" -ForegroundColor Red
    $allGood = $false
}

# Check AVSync PyInstaller bundle
Write-Host "`nChecking AVSync PyInstaller bundle..." -ForegroundColor Yellow
if (Test-Path "resources/avsync/avsync.exe") {
    Write-Host "  ✓ avsync.exe exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ avsync.exe missing - run: npm run build:pyinstaller" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path "resources/avsync/_internal") {
    Write-Host "  ✓ _internal folder exists (Python runtime)" -ForegroundColor Green
} else {
    Write-Host "  ✗ _internal folder missing - run: npm run build:pyinstaller" -ForegroundColor Red
    $allGood = $false
}

# Check external binaries
Write-Host "`nChecking external binaries..." -ForegroundColor Yellow

$binaries = @(
    @{Name="ffmpeg.exe"; Path="resources/bin/ffmpeg.exe"; Url="https://www.gyan.dev/ffmpeg/builds/"},
    @{Name="ffprobe.exe"; Path="resources/bin/ffprobe.exe"; Url="https://www.gyan.dev/ffmpeg/builds/"},
    @{Name="mkvmerge.exe"; Path="resources/bin/mkvmerge.exe"; Url="https://mkvtoolnix.download/"}
)

$missingBinaries = @()

foreach ($binary in $binaries) {
    if (Test-Path $binary.Path) {
        $size = (Get-Item $binary.Path).Length / 1MB
        Write-Host "  ✓ $($binary.Name) exists ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($binary.Name) missing" -ForegroundColor Red
        $missingBinaries += $binary
        $allGood = $false
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "Status: ALL CHECKS PASSED ✓" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Cyan
    Write-Host "Your setup is complete! Run the app with:" -ForegroundColor White
    Write-Host "  npm run dev" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Status: ISSUES FOUND ✗" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Cyan

    if ($missingBinaries.Count -gt 0) {
        Write-Host "Missing binaries - Download from:" -ForegroundColor Yellow
        foreach ($binary in $missingBinaries) {
            Write-Host "  • $($binary.Name): $($binary.Url)" -ForegroundColor White
        }
        Write-Host "`nSee DOWNLOAD_BINARIES.md for detailed instructions." -ForegroundColor Yellow
        Write-Host ""
    }

    Write-Host "Fix the issues above, then run this script again." -ForegroundColor White
    Write-Host ""
}

# Pause at the end
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
