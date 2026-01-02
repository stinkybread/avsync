# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files and hidden imports
datas = []
hiddenimports = [
    'cv2',
    'numpy',
    'scipy',
    'scipy.io',
    'scipy.io.wavfile',
    'tqdm',
    'pickle',
    'hashlib',
    'PIL',
    'PIL.Image',
    'imagehash',
]

# Collect OpenCV data files
datas += collect_data_files('cv2', include_py_files=False)

# Bundle external binaries
binaries = []
bin_dir = 'resources/bin'
if os.path.exists(bin_dir):
    bin_files = ['ffmpeg.exe', 'ffprobe.exe', 'mkvmerge.exe', 'mkvextract.exe']
    for bin_file in bin_files:
        bin_path = os.path.join(bin_dir, bin_file)
        if os.path.exists(bin_path):
            binaries.append((bin_path, 'bin'))

a = Analysis(
    ['AVSync_v12.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
        'PyQt5',
        'PySide2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='avsync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='avsync',
)
