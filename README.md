# avsync
A powerful and intelligent video synchronization tool that aligns foreign audio tracks with reference videos using advanced visual detection and content-aware audio analysis. This tool grew out of the need for better synchronization between foreign language video (eg Hindi NTSC DVDs) and higher-quality reference videos (R2 Japanese DVDs or UK PAL DVDs), particularly for preserving the viewing experience across different language versions of the same content, where there are multi issues for sync.

# Features

1. Hybrid Synchronization: Combines visual feature matching (SIFT) with content-aware audio detection for precise alignment
2. Smart Frame Extraction: Uses scene detection technology to identify key frames for optimal comparison
3. Temporal Consistency: Ensures natural timing throughout synchronized content
4. Content-Aware Audio Processing: Intelligently identifies audio segments for improved alignment
5. Multiple Audio Track Support: Preserves original audio tracks while adding synchronized foreign tracks
6. Advanced Visualization: Generate detailed sync maps and match visualizations to verify alignment quality
7. Flexible Processing: Customizable parameters for threshold sensitivity, frame counts, and preprocessing options

Ideal For

1. Synchronizing dubbed audio with original video
2. Aligning multiple language tracks for multilingual releases across different media
3. Fixing audio/video sync issues in existing media

# Requirements & Dependencies

Dependencies:
1. Python 3.x
2. NumPy
3. OpenCV (cv2)
4. scipy
5. matplotlib
6. librosa
7. soundfile
8. scikit-learn
9. FFmpeg (external command-line tool)

Installation Command:

```pip install numpy opencv-python scipy matplotlib librosa soundfile scikit-learn```

Additionally, you'll need to install FFmpeg separately since it's an external command-line tool:

For Windows (using Chocolatey):

```choco install ffmpeg```

For macOS (using Homebrew):

```brew install ffmpeg```

For Ubuntu/Debian:

```sudo apt-get update```

```sudo apt-get install ffmpeg```

Make sure FFmpeg is available in your system PATH so the Python scripts can call it properly.

# Command & Options
```python AudioSynchronizer.py "E:\Base\Upscaled Episodes\015.mkv" "E:\Base\Hindi Episodes\015.mkv" "E:\Base\Muxed Episodes\015.mkv" --feature sift --sample-mode multi_pass --max-frames 500 --max-comparisons 5000 --scene-threshold 0.25 --db-threshold -40 --min-duration 0.2 --min-silence 500 --temporal-consistency --enhanced-preprocessing --visualize --use-scene-detect --content-aware --preserve-aspect-ratio --force-track 0```

1. ```--feature```: Selects visual feature extraction method (currently only 'sift')
2. ```--sample-mode```: Controls frame sampling strategy (currently only 'multi_pass')
3. ```--use-scene-detect```: Enables scene change detection for frame extraction
4. ```--scene-threshold```: Sets sensitivity threshold for scene detection (0-1)
5. ```--temporal-consistency```: Enforces time consistency in visual sync
6. ```--enhanced-preprocessing```: Applies additional image preprocessing for better matching
7. ```--content-aware```: Enables content-aware audio synchronization
8. ```--db-threshold```: Sets audio detection threshold in dB
9. ```--min-duration```: Sets minimum audio segment duration in seconds
10. ```--min-silence```: Sets minimum silence duration in milliseconds
11. ```--force-track```: Forces use of specific audio track number
12. ```--keep-temp```: Keeps temporary files after completio
13. ```--visualize```: Creates visualizations of the alignment process
14. ```--preserve-aspect-ratio```: Maintains aspect ratio when resizing frames
