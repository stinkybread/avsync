# AVSync - Audio-Video Synchronization Tool

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FFmpeg](https://img.shields.io/badge/requires-FFmpeg-red.svg)

AVSync is a powerful Python tool that automatically synchronizes foreign audio tracks to reference videos using advanced visual anchor detection and precise audio timing algorithms. Perfect for dubbing, multilingual content creation, and audio replacement workflows.

## ✨ Features

- **🎯 Visual Anchor Detection**: Uses scene change detection and template matching to find corresponding frames between videos
- **🔊 Precise Audio Timing**: Iterative audio processing with millisecond-level precision
- **🌍 Multi-language Support**: Automatic audio stream detection by language codes
- **📊 Quality Control**: Generate side-by-side comparison images and detailed CSV reports
- **⚡ Parallel Processing**: Multi-threaded frame matching for faster processing
- **🎛️ Flexible Configuration**: Extensive customization options for different content types
- **📈 Progress Tracking**: Beautiful colored console output with progress bars

## 🎬 How It Works

1. **Image Pairing Stage**: Extracts scene change frames and matches them between reference and foreign videos
2. **Audio Synchronization Stage**: Processes audio segments iteratively to match reference timing precisely
3. **Muxing Stage**: Combines reference video, original audio, and synchronized foreign audio into final output

## 📋 Requirements

### System Dependencies
- **FFmpeg** and **FFprobe** (must be in system PATH)
- Python 3.7 or higher

### Python Dependencies
```bash
pip install opencv-python scipy numpy tqdm
```

### Optional Dependencies (for enhanced features)
```bash
pip install Pillow imagehash  # For similarity filtering
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/stinkybread/avsync.git
cd avsync
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg**:
   - **Windows**: Download from [FFmpeg.org](https://ffmpeg.org/download.html) or use `winget install FFmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or equivalent

4. **Verify installation**:
```bash
python AVSync.py --help
```

## 💡 Usage

### Basic Usage
```bash
python AVSync.py reference_video.mkv foreign_video.mkv output_video.mkv
```

### Advanced Examples

**Specify language codes:**
```bash
python AVSync.py ref.mkv foreign.mkv output.mkv --ref_lang eng --foreign_lang spa
```

**Use specific audio stream indices:**
```bash
python AVSync.py ref.mkv foreign.mkv output.mkv --ref_stream_idx 1 --foreign_stream_idx 2
```

**Generate QC images and CSV report:**
```bash
python AVSync.py ref.mkv foreign.mkv output.mkv \
  --qc_output_dir ./qc_images \
  --output_csv segments.csv
```

**Keep synchronized audio file:**
```bash
python AVSync.py ref.mkv foreign.mkv output.mkv --output_audio synced_audio.wav
```

**Fine-tune processing parameters:**
```bash
python AVSync.py ref.mkv foreign.mkv output.mkv \
  --scene_threshold 0.3 \
  --match_threshold 0.8 \
  --min_segment_duration 10 \
  --db_threshold -35
```

## ⚙️ Configuration Options

### Image Pairing Parameters
- `--scene_threshold`: Scene change detection sensitivity (0.0-1.0, default: 0.25)
- `--match_threshold`: Template matching threshold (0.0-1.0, default: 0.7)
- `--similarity_threshold`: Perceptual hash difference threshold (default: 4, -1 to disable)

### Audio Processing Parameters
- `--ref_lang` / `--foreign_lang`: Language codes for audio stream selection
- `--db_threshold`: Audio detection threshold in dBFS (default: -40.0)
- `--min_segment_duration`: Minimum segment duration in seconds (default: 5.0)
- `--ref_stream_idx` / `--foreign_stream_idx`: Force specific audio stream indices

### Output Options
- `--output_audio`: Save synchronized audio as WAV file
- `--output_csv`: Export segment timing information
- `--qc_output_dir`: Generate quality control images
- `--mux_foreign_codec`: Audio codec for foreign track (default: aac)
- `--mux_foreign_bitrate`: Bitrate for foreign track (default: 192k)

## 📊 Output Files

### Primary Output
- **Video File**: Reference video + original audio + synchronized foreign audio

### Optional Outputs
- **Synchronized Audio**: WAV file with precisely timed foreign audio
- **QC Images**: Side-by-side frame comparisons for visual verification
- **CSV Report**: Detailed segment timing and processing statistics

## 🎯 Tips for Best Results

### Video Content
- ✅ Use videos with clear scene changes and visual landmarks
- ✅ Ensure good video quality for accurate frame matching
- ✅ Ensure both reference and foreign video are essentially the same bar the audio (extra ads, different intro lengths etc will throw this off)

### Audio Content
- ✅ Ensure audio tracks have clear content boundaries as best as you can 
- ✅ Use similar audio quality between reference and foreign tracks

### Parameter Tuning
- **Lower scene threshold**: Detects more frames (more anchor points)
- **Higher match threshold**: Stricter frame matching (fewer false positives)
- **Longer min segment duration**: Fewer, longer segments (more stable sync)

## 🔧 Troubleshooting

### Common Issues

**"FFmpeg not found"**
- Ensure FFmpeg is installed and in your system PATH
- Test with `ffmpeg -version` in terminal

**"No matches found"**
- Try lowering `--scene_threshold` (e.g., 0.15)
- Try lowering `--match_threshold` (e.g., 0.6)
- Check that videos actually correspond to each other

**Audio sync drift**
- Adjust `--min_segment_duration` for your content type
- Check `--db_threshold` if audio boundaries are incorrectly detected
- Review QC images to verify visual anchor quality

**Performance issues**
- Reduce video resolution for faster processing
- Adjust `--similarity_threshold` to reduce redundant anchors
- Use SSD storage for temporary files

## 📈 Performance Notes

- Processing time scales with video length and frame extraction count
- Typical processing speed: 1-5x real-time depending on content and hardware
- Memory usage peaks during frame extraction and comparison phases
- Temporary disk space required: ~2-10GB for feature-length content

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/stinkybread/avsync.git
cd avsync
pip install -r requirements-dev.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FFmpeg team for the excellent multimedia framework
- OpenCV community for computer vision tools
- SciPy contributors for audio processing capabilities

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/stinkybread/avsync/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/stinkybread/avsync/discussions)
- 📧 **Email**: vaibhav.bhat@gmail.com

---
