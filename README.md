# AVSync v14 — Audio/Video Synchronization Engine

![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A command-line tool that synchronizes a **foreign-language audio track** to a **reference video** using visual anchor detection and precise per-segment audio timing. It aligns dubbed audio (and subtitles) from one release to the exact timeline of another, then muxes a clean output that preserves all of the reference video's original content.

Typical use case: you have an English reference release with correct timing, and a foreign release (e.g. Japanese or French) whose audio you want retimed to match the reference video frame-for-frame.

## How It Works

1. **Image Pairing** — Scene-change frames are extracted from both videos with FFmpeg, then matched via OpenCV template matching (`TM_CCOEFF_NORMED`) to build a set of visual anchors linking the two timelines.
2. **Audio Synchronization** — The foreign audio is split into segments defined by the anchors. Each segment is time-stretched (`atempo`) so its duration matches the corresponding reference segment, then concatenated and padded to align with the reference start.
3. **Subtitle Synchronization** — Text subtitles are retimed with the same per-segment stretch as the audio. Bitmap subtitles are passed through unchanged (see below).
4. **Muxing** — The reference video, its original audio, the synced foreign track(s), and subtitles are combined into the final file. On MKV, `mkvmerge` is used so all original streams, chapters, fonts, and metadata are preserved untouched.

## What's New in v14

- **Anchor-and-follow frame matching.** The first frame is located with a wide search window (±6% of reference duration) to establish an anchor. Every subsequent frame is then searched only in a narrow window **0 to +10 seconds forward** of its estimated position (derived from the anchor offset). Combined with a foreign-frame cache and lower match resolution, this is dramatically faster than the previous fixed-window scan.
- **Lower template-match resolution (640×360).** Frames are compared at half the previous resolution. Whole-frame scene matching is unaffected in accuracy but much faster.
- **Native ASS/SSA subtitle preservation.** Styled subtitles keep their fonts, colors, positioning, and inline tags. Only `Dialogue:` timestamps are adjusted; the entire header, styles, and any `[Fonts]` sections are preserved verbatim. Non-ASS text subs are handled as SRT.
- **Bitmap subtitle pass-through (PGS / VobSub).** Image-based subtitles (`hdmv_pgs_subtitle`, `dvd_subtitle`, etc.) cannot be text-parsed or retimed. They are now copied through into the output unchanged, with a clear log warning that their timing matches the foreign source rather than the adjusted audio.
- **Visible dropped-subtitle logging.** Any subtitle falling outside the anchored segment range is logged individually — with its index, timestamps, a text preview, and the specific reason (before the first boundary, after the last, or in a gap) — instead of being silently discarded.

### Maintenance / Bugfixes (post-v14 QC)

- Fixed broken stage-header log messages (control-character placeholders).
- Fixed ffmpeg concat demuxer path resolution (absolute paths are now used so processing no longer depends on the process working directory).
- Cache checkpoint version bumped to 14 for compatibility.
- Batch script: fixed a copy-paste bug that prevented detection of an empty foreign directory; improved episode-code matching diagnostics; the main engine is now located relative to the batch script itself.

## Requirements

- **Python** 3.8+
- **FFmpeg** and **FFprobe** (full build with SoxR recommended for high-quality resampling)
- **MKVToolNix** (`mkvmerge`) for MKV muxing
- Python packages: see `requirements.txt` (OpenCV, NumPy, SciPy, tqdm; optional `imagehash` + `Pillow` for similarity filtering)

```bash
pip install -r requirements.txt
```

FFmpeg full builds: https://ffbinaries.com/downloads · MKVToolNix: https://mkvtoolnix.download/

## Usage

### Single pair

```bash
python AVSync_v14.py "reference.mkv" "foreign.mkv" "output.mkv"
```

You'll be prompted to pick reference and foreign audio streams. Pass `--auto_detect` to skip the prompts and select streams by language automatically.

```bash
python AVSync_v14.py ref.mkv foreign.mkv out.mkv \
    --ref_lang eng --foreign_lang jpn --auto_detect
```

### Batch (match by SxxExx episode code)

`AVSync_batch_regex.py` pairs files across two folders by their `SxxExx` season/episode code (case-insensitive) rather than exact filename, then runs the engine on each pair. `--auto_detect` is injected automatically.

```bash
python AVSync_batch_regex.py ./ref ./foreign ./output --foreign_lang jpn --foreign_tracks all
```

- Skips outputs that already exist unless `--overwrite` is given.
- All arguments after the three folders are passed straight through to `AVSync_v14.py`.
- A valid `--foreign_lang` is required in batch mode (the engine needs it for metadata tagging).

## Key Options

| Option | Default | Description |
|---|---|---|
| `--ref_lang` / `--foreign_lang` | `eng` / `foreign` | ISO 639-2 language codes (e.g. `eng`, `jpn`, `fre`, `hin`) |
| `--auto_detect` | off | Skip interactive prompts; select streams by language |
| `--foreign_tracks` | primary | `primary`, `all`, or comma-separated stream indices to sync |
| `--scene_threshold` | `0.25` | Scene-change sensitivity for frame extraction (0.0–1.0) |
| `--match_threshold` | `0.7` | Template-match acceptance threshold (0.0–1.0) |
| `--similarity_threshold` | `4` | Perceptual-hash dedup distance (`-1` to disable) |
| `--db_threshold` | `-40.0` | Audio content detection threshold in dB |
| `--min_segment_duration` | `5.0` | Minimum reference segment length in seconds |
| `--first_segment_adjust` / `--last_segment_adjust` | `0.0` | Manual timing nudge in **milliseconds** for the first/last segment |
| `--force_sync_points` | none | Manual anchor points for problem sections |
| `--mux_foreign_codec` / `--mux_foreign_bitrate` | `aac` / `192k` | Output codec/bitrate for synced foreign audio |
| `--no_subtitles` | off | Skip subtitle handling entirely |
| `--qc_output_dir` | none | Write side-by-side QC comparison images |
| `--output_csv` | none | Write an anchor/segment report |
| `--verbose` | off | Show DEBUG-level detail |

## Subtitle Behavior at a Glance

| Source format | Handling | Timing adjusted? |
|---|---|---|
| SRT / SubRip | Parsed, retimed, written as SRT | Yes |
| ASS / SSA | Parsed natively, styles preserved, retimed | Yes (timestamps only) |
| PGS (`hdmv_pgs_subtitle`) | Passed through unchanged | No — matches foreign source |
| VobSub (`dvd_subtitle`) | Passed through unchanged | No — matches foreign source |

Subtitles that fall outside the anchored segment range are dropped and logged individually with a reason.

## Tips for Best Results

- Both videos should contain essentially the same scenes. Different intros, ads, or missing scenes will reduce anchor quality.
- If no matches are found, lower `--scene_threshold` (e.g. `0.15`) and/or `--match_threshold` (e.g. `0.6`).
- Uniform, clear scene changes give the most reliable anchors.
- Use `--force_sync_points` for sections that consistently misalign.

## License

MIT — see [LICENSE](LICENSE).

## Credits
**Shout-Outs** [NP-Gaming]((https://github.com/NP-Gaming)
**Developer:** [Vaibhav Bhat](https://github.com/stinkybread)
Built with FFmpeg, OpenCV, SciPy, and MKVToolNix.
