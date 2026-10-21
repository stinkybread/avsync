# AVSync — Audio/Video Synchronization Engine

![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

AVSync retimes a **foreign-language audio track** (and its subtitles) to the exact timeline of a **reference video**. It finds matching frames in both videos, stretches each foreign segment to the length of the matching reference segment, and muxes a clean output that keeps everything from the reference release.

Typical use: you have a release with the picture and timing you want (e.g. an English Blu-ray) and another release with the dub you want (e.g. a Hindi or Japanese TV rip). AVSync puts the dub on the good release, frame-accurate.

**Current version: 18.1** — desktop GUI, batch mode, uniform sampling, post-mux extraction, aspect-ratio-aware matching and bounded memory use. See [What's new](#whats-new).

## Files

| File | Purpose |
|---|---|
| `AVSync.py` | The engine (command line) |
| `AVSync_GUI.py` | Desktop GUI (Tkinter, no extra packages) — single files and batch |
| `AVSync_batch.py` | Batch runner: pairs two folders by `SxxExx` episode code |

Keep the three files in the same folder.

## Requirements

- **Python 3.9+** (tested on 3.12 and 3.13)
- **FFmpeg + FFprobe** on `PATH` — a full build with SoxR is recommended. FFmpeg 5.1+ is preferred; older builds still work.
- **MKVToolNix** (`mkvmerge`) — optional but recommended for MKV output; it preserves chapters, fonts and attachments untouched. Without it, FFmpeg is used for muxing. The GUI can locate or help install it.
- Python packages:

```bash
pip install -r requirements.txt
```

FFmpeg builds: https://ffbinaries.com/downloads · MKVToolNix: https://mkvtoolnix.download/

## Quick start

### GUI

```bash
python AVSync_GUI.py
```

Pick the reference video, the foreign video and an output path on **Quick Start**, set the foreign language code (e.g. `hin`, `jpn`, `spa`), and press **Run Sync**. Every engine option is available on the other tabs with tooltips, and **Preview Command** prints the equivalent command line. Settings are remembered between sessions.

The **Batch** tab syncs whole folders: episodes are paired by their `S01E05`-style code, **Preview Pairs** shows the pairing before you start, and episodes that already have an output are skipped unless you tick re-process.

### Command line

```bash
python AVSync.py reference.mkv foreign.mkv output.mkv --ref_lang eng --foreign_lang jpn --auto_detect
```

Without `--auto_detect` you are asked which audio streams to use (when there's no console, e.g. from the GUI, auto-detection is used).

### Batch

```bash
python AVSync_batch.py ./ref ./foreign ./output --foreign_lang jpn --foreign_tracks all
```

- Files are paired by `SxxExx` (case-insensitive), so the rest of the names can differ.
- Any option after the three folders is passed through to the engine; `--auto_detect` is added automatically.
- Existing outputs are skipped unless `--overwrite` is given. A real `--foreign_lang` is required.
- `--engine PATH` points at an engine elsewhere.

## How it works

1. **Image pairing** — Scene-change frames are extracted from both videos and matched with OpenCV template matching to create visual anchors linking the two timelines. The default *anchor-and-follow* search locks onto the first match, then looks only a few seconds ahead of the expected position for each next match, which is fast and resistant to false matches.
2. **Optional extra anchors** — Forced sync points you give by hand, and/or *uniform sampling* that adds a matched anchor every N seconds where scene detection is sparse.
3. **Audio sync** — The foreign audio is cut at the anchors; each segment is time-stretched (`atempo`, refined iteratively to within a few ms) to the length of its reference segment, then joined and padded to the reference start.
4. **Subtitles** — Text subtitles get the same per-segment retiming. Bitmap subtitles are passed through.
5. **Muxing** — The reference file plus the synced track(s) and subtitles, tagged with language codes.
6. **Optional post-mux extraction** — Save the synced tracks as separate files (and optionally drop the muxed video).

## Options

### Core

| Option | Default | Description |
|---|---|---|
| `--ref_lang` / `--foreign_lang` | `eng` / — | ISO 639-2 codes (`eng`, `jpn`, `hin`, `spa`…). A real foreign code is needed for tagging. |
| `--auto_detect` | off | Pick audio streams by language without prompting |
| `--foreign_tracks` | primary | `primary`, `all`, or absolute stream numbers like `1,2` |
| `--ref_stream_idx` / `--foreign_stream_idx` | auto | Force specific audio streams |
| `--no_subtitles` | off | Skip subtitles |
| `--output_audio` / `--output_csv` / `--qc_output_dir` | none | Also save the synced WAV, a segment report, side-by-side QC images |

### Matching

| Option | Default | Description |
|---|---|---|
| `--scene_threshold` | `0.25` | Scene-change sensitivity (lower = more frames) |
| `--match_threshold` | `0.7` | Template-match score needed to accept a pair |
| `--similarity_threshold` | `4` | pHash filter for near-duplicate reference frames (`-1` disables) |
| `--use_precise_sync` | off | Score every foreign frame within ±6% of the duration and keep the best, instead of anchor-and-follow. Slower; can help when many shots look alike. |
| `--match_keep_aspect` | off | Compare frames at the same height without stretching — use when aspect ratios differ (e.g. a cropped 4:3 dub vs a 16:9 HD release) |
| `--force_sync_points` | none | Manual anchors: `HH:MM:SS:MS>HH:MM:SS:MS,…` (reference > foreign) |
| `--force_sync_only` | off | Use only the forced points; skip scene matching |
| `--uniform_sample_interval` | `0` (off) | Add a matched anchor every N seconds. 60–120 s is a good start; go lower for content that drifts. |
| `--uniform_match_threshold` / `--uniform_crop_center` | `0.7` / `0.7` | Match score and centre-crop fraction for uniform samples |
| `--uniform_ref_aspect_ratio` / `--uniform_foreign_aspect_ratio` | none | Force an aspect ratio (e.g. `4:3`) for uniform-sample frames |
| `--uniform_min_variance` / `--uniform_min_blur` / `--uniform_min_edges` | `150` / `40` / `300` | Quality gate that rejects blank, blurry or featureless sample frames |

### Audio timing

| Option | Default | Description |
|---|---|---|
| `--db_threshold` | `-40.0` | Level that counts as content when finding where audio starts and ends |
| `--min_segment_duration` | `5.0` | Shortest reference segment kept (seconds) |
| `--first_segment_adjust` / `--last_segment_adjust` | `0` | Nudge the first/last segment in **milliseconds** (+ pads, − trims) |

### Output

| Option | Default | Description |
|---|---|---|
| `--mux_foreign_codec` / `--mux_foreign_bitrate` | `aac` / `192k` | Codec/bitrate of the synced track (`copy` keeps PCM) |
| `--clean_track_names` | off | Clear track titles so players show "Track N – Language" |
| `--extract_synced_audio` | off | Also write the synced audio as `<output>.<lang>.<ext>` next to the output |
| `--extract_synced_subtitles` | off | Also write the synced subtitles (`.srt`, `.ass`, `.sup`) |
| `--delete_output_after_extract` | off | Keep only the extracted files. Useful for Jellyfin/Plex: the original media stays untouched, so trickplay and intro detection aren't redone. |

### Performance

| Option | Default | Description |
|---|---|---|
| `--use-cache` / `--no-cache` | on | Reuse visual anchors between runs on the same files. Timing-only changes (segment adjustments) don't invalidate the cache. |
| `--low-memory` | off | Smaller frame cache (64 MB) and at most 2 threads, for very large sources on low-RAM machines |
| `--max-workers N` | auto | Cap FFmpeg/OpenCV threads |
| `--verbose`, `--show-warnings`, `--log-file`, `--no-log` | — | Console detail and log file control (a full log is written next to the output by default) |

Run `python AVSync.py -h` for the complete list.

## Subtitles

| Source format | Handling | Retimed? |
|---|---|---|
| SRT | Parsed and retimed | Yes |
| ASS / SSA | Retimed natively; styles, fonts and tags preserved | Yes (timestamps only) |
| PGS (`hdmv_pgs_subtitle`), VobSub | Copied through unchanged | No — keeps the foreign timing |

Subtitles outside the anchored range are dropped and each one is logged with the reason.

## Tips

- Both videos should contain the same scenes; extra intros or recaps are fine, missing scenes reduce anchors.
- No or few matches: lower `--scene_threshold` (e.g. `0.15`) and/or `--match_threshold` (e.g. `0.6`). If the two releases have different aspect ratios, try `--match_keep_aspect`.
- Drift between scene changes: enable uniform sampling.
- A stubborn section: add `--force_sync_points` there.
- Check results quickly with `--qc_output_dir` and `--output_csv`.

## What's new

**18.1**
- Post-mux extraction of synced audio/subtitles, with optional deletion of the muxed file
- `--match_keep_aspect` for releases with different aspect ratios
- `--use_precise_sync` brings back exhaustive best-match pairing as an option
- Bounded memory: audio analysis streams the WAV, foreign frames load into a size-limited cache, FFmpeg logs spool to disk, and WAV intermediates over 4 GB (RF64) work. New `--low-memory`, `--max-workers` and `--force_sync_only`.
- Uses `-fps_mode` instead of the deprecated `-vsync` (falls back automatically on old FFmpeg)
- FFmpeg muxing no longer retags the reference's own subtitle tracks with the foreign language

**18**
- Desktop GUI with batch mode
- Uniform sampling for full-timeline anchors, with aspect-ratio forcing and frame quality gates
- `--clean_track_names`
- `mkvmerge` is optional (FFmpeg fallback)
- Non-interactive runs fall back to auto-detection instead of waiting for input

**14**
- Anchor-and-follow matching
- Native ASS/SSA retiming
- PGS/VobSub pass-through
- Logging of dropped subtitles

## Credits

- **Developer:** [Vaibhav Bhat](https://github.com/stinkybread)
- **GUI:** original GUI shell by **uruseiyatsura** (Discord)
- **[NP-Gaming](https://github.com/NP-Gaming):** post-mux extraction, precise matching and aspect-ratio-aware matching ([#8](https://github.com/stinkybread/avsync/pull/8))
- **[xsoulp](https://github.com/xsoulp):** bounded-memory audio analysis and frame caching, RF64 support, forced-anchor-only mode ([#9](https://github.com/stinkybread/avsync/pull/9))
- **[Ethan-BB](https://github.com/Ethan-BB):** `-vsync` deprecation report ([#5](https://github.com/stinkybread/avsync/issues/5))

Built with FFmpeg, OpenCV, SciPy and MKVToolNix.

## License

MIT — see [LICENSE](LICENSE).
