#!/usr/bin/env python3
"""
AVSync GUI - a desktop front-end for AVSync_v18.py

Run this in the SAME folder as AVSync_v18.py and AVSync_batch_regex.py (or point it at the script
via the "Script path" field in Setup). Requires only the Python standard
library (Tkinter). AVSync_v18.py still needs its own dependencies
(opencv, scipy, tqdm, numpy) plus ffmpeg / ffprobe on PATH, and
optionally mkvmerge (this GUI can detect / install / locate it for you).
"""

import json
import os
import platform
import queue
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "AVSync v18 GUI"
ENGINE_NAME = "AVSync_v18.py"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DEFAULT = os.path.join(HERE, ENGINE_NAME)
BATCH_NAME = "AVSync_batch_regex.py"
VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov", ".ts", ".webm")  # keep in sync with the batch script
EPISODE_RE = re.compile(r"S\d{2}E\d{2}", re.IGNORECASE)
BATCH_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\] Processing: (\S+)")
SETTINGS_FILE = os.path.join(HERE, "avsync_gui_settings.json")

MKVTOOLNIX_URL = "https://mkvtoolnix.download/downloads.html"

# --------------------------------------------------------------------------
# Field registry: one place that describes every option. The UI, the CLI
# command builder, and settings save/load are all generated from this list
# instead of being written out by hand three separate times.
# --------------------------------------------------------------------------
# kind:      "entry" | "check" | "combo"
# tab:       which notebook tab this field appears on
# section:   optional group header drawn above this field (for readability)
# cli:       the CLI flag to emit ("--foo"); None for positionals or fields
#            with custom handling (e.g. the cache checkbox)
# positional: 0/1/2 sort order for the three required file arguments
FIELDS = [
    # ---- Quick Start: required files -------------------------------------
    dict(key="ref_video", label="Reference video (original language)", kind="entry",
         browse="open", tab="Quick Start", positional=0,
         section="Step 1 - Choose your files",
         tooltip="The video with the timing you want to match TO - e.g. the "
                 "original-language release."),
    dict(key="foreign_video", label="Foreign video (to be synced)", kind="entry",
         browse="open", tab="Quick Start", positional=1,
         tooltip="The video whose audio (and subtitles) will be stretched/shifted "
                 "to match the reference video."),
    dict(key="output_video", label="Output video (final result)", kind="entry",
         browse="save", tab="Quick Start", positional=2,
         tooltip="Where to save the finished video: reference picture + the "
                 "newly-synced foreign audio track."),

    # ---- Batch tab (whole folders, paired by SxxExx) -----------------------
    dict(key="batch_ref_dir", label="Reference videos folder", kind="entry",
         browse="dir", tab="Batch", batch_only=True,
         section="Folders",
         tooltip="Folder of reference videos. Each file needs an episode code like S01E05 in its name."),
    dict(key="batch_foreign_dir", label="Foreign videos folder", kind="entry",
         browse="dir", tab="Batch", batch_only=True,
         tooltip="Folder of foreign videos. Files are paired with the reference by S01E05-style "
                 "codes (case-insensitive), so the rest of the name can differ."),
    dict(key="batch_output_dir", label="Output folder", kind="entry",
         browse="dir", tab="Batch", batch_only=True,
         tooltip="Synced files are saved here using the reference file's name."),
    dict(key="batch_overwrite", label="Re-process episodes that already exist in the output folder", kind="check",
         default=False, cli="--overwrite", tab="Batch", batch_only=True,
         section="Options",
         tooltip="Off = skip episodes whose output already exists (lets you resume an interrupted batch)."),

    # ---- Quick Start: basic options ---------------------------------------
    dict(key="foreign_lang", label="Foreign audio language code", kind="combo",
         choices=["hin", "jpn", "spa", "fre", "ger", "ita", "kor", "chi", "por", "rus", "foreign"],
         default="foreign", cli="--foreign_lang", tab="Quick Start",
         section="Step 2 - Basic options",
         tooltip="3-letter language code (ISO 639-2/T) used to label the foreign "
                 "audio track, e.g. 'jpn' for Japanese."),
    dict(key="auto_detect", label="Auto-detect audio tracks (recommended)", kind="check",
         default=True, cli="--auto_detect", tab="Quick Start",
         tooltip="Let the tool pick the right audio tracks automatically instead of "
                 "pausing to ask you in the terminal."),
    dict(key="no_subs", label="Skip subtitle syncing", kind="check",
         default=False, cli="--no_subtitles", tab="Quick Start",
         tooltip="Turn on to skip subtitles. When off: SRT and ASS/SSA are retimed (ASS styling kept); "
                 "PGS/VobSub image subtitles are copied unchanged."),

    # ---- Quick Start: optional outputs -------------------------------------
    dict(key="out_audio", label="Also save synced audio as its own file (.wav)", kind="entry",
         browse="save", cli="--output_audio", tab="Quick Start",
         section="Optional outputs (leave blank to skip)",
         tooltip="Keep a standalone WAV copy of the synced foreign audio track."),
    dict(key="out_csv", label="Save a segment timing report (.csv)", kind="entry",
         browse="save", cli="--output_csv", tab="Quick Start",
         tooltip="Write every sync segment and its timing to a spreadsheet, useful for QC."),
    dict(key="qc_dir", label="Save side-by-side QC images to a folder", kind="entry",
         browse="dir", cli="--qc_output_dir", tab="Quick Start",
         tooltip="Saves paired reference/foreign frame images so you can visually verify sync points."),

    # ---- Audio tab ----------------------------------------------------------
    dict(key="ref_lang", label="Reference audio language code", kind="entry",
         default="eng", cli="--ref_lang", tab="Audio",
         section="Track selection",
         tooltip="3-letter language code used to find the reference audio track."),
    dict(key="ref_stream", label="Force reference track # (blank = auto)", kind="entry",
         cli="--ref_stream_idx", tab="Audio",
         tooltip="Only needed if auto-detection picks the wrong track in the reference video."),
    dict(key="foreign_stream", label="Force foreign track # (blank = auto)", kind="entry",
         cli="--foreign_stream_idx", tab="Audio",
         tooltip="Only needed if auto-detection picks the wrong track in the foreign video."),
    dict(key="foreign_tracks", label="Foreign tracks to include", kind="entry",
         cli="--foreign_tracks", tab="Audio",
         tooltip="Blank = primary track only. Or type 'primary', 'all', or absolute stream numbers like '1,2'."),
    dict(key="db_thresh", label="Silence threshold (dBFS)", kind="entry",
         default="-40.0", cli="--db_threshold", tab="Audio",
         section="Fine-tuning",
         tooltip="How quiet counts as 'silence' when detecting where dialogue starts/ends. "
                 "More negative = more sensitive."),
    dict(key="min_seg", label="Minimum segment length (seconds)", kind="entry",
         default="5.0", cli="--min_segment_duration", tab="Audio",
         tooltip="Sync segments shorter than this are treated as noise and discarded."),
    dict(key="first_adj", label="First segment adjustment (ms)", kind="entry",
         default="0.0", cli="--first_segment_adjust", tab="Audio",
         tooltip="Nudge the very start of the audio. Positive = add silence, negative = trim."),
    dict(key="last_adj", label="Last segment adjustment (ms)", kind="entry",
         default="0.0", cli="--last_segment_adjust", tab="Audio",
         tooltip="Nudge the very end of the audio. Positive = add silence, negative = trim."),

    # ---- Image Pairing tab ----------------------------------------------------
    dict(key="scene_thresh", label="Scene-change sensitivity (0-1)", kind="entry",
         default="0.25", cli="--scene_threshold", tab="Image Pairing",
         tooltip="Lower values detect more scene changes to use as sync anchors."),
    dict(key="match_thresh", label="Frame match confidence (0-1)", kind="entry",
         default="0.7", cli="--match_threshold", tab="Image Pairing",
         tooltip="How confident a frame match must be to count. Higher = stricter."),
    dict(key="sim_thresh", label="Duplicate-frame filter strength", kind="entry",
         default="4", cli="--similarity_threshold", tab="Image Pairing",
         tooltip="Filters near-duplicate reference frames before matching. -1 disables it."),
    dict(key="force_sync", label="Manual sync points (advanced)", kind="entry",
         cli="--force_sync_points", tab="Image Pairing", width=60,
         section="Manual override",
         tooltip="Force sync points where auto-detection fails.\n"
                 "Format: HH:MM:SS:MS>HH:MM:SS:MS,...\n"
                 "Example: 00:00:10:500>00:00:10:800,00:02:00:300>00:02:01:000"),


    # ---- Uniform Sampling tab -------------------------------------------
    dict(key="uni_interval", label="Sample every N seconds (0 = off)", kind="entry",
         default="0", cli="--uniform_sample_interval", tab="Uniform Sampling",
         section="Enable",
         tooltip="Adds an extra sync anchor every N seconds across the whole video, matched "
                 "to the foreign video. Helps when scene detection misses long stretches "
                 "(slow pans, dialogue scenes). 60-120 is a good start. 0 disables it."),
    dict(key="uni_match", label="Match confidence for samples (0-1)", kind="entry",
         default="0.7", cli="--uniform_match_threshold", tab="Uniform Sampling",
         section="Matching",
         tooltip="How confident a uniform-sample match must be. Slightly lower than the scene "
                 "threshold often works because only the frame centre is compared."),
    dict(key="uni_crop", label="Centre crop fraction (0-1)", kind="entry",
         default="0.7", cli="--uniform_crop_center", tab="Uniform Sampling",
         tooltip="Only the centre of each frame is compared, ignoring logos, burnt-in subs "
                 "and letterbox bars. 0.7 = central 70%."),
    dict(key="uni_ref_ar", label="Force reference aspect ratio (e.g. 4:3)", kind="entry",
         cli="--uniform_ref_aspect_ratio", tab="Uniform Sampling",
         tooltip="Leave blank normally. Set when the reference is stretched or letterboxed "
                 "differently from the foreign video."),
    dict(key="uni_for_ar", label="Force foreign aspect ratio (e.g. 16:9)", kind="entry",
         cli="--uniform_foreign_aspect_ratio", tab="Uniform Sampling",
         tooltip="Leave blank normally. Set when the foreign video uses a different aspect ratio."),
    dict(key="uni_min_var", label="Minimum frame variance (blank frames)", kind="entry",
         default="150.0", cli="--uniform_min_variance", tab="Uniform Sampling",
         section="Frame quality gate",
         tooltip="Sample frames flatter than this (black/solid colour) are skipped."),
    dict(key="uni_min_blur", label="Minimum sharpness (blur)", kind="entry",
         default="40.0", cli="--uniform_min_blur", tab="Uniform Sampling",
         tooltip="Sample frames blurrier than this (Laplacian variance) are skipped."),
    dict(key="uni_min_edges", label="Minimum edge pixels (detail)", kind="entry",
         default="300", cli="--uniform_min_edges", tab="Uniform Sampling",
         tooltip="Sample frames with fewer edge pixels than this lack detail and are skipped."),

    # ---- Muxing & Cache tab ------------------------------------------------
    dict(key="mux_codec", label="Output audio codec", kind="entry",
         default="aac", cli="--mux_foreign_codec", tab="Muxing & Cache",
         tooltip="Codec for the synced foreign track in the final file, e.g. aac, ac3, or 'copy'."),
    dict(key="mux_bitrate", label="Output audio bitrate", kind="entry",
         default="192k", cli="--mux_foreign_bitrate", tab="Muxing & Cache",
         tooltip="Bitrate used if re-encoding the audio track, e.g. 192k or 320k."),
    dict(key="clean_names", label="Clear track titles in the output", kind="check",
         default=False, cli="--clean_track_names", tab="Muxing & Cache",
         tooltip="After muxing, clear title tags on every audio/subtitle track so players show "
                 "'Track N - Language'. Also clears titles on the reference's original tracks."),
    dict(key="use_cache", label="Use checkpoint cache (faster re-runs)", kind="check",
         default=True, tab="Muxing & Cache",
         section="Performance",
         tooltip="Speeds up repeated runs on the same files by reusing intermediate work. "
                 "Adds --no-cache when turned off."),

    # ---- Logging tab -----------------------------------------------------
    dict(key="verbose", label="Verbose console output", kind="check",
         default=False, cli="-v", tab="Logging",
         tooltip="Show every detail in the log console instead of just the important messages."),
    dict(key="show_warnings", label="Show warnings", kind="check",
         default=False, cli="--show-warnings", tab="Logging",
         tooltip="Include warning messages in the console output."),
    dict(key="no_log", label="Disable log file", kind="check",
         default=False, cli="--no-log", tab="Logging",
         tooltip="Don't write a .log file to disk - console output only."),
    dict(key="log_file", label="Custom log file path (blank = auto)", kind="entry",
         cli="--log-file", browse="save", tab="Logging",
         tooltip="Optional. Default is a file next to the output video."),
]

TAB_ORDER = ["Quick Start", "Batch", "Audio", "Image Pairing", "Uniform Sampling", "Muxing & Cache", "Logging"]
TAB_HELP = {
    "Quick Start": "Fill these in, then hit Run at the bottom. Everything else has sensible defaults.",
    "Batch": "Sync whole folders: episodes are paired by their S01E05-style code. Every setting on the "
             "other tabs (except the three Quick Start files) is applied to each episode.",
    "Audio": "Controls how audio tracks are picked and fine-tuned. Defaults work for most videos.",
    "Image Pairing": "Controls how scenes are matched between the two videos to find sync points.",
    "Uniform Sampling": "Optional extra sync anchors at fixed intervals. Off by default; turn on if long stretches drift.",
    "Muxing & Cache": "Controls the final file's audio encoding, track naming and re-run speed.",
    "Logging": "Controls how much detail is printed while the job runs.",
}


class Tooltip:
    """Small hover tooltip shown next to any widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid",
                 borderwidth=1, wraplength=340, justify="left", padx=6, pady=4).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class AVSyncGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("920x760")
        self.minsize(800, 620)

        self.proc = None
        self.log_queue = queue.Queue()
        self.vars = {}                # field key -> tk Variable
        self.extra_path_dir = tk.StringVar(value="")   # manually located mkvmerge folder
        self.dep_status = {}          # tool name -> bool found

        self._build_ui()
        self._load_settings()
        self._refresh_dependencies()
        self._update_ready_state()
        self.after(100, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================ UI ==
    def _build_ui(self):
        self._build_dependency_banner()

        paned = ttk.Panedwindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=3)
        paned.add(bottom, weight=2)

        self._build_script_row(top)
        self._build_tabs(top)
        self._build_run_panel(bottom)

    def _build_dependency_banner(self):
        banner = ttk.Frame(self, padding=(8, 6))
        banner.pack(fill="x")
        ttk.Label(banner, text="Dependencies:", font=("", 9, "bold")).pack(side="left")

        self.dep_labels = {}
        for tool in ("ffmpeg", "ffprobe", "mkvmerge"):
            lbl = ttk.Label(banner, text=f"  {tool}: checking...")
            lbl.pack(side="left", padx=(10, 0))
            self.dep_labels[tool] = lbl

        ttk.Button(banner, text="Refresh", command=self._refresh_dependencies).pack(side="right")
        self.btn_locate_mkv = ttk.Button(banner, text="Locate mkvmerge...", command=self._locate_mkvmerge)
        self.btn_locate_mkv.pack(side="right", padx=6)
        self.btn_install_mkv = ttk.Button(banner, text="Install mkvmerge", command=self._attempt_install_mkvmerge)
        self.btn_install_mkv.pack(side="right", padx=6)

        note = ttk.Label(
            self,
            text=("mkvmerge is only required for .mkv output (it preserves every original stream). "
                  "For other formats the tool falls back to ffmpeg automatically."),
            foreground="#666", padding=(8, 0, 8, 6),
        )
        note.pack(fill="x")

    def _build_script_row(self, parent):
        row = ttk.Frame(parent, padding=(4, 0, 4, 6))
        row.pack(fill="x")
        ttk.Label(row, text=f"{ENGINE_NAME} location:").pack(side="left")
        self.e_script = tk.StringVar(value=SCRIPT_DEFAULT if os.path.exists(SCRIPT_DEFAULT) else "")
        entry = ttk.Entry(row, textvariable=self.e_script)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse...", command=self._browse_script).pack(side="left")

    def _browse_script(self):
        path = filedialog.askopenfilename(filetypes=[("Python script", "*.py")])
        if path:
            self.e_script.set(path)

    def _build_tabs(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True, padx=4)

        self.tab_frames = {}
        for name in TAB_ORDER:
            outer = ttk.Frame(self.nb)
            self.nb.add(outer, text=name)
            ttk.Label(outer, text=TAB_HELP[name], foreground="#666", wraplength=820,
                      padding=(6, 8, 6, 4)).pack(fill="x")
            body = ttk.Frame(outer, padding=(6, 0))
            body.pack(fill="both", expand=True)
            body.columnconfigure(1, weight=1)
            self.tab_frames[name] = body

        row_counters = {name: 0 for name in TAB_ORDER}
        last_section = {name: None for name in TAB_ORDER}

        for spec in FIELDS:
            tab = spec["tab"]
            parent_frame = self.tab_frames[tab]
            row = row_counters[tab]

            section = spec.get("section")
            if section and section != last_section[tab]:
                ttk.Label(parent_frame, text=section, font=("", 9, "bold")).grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(10, 2))
                row += 1
                last_section[tab] = section

            row = self._build_field(parent_frame, spec, row)
            row_counters[tab] = row

        # Batch tab action buttons
        batch_body = self.tab_frames["Batch"]
        btns = ttk.Frame(batch_body)
        btns.grid(row=row_counters["Batch"], column=0, columnspan=3, sticky="w", padx=4, pady=(12, 4))
        ttk.Button(btns, text="Preview Pairs", command=self._on_preview_pairs).pack(side="left")
        self.btn_run_batch = ttk.Button(btns, text="Run Batch", command=self._on_run_batch)
        self.btn_run_batch.pack(side="left", padx=6)

        # Required-field indicator on Quick Start
        for key in ("ref_video", "foreign_video", "output_video"):
            self.vars[key].trace_add("write", lambda *_: self._update_ready_state())
        self.e_script.trace_add("write", lambda *_: self._update_ready_state())

    def _build_field(self, parent, spec, row):
        key = spec["key"]
        kind = spec["kind"]
        default = spec.get("default", False if kind == "check" else "")
        tooltip_text = spec.get("tooltip", "")

        if kind == "check":
            var = tk.BooleanVar(value=default)
            widget = ttk.Checkbutton(parent, text=spec["label"], variable=var)
            widget.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        elif kind == "combo":
            var = tk.StringVar(value=default)
            ttk.Label(parent, text=spec["label"]).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            widget = ttk.Combobox(parent, textvariable=var, values=spec.get("choices", []),
                                   width=spec.get("width", 28))
            widget.grid(row=row, column=1, sticky="w", pady=4)
        else:  # entry
            var = tk.StringVar(value=default)
            ttk.Label(parent, text=spec["label"]).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            widget = ttk.Entry(parent, textvariable=var, width=spec.get("width", 46))
            widget.grid(row=row, column=1, sticky="we", pady=4)
            if spec.get("browse"):
                ttk.Button(parent, text="Browse...",
                           command=lambda v=var, mode=spec["browse"]: self._browse_into(v, mode)
                           ).grid(row=row, column=2, padx=4)

        if tooltip_text:
            Tooltip(widget, tooltip_text)

        self.vars[key] = var
        return row + 1

    @staticmethod
    def _browse_into(var, mode):
        if mode == "open":
            path = filedialog.askopenfilename()
        elif mode == "save":
            path = filedialog.asksaveasfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _build_run_panel(self, parent):
        bar = ttk.Frame(parent, padding=(4, 4))
        bar.pack(fill="x")

        self.btn_run = ttk.Button(bar, text="Run Sync", command=self._on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="Stop", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Preview Command", command=self._on_show_command).pack(side="left", padx=6)
        ttk.Button(bar, text="Save Settings", command=self._save_settings).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear Log", command=self._clear_log).pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status_var, foreground="#0a5", font=("", 9, "bold")).pack(side="right")

        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", padx=4, pady=(0, 4))

        log_frame = ttk.Frame(parent)
        log_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.log_text = tk.Text(log_frame, wrap="word", bg="#111", fg="#ddd", insertbackground="#ddd")
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    # ==================================================== dependencies ====
    def _refresh_dependencies(self):
        self.dep_status["ffmpeg"] = shutil.which("ffmpeg") is not None
        self.dep_status["ffprobe"] = shutil.which("ffprobe") is not None
        mkv_on_path = shutil.which("mkvmerge") is not None
        mkv_manual = bool(self.extra_path_dir.get()) and os.path.isfile(
            os.path.join(self.extra_path_dir.get(), self._mkvmerge_exe_name()))
        self.dep_status["mkvmerge"] = mkv_on_path or mkv_manual

        for tool in ("ffmpeg", "ffprobe", "mkvmerge"):
            found = self.dep_status[tool]
            text = f"  {tool}: {'found' if found else 'NOT FOUND'}"
            color = "#0a5" if found else "#c33"
            if tool == "mkvmerge" and found and mkv_manual and not mkv_on_path:
                text += " (manual)"
            self.dep_labels[tool].configure(text=text, foreground=color)

        mkv_found = self.dep_status["mkvmerge"]
        self.btn_install_mkv.configure(state="disabled" if mkv_found else "normal")

    @staticmethod
    def _mkvmerge_exe_name():
        return "mkvmerge.exe" if platform.system() == "Windows" else "mkvmerge"

    def _locate_mkvmerge(self):
        path = filedialog.askopenfilename(title="Select the mkvmerge executable")
        if not path:
            return
        if os.path.basename(path).lower() not in ("mkvmerge", "mkvmerge.exe"):
            if not messagebox.askyesno(APP_TITLE, "That file isn't named 'mkvmerge'. Use it anyway?"):
                return
        self.extra_path_dir.set(os.path.dirname(path))
        self._append_log(f"\n[Using mkvmerge from: {path}]\n"
                          f"[This folder will be added to PATH for AVSync runs from this GUI.]\n")
        self._refresh_dependencies()

    def _attempt_install_mkvmerge(self):
        if not messagebox.askyesno(
            APP_TITLE,
            "This will try to install MKVToolNix (mkvmerge) using your system's package "
            "manager. It may prompt you for an admin/sudo password. Continue?",
        ):
            return
        self.btn_install_mkv.configure(state="disabled")
        self._append_log("\n[Attempting to install mkvmerge...]\n")
        threading.Thread(target=self._install_mkvmerge_worker, daemon=True).start()

    def _install_mkvmerge_worker(self):
        system = platform.system()
        cmd = None

        if system == "Windows":
            if shutil.which("winget"):
                cmd = ["winget", "install", "-e", "--id", "MkvToolNix.MkvToolNix",
                       "--accept-package-agreements", "--accept-source-agreements"]
            elif shutil.which("choco"):
                cmd = ["choco", "install", "mkvtoolnix", "-y"]
        elif system == "Darwin":
            if shutil.which("brew"):
                cmd = ["brew", "install", "mkvtoolnix"]
        else:  # Linux and others
            if shutil.which("pkexec") and shutil.which("apt-get"):
                cmd = ["pkexec", "apt-get", "install", "-y", "mkvtoolnix"]
            elif shutil.which("pkexec") and shutil.which("dnf"):
                cmd = ["pkexec", "dnf", "install", "-y", "mkvtoolnix"]
            elif shutil.which("pkexec") and shutil.which("pacman"):
                cmd = ["pkexec", "pacman", "-S", "--noconfirm", "mkvtoolnix-cli"]

        if not cmd:
            self.log_queue.put(
                "[No supported automatic installer found on this system.]\n"
                f"[Opening the MKVToolNix download page instead: {MKVTOOLNIX_URL}]\n"
            )
            webbrowser.open(MKVTOOLNIX_URL)
            self.log_queue.put(("__RECHECK_DEPS__",))
            return

        self.log_queue.put(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                self.log_queue.put(line)
            proc.wait()
            if proc.returncode == 0:
                self.log_queue.put("[Install command finished successfully.]\n")
            else:
                self.log_queue.put(f"[Install command exited with code {proc.returncode}.]\n")
        except Exception as e:
            self.log_queue.put(f"[Install failed to run: {e}]\n")

        self.log_queue.put(("__RECHECK_DEPS__",))

    # ========================================================= building ===
    def _build_args(self):
        script = self.e_script.get().strip()
        if not script or not os.path.isfile(script):
            raise ValueError(f"Please set a valid path to {ENGINE_NAME} at the top of the window.")

        positionals = sorted((s for s in FIELDS if "positional" in s), key=lambda s: s["positional"])
        pos_values = []
        for spec in positionals:
            val = self.vars[spec["key"]].get().strip()
            if not val:
                raise ValueError(f"'{spec['label']}' is required.")
            pos_values.append(val)

        args = [script, *pos_values]

        args += self._option_args(include_auto_detect=True)
        return args

    def _option_args(self, include_auto_detect):
        """Shared options from every tab, excluding the Quick Start files and batch-only fields."""
        args = []
        for spec in FIELDS:
            if "positional" in spec or "cli" not in spec or spec.get("batch_only"):
                continue
            if spec["cli"] == "--auto_detect" and not include_auto_detect:
                continue  # the batch runner always injects --auto_detect itself
            var = self.vars[spec["key"]]
            if spec["kind"] == "check":
                if var.get():
                    args.append(spec["cli"])
            else:
                val = var.get().strip()
                if val:
                    args.extend([spec["cli"], val])

        # use_cache has custom handling: default True, no flag; False -> --no-cache
        if not self.vars["use_cache"].get():
            args.append("--no-cache")

        return args

    def _full_command(self):
        return [sys.executable, "-u"] + self._build_args()

    def _batch_script_path(self):
        engine = self.e_script.get().strip()
        for folder in (os.path.dirname(engine) if engine else "", HERE):
            candidate = os.path.join(folder, BATCH_NAME) if folder else ""
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _batch_dirs(self):
        dirs = [self.vars[k].get().strip() for k in ("batch_ref_dir", "batch_foreign_dir", "batch_output_dir")]
        labels = ("Reference videos folder", "Foreign videos folder", "Output folder")
        for d, label in zip(dirs, labels):
            if not d:
                raise ValueError(f"Batch: '{label}' is required.")
        for d, label in zip(dirs[:2], labels[:2]):
            if not os.path.isdir(d):
                raise ValueError(f"Batch: '{label}' does not exist:\n{d}")
        return dirs

    def _build_batch_command(self):
        engine = self.e_script.get().strip()
        if not engine or not os.path.isfile(engine):
            raise ValueError(f"Please set a valid path to {ENGINE_NAME} at the top of the window.")
        batch = self._batch_script_path()
        if not batch:
            raise ValueError(f"{BATCH_NAME} was not found next to {ENGINE_NAME} or next to this GUI.")
        ref_dir, foreign_dir, out_dir = self._batch_dirs()
        lang = self.vars["foreign_lang"].get().strip().lower()
        if len(lang) != 3 or not lang.isalpha() or lang in ("und", "unk"):
            raise ValueError("Batch mode needs a real 3-letter foreign language code on Quick Start "
                             "(e.g. 'jpn', 'spa'), not 'foreign'.")
        cmd = [sys.executable, "-u", batch, ref_dir, foreign_dir, out_dir, "--engine", engine]
        if self.vars["batch_overwrite"].get():
            cmd.append("--overwrite")
        return cmd + self._option_args(include_auto_detect=False)

    def _scan_pairs(self, ref_dir, foreign_dir):
        def index(folder):
            found, no_code, dupes = {}, [], []
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                m = EPISODE_RE.search(name)
                if not m:
                    no_code.append(name)
                    continue
                code = m.group().upper()
                if code in found:
                    dupes.append(code)
                found[code] = name
            return found, no_code, dupes
        return index(ref_dir), index(foreign_dir)

    def _build_env(self):
        env = os.environ.copy()
        extra = self.extra_path_dir.get().strip()
        if extra:
            env["PATH"] = extra + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    # =============================================================== run ==
    def _update_ready_state(self):
        script_ok = bool(self.e_script.get().strip()) and os.path.isfile(self.e_script.get().strip())
        files_ok = all(self.vars.get(k) and self.vars[k].get().strip()
                        for k in ("ref_video", "foreign_video", "output_video"))
        if script_ok and files_ok:
            self.status_var.set("Ready to run")
        else:
            self.status_var.set("Fill in the required files on Quick Start")

    def _active_tab(self):
        try:
            return self.nb.tab(self.nb.select(), "text")
        except Exception:
            return ""

    def _on_show_command(self):
        try:
            cmd = self._build_batch_command() if self._active_tab() == "Batch" else self._full_command()
        except ValueError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        pretty = " ".join(shlex.quote(c) for c in cmd)
        top = tk.Toplevel(self)
        top.title("Command that will run")
        txt = tk.Text(top, wrap="word", width=110, height=10)
        txt.insert("1.0", pretty)
        txt.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_run(self):
        if self.proc is not None:
            messagebox.showwarning(APP_TITLE, "A sync job is already running.")
            return
        try:
            cmd = self._full_command()
        except ValueError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        self._launch(cmd, "Running...")

    def _launch(self, cmd, status):
        self._clear_log()
        self._append_log(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        self.status_var.set(status)
        self.btn_run.configure(state="disabled")
        self.btn_run_batch.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.start(12)
        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()

    def _on_run_batch(self):
        if self.proc is not None:
            messagebox.showwarning(APP_TITLE, "A job is already running.")
            return
        try:
            cmd = self._build_batch_command()
            (ref_idx, _, ref_dupes), (for_idx, _, for_dupes) = self._scan_pairs(cmd[3], cmd[4])
        except (ValueError, OSError) as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        if ref_dupes or for_dupes:
            messagebox.showerror(APP_TITLE, "Duplicate episode codes found: "
                                 f"{', '.join(sorted(set(ref_dupes + for_dupes)))}. Use Preview Pairs to see them.")
            return
        pairs = sorted(set(ref_idx) & set(for_idx))
        if not pairs:
            messagebox.showerror(APP_TITLE, "No matching episode codes between the two folders. Use Preview Pairs.")
            return
        if not messagebox.askyesno(APP_TITLE, f"Run AVSync on {len(pairs)} matched episode(s)?"):
            return
        self._launch(cmd, f"Batch: starting {len(pairs)} episode(s)...")

    def _on_preview_pairs(self):
        try:
            ref_dir, foreign_dir, out_dir = self._batch_dirs()
            (ref_idx, ref_none, ref_dupes), (for_idx, for_none, for_dupes) = self._scan_pairs(ref_dir, foreign_dir)
        except (ValueError, OSError) as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        overwrite = self.vars["batch_overwrite"].get()
        pairs = sorted(set(ref_idx) & set(for_idx))
        lines = ["", "===== Batch pair preview ====="]
        todo = 0
        for code in pairs:
            exists = os.path.isfile(os.path.join(out_dir, ref_idx[code]))
            if exists and not overwrite:
                tag = "SKIP (output exists)"
            else:
                tag = "OVERWRITE" if exists else "RUN"
                todo += 1
            lines.append(f"  {code}  [{tag}]\n      ref:     {ref_idx[code]}\n      foreign: {for_idx[code]}")
        for code in sorted(set(ref_idx) - set(for_idx)):
            lines.append(f"  {code}  [NO FOREIGN MATCH]  {ref_idx[code]}")
        for code in sorted(set(for_idx) - set(ref_idx)):
            lines.append(f"  {code}  [NO REFERENCE MATCH]  {for_idx[code]}")
        for name in ref_none:
            lines.append(f"  [IGNORED - no SxxExx code] ref: {name}")
        for name in for_none:
            lines.append(f"  [IGNORED - no SxxExx code] foreign: {name}")
        if ref_dupes or for_dupes:
            lines.append(f"  [ERROR] Duplicate episode codes: {', '.join(sorted(set(ref_dupes + for_dupes)))} "
                         "- the batch will refuse to run until each code appears once per folder.")
        lines.append(f"Matched: {len(pairs)} | Will process: {todo} | Skipped: {len(pairs) - todo}\n")
        self._append_log("\n".join(lines) + "\n")
        self.status_var.set(f"Preview: {todo} of {len(pairs)} matched episode(s) will run")

    def _run_process(self, cmd):
        code = -1
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,   # no terminal prompts: the engine falls back to auto-detection
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=os.path.dirname(cmd[2]) or None,  # cmd = [python, -u, script, ...]
                env=self._build_env(),
                **self._group_kwargs(),
            )
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.wait()
            code = self.proc.returncode
            self.log_queue.put(f"\n[process exited with code {code}]\n")
        except Exception as e:
            self.log_queue.put(f"\n[failed to launch process: {e}]\n")
        finally:
            self.proc = None
            self.log_queue.put(("__DONE__", code))

    @staticmethod
    def _group_kwargs():
        if platform.system() == "Windows":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _kill_tree(self, proc):
        """Stop the job and everything it started (the batch runner spawns one engine per episode)."""
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _on_stop(self):
        if self.proc is not None:
            try:
                self._kill_tree(self.proc)
                self._append_log("\n[stop requested - terminating job and any child processes]\n")
            except Exception as e:
                self._append_log(f"\n[could not terminate process: {e}]\n")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple):
                    if item[0] == "__DONE__":
                        self.status_var.set("Idle")
                        self.btn_run.configure(state="normal")
                        self.btn_run_batch.configure(state="normal")
                        self.btn_stop.configure(state="disabled")
                        self.progress.stop()
                        self._update_ready_state()
                    elif item[0] == "__RECHECK_DEPS__":
                        self._refresh_dependencies()
                        self.btn_install_mkv.configure(
                            state="disabled" if self.dep_status.get("mkvmerge") else "normal")
                else:
                    m = BATCH_PROGRESS_RE.match(item.strip())
                    if m:
                        self.status_var.set(f"Batch {m.group(1)}/{m.group(2)}: {m.group(3)}")
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _append_log(self, text):
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    # ========================================================== settings ==
    def _collect_settings(self):
        data = {"script": self.e_script.get(), "mkvmerge_dir": self.extra_path_dir.get()}
        for spec in FIELDS:
            data[spec["key"]] = self.vars[spec["key"]].get()
        return data

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._collect_settings(), fh, indent=2)
            self.status_var.set(f"Settings saved to {os.path.basename(SETTINGS_FILE)}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Could not save settings: {e}")

    def _load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return

        saved_script = data.get("script") or ""
        if saved_script and os.path.basename(saved_script) != ENGINE_NAME and os.path.isfile(SCRIPT_DEFAULT):
            self._append_log(f"[Saved settings pointed at {os.path.basename(saved_script)}; "
                             f"switched to {ENGINE_NAME} next to this GUI.]\n")
        elif saved_script:
            self.e_script.set(saved_script)
        self.extra_path_dir.set(data.get("mkvmerge_dir", ""))

        for spec in FIELDS:
            key = spec["key"]
            if key in data and key in self.vars:
                try:
                    self.vars[key].set(data[key])
                except Exception:
                    pass

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno(APP_TITLE, "A job is still running. Stop it and quit?"):
                return
            try:
                self._kill_tree(self.proc)
            except Exception:
                pass
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    app = AVSyncGUI()
    app.mainloop()
