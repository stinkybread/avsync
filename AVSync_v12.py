import cv2
import os
import shutil
import subprocess
from tqdm import tqdm
import glob
import concurrent.futures
import numpy as np
import argparse
import re
import tempfile
import sys
from scipy.io import wavfile
import time
import json
import logging
import csv
import platform
import ctypes
import pickle
import hashlib

# --- DUAL LOGGING SYSTEM ---
# File logger: detailed output to log file
# Console logger: minimal output with progress indicators

class FileFormatter(logging.Formatter):
    """Detailed formatter for log file"""
    def __init__(self):
        super().__init__('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class ConsoleFormatter(logging.Formatter):
    """Minimal formatter for console - ASCII safe"""
    
    ANSI_COLORS = {
        'RESET': '\033[0m', 'RED': '\033[31m', 'GREEN': '\033[32m',
        'YELLOW': '\033[33m', 'CYAN': '\033[36m', 'BOLD': '\033[1m', 'DIM': '\033[2m',
    }
    NO_COLORS = {k: '' for k in ANSI_COLORS.keys()}

    def __init__(self):
        super().__init__('%(message)s')
        self.use_colors = self._init_colors()

    def _init_colors(self):
        """Initialize color support"""
        if os.getenv('NO_COLOR'):
            self.colors = self.NO_COLORS
            return False
        if platform.system() == 'Windows':
            try:
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                if handle and handle != -1:
                    mode = ctypes.c_uint32()
                    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                        self.colors = self.ANSI_COLORS
                        return True
            except: pass
            self.colors = self.NO_COLORS
            return False
        self.colors = self.ANSI_COLORS
        return True

    def format(self, record):
        msg = super().format(record)
        c = self.colors
        
        # Only show important messages on console
        if record.levelno >= logging.ERROR:
            return f"{c['RED']}{c['BOLD']}[ERROR] {msg}{c['RESET']}"
        elif record.levelno >= logging.WARNING:
            return f"{c['YELLOW']}[WARN] {msg}{c['RESET']}"
        elif record.levelno >= logging.INFO:
            # Stage headers
            if msg.startswith("=====") or msg.startswith("---"):
                return f"{c['CYAN']}{c['BOLD']}{msg}{c['RESET']}"
            # Success messages
            if "[OK]" in msg or "SUCCESS" in msg.upper():
                return f"{c['GREEN']}{msg}{c['RESET']}"
            # Cache messages
            if "[CACHE]" in msg:
                return f"{c['DIM']}{msg}{c['RESET']}"
            return msg
        return msg


# Custom filter to suppress verbose messages on console
class ConsoleFilter(logging.Filter):
    """Filter out verbose messages from console output"""
    
    # Class variable to control warning display (set via --show-warnings)
    show_warnings = False
    
    VERBOSE_PATTERNS = [
        "Iter ",              # Individual iteration details
        "-> Running:",        # FFmpeg command starts
        "-> Completed:",      # FFmpeg command completions
        "Speed=",             # Speed adjustments
        "Duration=",          # Duration checks
        "MinRefDur Filter",
        "MaxDiff Filter",
        "Skipping visual anchor",
        "Process Segment",
        "Trim Segment",
        "Create Silence",
        "Add Silence Pad",
        "segment duration",
        # Additional patterns for clean progress bar (no scrolling)
        "-> Segment ",        # Segment processing details
        "Segment ",           # All segment-related messages
        "Target=",            # Target duration info
        "Achieved target",    # Success messages per segment
        "First segment:",     # First segment adjustments
        "Last segment:",      # Last segment adjustments
        "Adjusting speed",    # Speed adjustment debug
        "silence pad",        # Silence padding messages
    ]
    
    def filter(self, record):
        msg = record.getMessage()
        # Errors always shown
        if record.levelno >= logging.ERROR:
            return True
        # Warnings only shown if --show-warnings is set
        if record.levelno >= logging.WARNING:
            return ConsoleFilter.show_warnings
        # Filter out verbose patterns for INFO level
        for pattern in self.VERBOSE_PATTERNS:
            if pattern in msg:
                return False
        return True


def setup_logging(log_file=None, verbose=False):
    """
    Set up dual logging: file (detailed) + console (minimal).
    
    Args:
        log_file: Path to log file. If None, uses default based on output.
        verbose: If True, console shows all messages (like old behavior)
    
    Returns:
        tuple: (logger, log_file_path)
    """
    logger = logging.getLogger('avsync')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler - always detailed
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
    
    # Console handler - minimal by default
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    if not verbose:
        console_handler.addFilter(ConsoleFilter())
    logger.addHandler(console_handler)
    
    return logger, log_file


# Global logger placeholder
logger = None


def get_log_path(output_video):
    """Generate log file path based on output video"""
    output_dir = os.path.dirname(os.path.abspath(output_video))
    output_name = os.path.splitext(os.path.basename(output_video))[0]
    return os.path.join(output_dir, f"{output_name}_avsync.log")


# --- PROGRESS BAR HELPERS ---
class SegmentProgressBar:
    """Progress bar for segment processing"""
    
    def __init__(self, total_segments, desc="Processing segments"):
        self.pbar = tqdm(
            total=total_segments,
            desc=desc,
            unit="seg",
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.current = 0
        self.warnings = 0
        self.corrections = 0
    
    def update(self, correction_applied=False):
        """Update progress bar"""
        self.current += 1
        if correction_applied:
            self.corrections += 1
        self.pbar.update(1)
    
    def set_warning(self):
        """Increment warning counter"""
        self.warnings += 1
    
    def close(self):
        """Close progress bar and show summary"""
        self.pbar.close()
        if self.corrections > 0:
            print(f"  -> {self.corrections} segments needed final correction")


# --- END OF LOGGING SYSTEM ---

# --- CACHING FUNCTIONS ---
def get_file_hash(filepath):
    """Calculate SHA256 hash of file for cache validation"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_cache_key(args):
    """
    Generate cache key based on inputs and detection parameters.
    Excludes adjustment parameters that don't affect detection.
    """
    cache_params = {
        'ref_video': get_file_hash(args.ref_video) if os.path.exists(args.ref_video) else None,
        'foreign_video': get_file_hash(args.foreign_video) if os.path.exists(args.foreign_video) else None,
        'scene_threshold': args.scene_threshold,
        'match_threshold': args.match_threshold,
        'similarity_threshold': args.similarity_threshold,
        'db_threshold': args.db_threshold,
        'min_segment_duration': args.min_segment_duration,
        'ref_lang': args.ref_lang,
        'foreign_lang': args.foreign_lang,
    }
    
    cache_str = json.dumps(cache_params, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()


def get_cache_path(args):
    """Get cache file path based on output video name"""
    output_dir = os.path.dirname(os.path.abspath(args.output_video))
    output_name = os.path.splitext(os.path.basename(args.output_video))[0]
    cache_dir = os.path.join(output_dir, f".avsync_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = get_cache_key(args)
    cache_file = os.path.join(cache_dir, f"{output_name}_{cache_key[:16]}.pkl")
    return cache_file


def save_checkpoint(cache_path, visual_anchors_details):
    """Save checkpoint data to disk"""
    checkpoint_data = {
        'version': 12,
        'visual_anchors_details': visual_anchors_details,
        'timestamp': time.time()
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"[CACHE] Checkpoint saved to {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint(cache_path):
    """Load checkpoint data from disk"""
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if checkpoint_data.get('version') != 12:
            logger.warning(f"[CACHE] Version mismatch, ignoring cache")
            return None
            
        logger.info(f"[CACHE] Loaded checkpoint from {cache_path}")
        cache_age = time.time() - checkpoint_data.get('timestamp', 0)
        logger.info(f"[CACHE] Age: {cache_age/3600:.1f} hours")
        
        return checkpoint_data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None
# --- END CACHING FUNCTIONS ---


# --- FORCED SYNC POINT FUNCTIONS ---
def parse_timestamp_to_seconds(timestamp_str):
    """
    Parse HH:MM:SS:MS timestamp format to seconds.
    
    Format: HH:MM:SS:MS (e.g., "00:01:30:500" = 1 minute, 30 seconds, 500 milliseconds)
    
    Returns:
        float: Time in seconds, or None if parsing fails.
    """
    timestamp_str = timestamp_str.strip()
    parts = timestamp_str.split(':')
    
    if len(parts) != 4:
        logger.warning(f"Invalid timestamp format '{timestamp_str}' (expected HH:MM:SS:MS)")
        return None
    
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(parts[3])
        
        if minutes >= 60 or seconds >= 60 or milliseconds >= 1000:
            logger.warning(f"Invalid timestamp values in '{timestamp_str}' (MM<60, SS<60, MS<1000)")
            return None
        
        if hours < 0 or minutes < 0 or seconds < 0 or milliseconds < 0:
            logger.warning(f"Timestamp values must be non-negative in '{timestamp_str}'")
            return None
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
        
    except ValueError as e:
        logger.warning(f"Could not parse timestamp '{timestamp_str}': {e}")
        return None


def parse_force_sync_points(force_sync_str):
    """
    Parse forced sync points from command line argument.
    
    Format: "HH:MM:SS:MS>HH:MM:SS:MS,HH:MM:SS:MS>HH:MM:SS:MS,..."
    Example: "00:00:10:500>00:00:10:800,00:02:00:300>00:02:01:000"
    
    The '>' separates reference time from foreign time.
    Multiple sync points are comma-separated.
    
    Returns:
        List of tuples: [(ref_time_seconds, foreign_time_seconds), ...]
        Returns empty list if parsing fails or input is None/empty.
    """
    if not force_sync_str:
        return []
    
    sync_points = []
    pairs = force_sync_str.strip().split(',')
    
    for i, pair in enumerate(pairs):
        pair = pair.strip()
        if not pair:
            continue
        
        if '>' not in pair:
            logger.warning(f"Invalid sync point format '{pair}' (expected 'HH:MM:SS:MS>HH:MM:SS:MS'). Skipping.")
            continue
        
        try:
            parts = pair.split('>')
            if len(parts) != 2:
                logger.warning(f"Invalid sync point format '{pair}' (expected exactly 2 timestamps separated by '>'). Skipping.")
                continue
            
            ref_time = parse_timestamp_to_seconds(parts[0])
            foreign_time = parse_timestamp_to_seconds(parts[1])
            
            if ref_time is None or foreign_time is None:
                logger.warning(f"Skipping sync point '{pair}' due to timestamp parsing errors.")
                continue
            
            sync_points.append((ref_time, foreign_time))
            logger.debug(f"  Parsed forced sync point {i+1}: Ref={ref_time:.3f}s, Foreign={foreign_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Could not parse sync point '{pair}': {e}. Skipping.")
            continue
    
    return sync_points


def inject_forced_sync_points(visual_anchors_details, forced_sync_points):
    """
    Inject forced sync points into visual_anchors_details list.
    
    Forced sync points are given synthetic filenames starting with 'FORCED_SYNC_' 
    to distinguish them from scene-detected anchors. This prefix is used later
    to ensure they bypass filtering.
    
    Args:
        visual_anchors_details: List of tuples (ref_filename, foreign_filename, ref_time, foreign_time)
        forced_sync_points: List of tuples (ref_time, foreign_time)
    
    Returns:
        New list with forced sync points injected, sorted by reference time.
    """
    if not forced_sync_points:
        return visual_anchors_details
    
    logger.info(f"--- Injecting {len(forced_sync_points)} Forced Sync Points ---")
    
    # Create a mutable copy
    combined = list(visual_anchors_details) if visual_anchors_details else []
    
    # Add forced sync points with synthetic filenames
    for i, (ref_time, foreign_time) in enumerate(forced_sync_points):
        synthetic_ref_name = f"FORCED_SYNC_{i+1:03d}_ref"
        synthetic_foreign_name = f"FORCED_SYNC_{i+1:03d}_foreign"
        combined.append((synthetic_ref_name, synthetic_foreign_name, ref_time, foreign_time))
        logger.info(f"  > Injected forced sync point {i+1}: Ref={format_time(ref_time)} ({ref_time:.3f}s) -> Foreign={format_time(foreign_time)} ({foreign_time:.3f}s)")
    
    # Sort by reference time (index 2 in the tuple)
    combined.sort(key=lambda x: x[2])
    
    logger.info(f"  -> Total anchors after injection: {len(combined)} ({len(forced_sync_points)} forced + {len(visual_anchors_details) if visual_anchors_details else 0} detected)")
    
    return combined
# --- END FORCED SYNC POINT FUNCTIONS ---



# --- Try Importing Image Similarity Libs ---
try:
    import imagehash
    from PIL import Image
    similarity_libs_available = True
except ImportError as e:
    print(f"Warning: 'imagehash' or 'Pillow' not found. Similarity filtering will be skipped. Error: {e}")
    similarity_libs_available = False

# --- Constants ---
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720
DEFAULT_DB_THRESHOLD = -40.0
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2
MIN_ATEMPO = 0.5
MAX_ATEMPO = 100.0
# MIN_ALLOWED_REF_DURATION_S = 5.0 # Replaced by argument --min_segment_duration
DEFAULT_MIN_SEGMENT_DURATION = 5.0 # New default value
MAX_ALLOWED_DURATION_PERCENT_DIFF = 6.0 # Max % difference allowed between ref/foreign segment durations
MIN_DELAY_S = 0.001 # Minimum delay to apply padding
DEFAULT_REF_LANG = "eng"
DEFAULT_FOREIGN_LANG = "foreign" # Changed from "hin"
DEFAULT_MUX_ACODEC = "aac"
DEFAULT_MUX_ABITRATE = "192k"
QC_IMAGE_HEIGHT = 720
FFMPEG_EXEC = None
FFPROBE_EXEC = None
MATCH_WINDOW_PERCENT = 0.06 # Percentage of ref video duration for match search window

# Logger will be initialized in main() with proper file output
# Use a temporary logger for early errors
logger = logging.getLogger('avsync')
logger.setLevel(logging.INFO)
if not logger.handlers:
    _temp_handler = logging.StreamHandler(sys.stdout)
    _temp_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_temp_handler)

# --- Utility & FFmpeg/FFprobe Functions ---
def find_executable(name):
    """Finds an executable in bundled resources first, then system PATH."""
    # Check if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        # sys._MEIPASS points to the temporary extraction directory at runtime
        bundle_dir = sys._MEIPASS
        ext = '.exe' if platform.system() == 'Windows' else ''
        # In PyInstaller COLLECT mode, binaries are directly in the extraction directory
        bundled_path = os.path.join(bundle_dir, 'bin', f'{name}{ext}')
        if os.path.exists(bundled_path):
            logger.debug(f"Found bundled executable '{name}' at: {bundled_path}")
            return bundled_path

    # Fall back to system PATH
    exec_path = shutil.which(name)
    if exec_path:
        logger.debug(f"Found executable '{name}' at: {exec_path}")
        return exec_path
    else:
        logger.error(f"'{name}' command not found in bundled resources or system PATH.")
        return None

def run_ffmpeg(cmd_list, desc="FFmpeg Task", verbose_success=False, capture_stderr=False):
    """Runs an FFmpeg command with logging and error handling."""
    global FFMPEG_EXEC
    if not FFMPEG_EXEC:
        logger.error("FFmpeg path not set.")
        return False, "" if capture_stderr else False
    cmd_list[0] = FFMPEG_EXEC
    logger.info(f"  -> Running: {desc}...")
    start_time = time.time()
    stderr_output = ""
    try:
        # Use Popen for better handling of large stderr, prevent potential deadlocks
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', errors='ignore', startupinfo=None)
        stdout, stderr = process.communicate() # Wait for process to finish
        returncode = process.returncode
        elapsed_time = time.time() - start_time

        if capture_stderr:
            stderr_output = stderr

        if returncode != 0:
            logger.error(f"\nERROR: {desc} failed! (Exit code: {returncode}, Time: {elapsed_time:.2f}s)")
            logger.error(f"  FFmpeg Command: {' '.join(cmd_list)}")
            # Log only last few lines of stderr if it's huge
            stderr_lines = stderr.strip().splitlines()
            max_lines = 20
            log_stderr = "\n".join(stderr_lines[-max_lines:])
            if len(stderr_lines) > max_lines:
                 log_stderr = f"(Showing last {max_lines} lines)\n" + log_stderr
            logger.error(f"  FFmpeg Stderr:\n{log_stderr}")
            return False, stderr_output if capture_stderr else False
        else:
            logger.info(f"  -> Completed: {desc} (Time: {elapsed_time:.2f}s)")
            if verbose_success and stderr:
                 warnings = [line for line in stderr.splitlines() if 'warning' in line.lower()]
                 if warnings:
                      logger.warning(f"FFmpeg warnings during {desc}:\n" + "\n".join(warnings))
            return True, stderr_output if capture_stderr else True
    except FileNotFoundError:
        logger.error(f"'{cmd_list[0]}' not found. Check installation/PATH.")
        return False, "" if capture_stderr else False
    except Exception as e:
        logger.error(f"ERROR: Unexpected error running {desc}: {e}", exc_info=True)
        return False, "" if capture_stderr else False

def get_stream_info(video_path):
    """Gets stream information using ffprobe."""
    global FFPROBE_EXEC
    if not FFPROBE_EXEC:
        logger.error("FFprobe path not set.")
        return None
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    try:
        # Added timeout to prevent hangs on corrupted files
        result = subprocess.run([FFPROBE_EXEC, '-v', 'error', '-print_format', 'json', '-show_streams', video_path],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', timeout=60) # 60 second timeout
        return json.loads(result.stdout).get('streams', [])
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out processing {os.path.basename(video_path)}")
        return None
    except Exception as e:
        logger.error(f"Failed to get stream info for {os.path.basename(video_path)}: {e}", exc_info=False) # Less verbose on failure
        return None

def find_audio_stream_index_by_lang(streams, lang_code):
    """Finds the first audio stream matching the given language code."""
    if not streams:
        return None
    logger.debug(f"Searching for language '{lang_code}' in streams...")
    stream_index = None
    found_match = False
    # Prioritize exact language match
    for stream in streams:
        if stream.get('codec_type') == 'audio':
            idx = stream.get('index') # This is the ABSOLUTE index
            tags = stream.get('tags', {})
            lang = tags.get('language', 'und') # 'und' for undetermined
            if idx is not None and lang.lower() == lang_code.lower():
                stream_index = idx
                logger.debug(f"    ^ Found matching language tag at index {idx}")
                found_match = True
                break # Take the first match

    if found_match:
        return stream_index

    # If no exact match, fall back to the first audio stream found
    logger.warning(f"Language tag '{lang_code}' not found. Looking for first available audio stream.")
    for stream in streams:
         if stream.get('codec_type') == 'audio':
              idx = stream.get('index') # Absolute index
              if idx is not None:
                   logger.warning(f"Falling back to first audio stream found (index: {idx}).")
                   return idx

    logger.error("No audio streams found at all.")
    return None

def get_audio_stream_details(video_path):
    """Gets detailed information about audio streams in a video file."""
    streams = get_stream_info(video_path)
    if not streams:
        return []

    audio_streams = []
    for i, stream in enumerate(streams):
        if stream.get('codec_type') == 'audio':
            stream_info = {
                'index': stream.get('index'), # Absolute index
                'codec': stream.get('codec_name', 'unknown'),
                'channels': stream.get('channels', 0),
                'sample_rate': stream.get('sample_rate', 'unknown'),
                'bit_rate': stream.get('bit_rate', 'unknown'),
                'language': stream.get('tags', {}).get('language', 'und'),
                'title': stream.get('tags', {}).get('title', '')
            }
            audio_streams.append(stream_info)

    return audio_streams

def prompt_user_for_audio_stream(video_path, stream_type="foreign"):
    """Prompts the user to select an audio stream from a video file."""
    audio_streams = get_audio_stream_details(video_path)
    if not audio_streams:
        logger.error(f"No audio streams found in {os.path.basename(video_path)}")
        return None

    print(f"\n--- Available {stream_type.capitalize()} Audio Streams in {os.path.basename(video_path)} ---")
    print("{:<5} {:<10} {:<8} {:<12} {:<10} {:<12}".format(
        "Sel#", "Stream#", "Language", "Codec", "Channels", "Sample Rate"))
    print("-" * 70)

    for i, stream in enumerate(audio_streams):
        print("{:<5} {:<10} {:<8} {:<12} {:<10} {:<12}".format(
            i,  # Selection number (0-based)
            stream['index'],  # Actual stream index (absolute)
            stream['language'],
            stream['codec'],
            stream['channels'],
            stream['sample_rate']))

    # Prompt user for selection
    while True:
        try:
            selection = input(f"\nSelect {stream_type} audio stream by Sel# (or press Enter for auto-detection): ")
            if not selection.strip():
                return None  # Auto-detection

            selected_idx = int(selection)
            # Check if the selection is valid based on our displayed numbers
            if 0 <= selected_idx < len(audio_streams):
                # Return the actual stream index, not our selection number
                return audio_streams[selected_idx]['index'] # Return the absolute index
            else:
                print(f"Error: Selection number {selected_idx} is out of range. Please choose 0-{len(audio_streams)-1}.")
                continue
        except ValueError:
            print("Error: Please enter a valid number.")

def format_time(seconds):
    """Formats seconds into HH:MM:SS:ms format."""
    if seconds is None or seconds < 0:
        return "00:00:00:000"
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"

def get_file_duration(file_path, media_type='audio'):
    """Get the duration of an audio or video file using ffprobe."""
    global FFPROBE_EXEC
    if not FFPROBE_EXEC:
        logger.error(f"FFprobe path not set for duration check of {os.path.basename(file_path)}.")
        return None
    if not os.path.exists(file_path):
        logger.error(f"File not found for duration check: {file_path}")
        return None

    try:
        probe_cmd = [FFPROBE_EXEC, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        # Added timeout
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=30)
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        logger.warning(f"ffprobe returned empty duration for {os.path.basename(file_path)}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out getting duration for {os.path.basename(file_path)}")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed to get duration for {os.path.basename(file_path)}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error getting duration for {os.path.basename(file_path)}: {e}", exc_info=False)
        return None

# --- Image Pairing Stage Functions ---
def extract_frames_ffmpeg(video_path, output_folder, scene_threshold):
    """Extracts scene change frames using FFmpeg's scene detection."""
    os.makedirs(output_folder, exist_ok=True)
    parsed_pts_times = []
    vid_name = os.path.basename(video_path)

    # Command using scene detection filter
    ffmpeg_command = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
        "-i", video_path,
        "-vf", f"select='gt(scene,{scene_threshold})',showinfo", # Select frames where scene score > threshold, showinfo logs PTS
        "-vsync", "vfr", # Variable frame rate to capture exact frames
        "-q:v", "2", # High quality PNG output
        os.path.join(output_folder, "frame_%06d.png")
    ]

    logger.info(f"--- Extracting Scene Frames (threshold={scene_threshold}) from {vid_name} ---")
    success, stderr_output = run_ffmpeg(ffmpeg_command, f"Extract Frames ({vid_name})", verbose_success=False, capture_stderr=True)

    if success and stderr_output:
        # Regex to find pts_time in showinfo output
        pts_time_re = re.compile(r'n:\s*\d+\s+pts:\s*\d+\s+pts_time:(\d+\.?\d*)')
        lines = stderr_output.splitlines()
        for line in lines:
            # Filter for the specific showinfo log lines
            if '[Parsed_showinfo' in line and 'pts_time:' in line:
                 match = pts_time_re.search(line)
                 if match:
                     try:
                         parsed_pts_times.append(float(match.group(1)))
                     except (ValueError, IndexError):
                         logger.warning(f"Could not parse pts_time from line: {line}")
    elif not success:
        logger.error(f"  Frame extraction command failed for {vid_name}.")
        return False, [], []

    # Verify extracted frames match timestamps
    frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.png")))
    frame_count = len(frame_files)
    timestamp_count = len(parsed_pts_times)

    if frame_count == 0 or timestamp_count == 0:
        logger.error(f"  Frame extraction yielded zero frames or timestamps for {vid_name}.")
        return False, [], []

    final_count = 0
    if frame_count != timestamp_count:
         logger.warning(f"  Frame/Timestamp count mismatch ({frame_count} frames vs {timestamp_count} timestamps) for {vid_name}. Using minimum.")
         final_count = min(frame_count, timestamp_count)
         # Trim lists to the minimum count to maintain correspondence
         parsed_pts_times = parsed_pts_times[:final_count]
         frame_files = frame_files[:final_count]
    else:
         final_count = frame_count

    logger.info(f"> Extracted {final_count} scene frames from {vid_name}")
    return True, [os.path.basename(f) for f in frame_files], parsed_pts_times

def process_image_pair_for_match(ref_img_name, foreign_image_list, ref_extract_folder, foreign_extract_folder, match_threshold):
    """Compares one reference image against a list of foreign images using template matching."""
    ref_img_path = os.path.join(ref_extract_folder, ref_img_name)
    try:
        # Read reference image, convert to grayscale, and resize for consistent comparison
        ref_frame_orig = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        if ref_frame_orig is None:
            logger.warning(f"Could not read reference image: {ref_img_name}"); return None
        ref_frame_comp = cv2.resize(ref_frame_orig, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        if ref_frame_comp is None or ref_frame_comp.size == 0:
             logger.warning(f"Failed to resize reference image: {ref_img_name}"); return None
    except Exception as e:
        logger.error(f"Error processing reference image {ref_img_name}: {e}", exc_info=False); return None

    best_match_foreign_name = None
    best_score = -1.0 # Initialize score below any possible match

    # Iterate through potential foreign matches (this list might be pre-filtered)
    for foreign_img_name in foreign_image_list:
        foreign_img_path = os.path.join(foreign_extract_folder, foreign_img_name)
        try:
            # Read, grayscale, and resize foreign image
            foreign_frame_orig = cv2.imread(foreign_img_path, cv2.IMREAD_GRAYSCALE)
            if foreign_frame_orig is None: continue # Skip if image can't be read
            foreign_frame_comp = cv2.resize(foreign_frame_orig, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
            if foreign_frame_comp is None or foreign_frame_comp.size == 0: continue # Skip if resize fails

            # Perform template matching
            # TM_CCOEFF_NORMED gives a score between -1 and 1, where 1 is a perfect match
            result = cv2.matchTemplate(foreign_frame_comp, ref_frame_comp, cv2.TM_CCOEFF_NORMED)
            _minVal, maxVal, _minLoc, _maxLoc = cv2.minMaxLoc(result) # We only need the max value

            # Update best match if current score is higher
            if maxVal > best_score:
                best_score = maxVal
                best_match_foreign_name = foreign_img_name
        except Exception as e:
            # Log error but continue checking other foreign images
            logger.debug(f"Error comparing {ref_img_name} with {foreign_img_name}: {e}", exc_info=False)
            continue

    # Return the best match only if its score meets the threshold
    if best_match_foreign_name is not None and best_score >= match_threshold:
        return (best_match_foreign_name, best_score)
    else:
        return None # No match found above the threshold

def filter_similar_ref_images(initial_matches_with_times, ref_extract_folder, similarity_threshold):
    """Filters out reference frames that are too visually similar using perceptual hashing."""
    global similarity_libs_available
    if not similarity_libs_available:
        logger.info("  Skipping similarity filtering: imagehash/Pillow libraries not available.")
        return initial_matches_with_times
    if similarity_threshold < 0:
        logger.info(f"  Skipping similarity filtering: threshold ({similarity_threshold}) is negative.")
        return initial_matches_with_times
    if not initial_matches_with_times:
        return {} # Return empty if no initial matches

    logger.info(f"--- Filtering Similar Reference Frames (pHash Threshold: {similarity_threshold}) ---")
    start_time = time.time()

    # Sort reference frame names numerically based on frame index (e.g., frame_000001.png)
    frame_num_re = re.compile(r'frame_(\d+).png')
    try:
        ref_names_sorted = sorted(
            initial_matches_with_times.keys(),
            key=lambda name: int(frame_num_re.search(name).group(1))
        )
    except Exception as e:
        logger.warning(f"Could not sort reference frames numerically, using default sort. Error: {e}")
        ref_names_sorted = sorted(initial_matches_with_times.keys())


    hashes = {} # Store phash -> (ref_name, file_size)
    to_remove_ref_names = set() # Keep track of reference frames to discard

    for ref_name in tqdm(ref_names_sorted, desc="  Filtering Similar Refs", unit="frame", ncols=100, leave=False):
        if ref_name in to_remove_ref_names:
            continue # Skip if already marked for removal

        ref_path = os.path.join(ref_extract_folder, ref_name)
        if not os.path.exists(ref_path):
             logger.warning(f"Reference frame {ref_name} not found, skipping similarity check.")
             to_remove_ref_names.add(ref_name)
             continue

        try:
            # Get file size and compute perceptual hash
            current_size = os.path.getsize(ref_path)
            with Image.open(ref_path) as img_file:
                img_hash = imagehash.phash(img_file)
        except Exception as e:
            logger.warning(f"Could not process {ref_name} for similarity hashing: {e}")
            continue # Skip this frame if hashing fails

        found_similar = False
        hashes_to_update = {} # Store updates for the current hash if it replaces an existing one
        hashes_to_delete = [] # Hashes to remove if replaced by a larger frame

        # Compare current hash with existing stored hashes
        # Create a copy of items to allow modification during iteration
        for existing_hash, (existing_ref_name, existing_size) in list(hashes.items()):
             # Skip comparison if the existing frame was already marked for removal
             if existing_ref_name in to_remove_ref_names:
                 hashes_to_delete.append(existing_hash) # Mark the old hash for deletion
                 continue

             # Calculate Hamming distance between hashes (lower means more similar)
             hash_diff = img_hash - existing_hash

             if hash_diff < similarity_threshold:
                 # Found a similar frame
                 found_similar = True
                 # Decide which frame to keep: prefer the one with larger file size (potentially higher quality/detail)
                 if current_size >= existing_size:
                     # Current frame is better or equal, mark existing for removal and update hash mapping
                     logger.debug(f"    '{ref_name}' ({current_size}b) replacing similar '{existing_ref_name}' ({existing_size}b), diff={hash_diff}")
                     to_remove_ref_names.add(existing_ref_name)
                     hashes_to_update[img_hash] = (ref_name, current_size) # Map new hash to this frame
                     hashes_to_delete.append(existing_hash) # Mark old hash for deletion
                 else:
                     # Existing frame is better, mark current frame for removal
                     logger.debug(f"    '{ref_name}' ({current_size}b) removed due to similarity with '{existing_ref_name}' ({existing_size}b), diff={hash_diff}")
                     to_remove_ref_names.add(ref_name)
                 break # Stop comparing once a similar frame is found

        # Clean up hashes map after comparisons
        for h_del in hashes_to_delete:
             if h_del in hashes:
                 del hashes[h_del]
        hashes.update(hashes_to_update) # Apply updates

        # If no similar frame was found and this frame wasn't marked for removal, add its hash
        if not found_similar and ref_name not in to_remove_ref_names:
            hashes[img_hash] = (ref_name, current_size)

    # Create the final dictionary excluding the removed frames
    filtered_matches = {
        ref_name: data
        for ref_name, data in initial_matches_with_times.items()
        if ref_name not in to_remove_ref_names
    }

    removed_count = len(initial_matches_with_times) - len(filtered_matches)
    elapsed_time = time.time() - start_time
    logger.info(f"  -> Similarity filtering complete. Removed {removed_count} potentially redundant pairs ({elapsed_time:.2f}s).")
    return filtered_matches

def filter_temporal_inconsistency(matches_after_similarity):
    """Filters matches where the foreign frame order doesn't match the reference frame order."""
    if not matches_after_similarity:
        return [] # Return empty list if no matches
    logger.info("--- Filtering Temporal Inconsistencies ---")
    start_time = time.time()

    # Extract items and sort them based on the reference frame number
    match_items = list(matches_after_similarity.items())
    frame_num_re = re.compile(r'frame_(\d+).png')
    try:
        # Sort by reference frame number extracted from filename
        match_items.sort(key=lambda item: int(frame_num_re.search(item[0]).group(1)))
    except Exception as e:
        logger.warning(f"Could not sort matches numerically by reference frame, using timestamp sort. Error: {e}")
        # Fallback sort by reference timestamp if filename parsing fails
        match_items.sort(key=lambda item: item[1][1]) # Sort by ref_time (index 1 of tuple value)

    filtered_list = []
    last_accepted_foreign_num = -1 # Track the frame number of the last accepted foreign match
    removed_count = 0

    for ref_name, (foreign_name, ref_time, foreign_time) in match_items:
        # Extract frame number from the foreign filename
        foreign_match = frame_num_re.search(foreign_name)
        if not foreign_match:
            # If filename format is unexpected, keep the match but log a warning
            logger.warning(f"Could not parse frame number from foreign image '{foreign_name}'. Keeping match.")
            filtered_list.append((ref_name, foreign_name, ref_time, foreign_time))
            continue

        try:
            current_foreign_num = int(foreign_match.group(1))
        except ValueError:
             logger.warning(f"Could not convert foreign frame number to int for '{foreign_name}'. Keeping match.")
             filtered_list.append((ref_name, foreign_name, ref_time, foreign_time))
             continue

        # Core logic: Check if the current foreign frame number is >= the last accepted one
        # This ensures that the sequence of matched foreign frames is monotonically increasing
        if current_foreign_num >= last_accepted_foreign_num:
            filtered_list.append((ref_name, foreign_name, ref_time, foreign_time))
            last_accepted_foreign_num = current_foreign_num # Update the last accepted number
        else:
            # Temporal inconsistency detected (e.g., Ref frame 5 matches Foreign 10, Ref 6 matches Foreign 8)
            logger.debug(f"    Temporal inconsistency: Ref '{ref_name}' -> Foreign '{foreign_name}' ({current_foreign_num}) is earlier than last accepted ({last_accepted_foreign_num}). Removing.")
            removed_count += 1

    elapsed_time = time.time() - start_time
    logger.info(f"  -> Temporal filtering complete. Removed {removed_count} inconsistent pairs ({elapsed_time:.2f}s).")
    # Return a list of tuples: [(ref_filename, foreign_filename, ref_time, foreign_time)]
    return filtered_list


def run_image_pairing_stage(ref_video_path, foreign_video_path, temp_dir, scene_threshold, match_threshold, similarity_threshold):
    """Orchestrates the entire image pairing stage."""
    logger.info("\n===== \1 =====")
    stage_start_time = time.time()

    # Define paths for extracted frames
    ref_extract_path = os.path.join(temp_dir, "Extracted_Reference")
    foreign_extract_path = os.path.join(temp_dir, "Extracted_Foreign")

    # --- Step 1: Extract Frames ---
    ref_extract_ok, ref_filenames, ref_timestamps_list = extract_frames_ffmpeg(ref_video_path, ref_extract_path, scene_threshold)
    if not ref_extract_ok:
        logger.error("Failed to extract reference frames.")
        return None

    foreign_extract_ok, foreign_filenames, foreign_timestamps_list = extract_frames_ffmpeg(foreign_video_path, foreign_extract_path, scene_threshold)
    if not foreign_extract_ok:
        logger.error("Failed to extract foreign frames.")
        return None

    # --- Step 1.5: Calculate Search Window ---
    logger.info("--- Calculating Frame Match Search Window ---")
    ref_duration = get_file_duration(ref_video_path, media_type='video')
    if ref_duration is None or ref_duration <= 0:
        logger.warning("Could not determine reference video duration or duration is zero. Frame matching will compare against ALL foreign frames.")
        match_search_window_seconds = float('inf') # Effectively disable windowing
    else:
        match_search_window_seconds = ref_duration * MATCH_WINDOW_PERCENT
        logger.info(f"  Reference duration: {ref_duration:.2f}s")
        logger.info(f"  Calculated frame match search window: +/- {match_search_window_seconds:.2f}s ({MATCH_WINDOW_PERCENT*100}%)")

    # --- Step 2: Map filenames to timestamps ---
    logger.info("--- Mapping Timestamps to Extracted Frames ---")
    ref_timestamps_dict = {name: ts for name, ts in zip(ref_filenames, ref_timestamps_list)}
    foreign_timestamps_dict = {name: ts for name, ts in zip(foreign_filenames, foreign_timestamps_list)}
    if not ref_timestamps_dict or not foreign_timestamps_dict:
        logger.error("  ERROR: Failed to create timestamp dictionaries.")
        return None
    logger.info(f"  -> Mapped {len(ref_timestamps_dict)} reference and {len(foreign_timestamps_dict)} foreign timestamps.")

    # --- Step 3: Initial Frame Matching (Parallel with Windowing) ---
    logger.info(f"--- Initial Frame Matching (Template Threshold: {match_threshold}, Window: +/- {match_search_window_seconds:.2f}s) ---")
    match_start_time = time.time()
    initial_matches_dict = {} # Stores {ref_name: (foreign_name, ref_time, foreign_time)}
    # === CORRECTION START ===
    future_to_ref_name = {} # Use a dictionary to map Futures to ref_names
    # === CORRECTION END ===

    # Use ThreadPoolExecutor for parallel image comparison
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks: compare each reference frame against a WINDOW of foreign frames
        for ref_name in ref_filenames:
            ref_time = ref_timestamps_dict.get(ref_name)
            if ref_time is None:
                logger.warning(f"Skipping match for ref frame {ref_name}: missing timestamp.")
                continue

            # Determine the time window for foreign frame search
            min_foreign_time = ref_time - match_search_window_seconds
            max_foreign_time = ref_time + match_search_window_seconds

            # Filter foreign filenames based on timestamp window
            candidate_foreign_frames = [
                f_name for f_name in foreign_filenames
                if min_foreign_time <= foreign_timestamps_dict.get(f_name, -1) <= max_foreign_time # Check if foreign ts is within window
            ]

            # Only submit task if there are candidates within the window
            if candidate_foreign_frames:
                future = executor.submit(process_image_pair_for_match,
                                         ref_name,
                                         candidate_foreign_frames, # Pass the filtered list
                                         ref_extract_path,
                                         foreign_extract_path,
                                         match_threshold)
                # === CORRECTION START ===
                future_to_ref_name[future] = ref_name # Store mapping in dictionary
                # === CORRECTION END ===
            else:
                logger.debug(f"No foreign frame candidates found within window for ref frame {ref_name} (time {ref_time:.3f}s)")


        # Process results as they complete
        num_processed = 0
        skipped_count = 0
        # === CORRECTION START ===
        # Pass only the futures (dictionary keys) to as_completed
        futures_iterable = concurrent.futures.as_completed(future_to_ref_name)
        progress_bar = tqdm(total=len(future_to_ref_name), desc="  Matching Frames", unit="frame", ncols=100, leave=False)

        for future in futures_iterable: # Iterate through completed futures
            ref_name = future_to_ref_name[future] # Get the ref_name using the dictionary
        # === CORRECTION END ===
            try:
                result = future.result() # Get result from the completed thread
                if result:
                    # If a match was found (foreign_name, score)
                    foreign_name = result[0]
                    # Retrieve corresponding timestamps
                    # No need to get ref_time again, it's already mapped
                    foreign_time = foreign_timestamps_dict.get(foreign_name)
                    # Store the match details if timestamps are valid
                    if ref_timestamps_dict.get(ref_name) is not None and foreign_time is not None: # Ensure both timestamps are valid
                        initial_matches_dict[ref_name] = (foreign_name, ref_timestamps_dict[ref_name], foreign_time)
                    else:
                         logger.warning(f"Timestamp missing for match: Ref '{ref_name}', Foreign '{foreign_name}'")
                         skipped_count += 1
                # else: No match found above threshold in the window for this ref_frame
            except Exception as e:
                # Catch errors from individual threads
                logger.error(f"\nError processing match task for ref frame {ref_name}: {e}", exc_info=False) # Less verbose stack trace for worker errors
                skipped_count += 1
            finally:
                num_processed += 1
                progress_bar.update(1)
        progress_bar.close()


    match_elapsed_time = time.time() - match_start_time
    logger.info(f"  -> Initial matching complete. Found {len(initial_matches_dict)} potential pairs ({skipped_count} skipped due to errors/missing data). ({match_elapsed_time:.2f}s).")

    if not initial_matches_dict:
        logger.error("No initial matches found between reference and foreign frames. Cannot proceed.")
        return None

    # --- Step 4: Filter Similar Reference Images ---
    matches_after_sim_filter = filter_similar_ref_images(initial_matches_dict, ref_extract_path, similarity_threshold)
    if not matches_after_sim_filter:
        logger.error("No matches remaining after similarity filtering.")
        return None

    # --- Step 5: Filter Temporal Inconsistencies ---
    # Result is a list: [(ref_filename, foreign_filename, ref_time, foreign_time), ...]
    visual_anchors_details = filter_temporal_inconsistency(matches_after_sim_filter)
    if not visual_anchors_details:
        logger.error("No matches remaining after temporal filtering.")
        return None

    final_anchor_count = len(visual_anchors_details)
    stage_elapsed_time = time.time() - stage_start_time
    logger.info(f"---=== Image Pairing Stage Finished ({stage_elapsed_time:.2f}s). Generated {final_anchor_count} visual anchors ===---")
    return visual_anchors_details # Return list of detailed anchor tuples



# --- Audio Syncing Stage Functions ---

def find_audio_start_end(wav_path, db_threshold):
    """Finds the start and end times of audio content above a dB threshold in a WAV file."""
    logger.debug(f"Analyzing audio boundaries for: {os.path.basename(wav_path)} (Threshold: {db_threshold} dB)")
    try:
        sample_rate, audio_data = wavfile.read(wav_path)
        if audio_data.size == 0:
            logger.warning(f"Audio data is empty for {os.path.basename(wav_path)}")
            return 0.0, 0.0 # Return 0 duration if empty

        # Normalize audio data to float range [-1.0, 1.0] for consistent thresholding
        if np.issubdtype(audio_data.dtype, np.integer):
            dtype_info = np.iinfo(audio_data.dtype)
            max_val = float(dtype_info.max)
            min_val = float(dtype_info.min)
            # Avoid division by zero if audio is silent
            norm_factor = max(abs(max_val), abs(min_val))
            if norm_factor == 0: return 0.0, 0.0
            audio_float = audio_data.astype(np.float64) / norm_factor
        elif np.issubdtype(audio_data.dtype, np.floating):
             audio_float = audio_data.astype(np.float64)
             # Handle potential clipping in float audio > 1.0
             abs_max = np.max(np.abs(audio_float)) if audio_float.size > 0 else 0.0
             if abs_max > 1.0 and abs_max > 0:
                 audio_float /= abs_max
             elif abs_max == 0: # Silent float audio
                  return 0.0, 0.0
        else:
             logger.error(f"Unsupported audio data type: {audio_data.dtype} in {os.path.basename(wav_path)}")
             return None, None # Indicate error

        # Convert to mono by taking the max absolute amplitude across channels if stereo
        if audio_float.ndim > 1 and audio_float.shape[1] > 1:
            amplitude = np.max(np.abs(audio_float), axis=1)
        else:
            amplitude = np.abs(audio_float.flatten())

        if amplitude.size == 0: return 0.0, 0.0 # Check again after potential flattening

        # Convert dB threshold to linear amplitude threshold
        # threshold = 10^(dB/20)
        threshold_amplitude = 10.0**(db_threshold / 20.0)

        # Find indices where amplitude exceeds the threshold
        indices_above_thresh = np.where(amplitude >= threshold_amplitude)[0]

        if len(indices_above_thresh) > 0:
            start_index = indices_above_thresh[0]
            end_index = indices_above_thresh[-1]
            # Calculate times in seconds
            start_time_sec = start_index / sample_rate
            # Add 1 sample duration to end time to include the last sample's duration
            end_time_sec = (end_index + 1) / sample_rate

            # Ensure end time is strictly after start time (handle edge cases)
            if end_time_sec <= start_time_sec:
                 # If difference is less than half a sample, treat as single point
                 if abs(end_time_sec - start_time_sec) < (0.5 / sample_rate):
                     end_time_sec = start_time_sec
                 else: # Otherwise, force end time to be slightly after start
                     end_time_sec = start_time_sec + (1.0 / sample_rate)

            logger.debug(f"  -> Detected boundaries: {start_time_sec:.3f}s - {end_time_sec:.3f}s")
            return start_time_sec, end_time_sec
        else:
            # No audio above threshold found
            logger.warning(f"  No audio found above {db_threshold:.1f} dB threshold in {os.path.basename(wav_path)}. Returning full duration or zero.")
            # Optionally return full duration: return 0.0, audio_data.shape[0] / sample_rate
            # Returning zero duration seems safer if threshold is meaningful
            return 0.0, 0.0

    except FileNotFoundError:
        logger.error(f"WAV file not found: {wav_path}")
        return None, None
    except Exception as e:
        logger.error(f"ERROR processing WAV {os.path.basename(wav_path)}: {e}", exc_info=True)
        return None, None

def process_segment_iteratively(foreign_wav_full, foreign_start, foreign_end, ref_duration, segment_num, temp_dir, max_iterations=3, target_precision_ms=5, is_first_segment=False, is_last_segment=False, first_adjust_ms=0.0, last_adjust_ms=0.0):
    """
    Processes an audio segment, iteratively adjusting 'atempo' to match a target duration precisely.

    Args:
        foreign_wav_full (str): Path to the full foreign audio WAV file.
        foreign_start (float): Start time (seconds) of the segment in the foreign audio.
        foreign_end (float): End time (seconds) of the segment in the foreign audio.
        ref_duration (float): The target duration (seconds) for the processed segment.
        segment_num (int): The segment number (for logging).
        temp_dir (str): Path to the temporary directory for intermediate files.
        max_iterations (int): Maximum number of refinement iterations.
        target_precision_ms (int): Desired duration precision in milliseconds.
        is_first_segment (bool): Whether this is the first segment (for adjustment).
        is_last_segment (bool): Whether this is the last segment (for adjustment).
        first_adjust_ms (float): Milliseconds to adjust first segment (+ pad, - trim).
        last_adjust_ms (float): Milliseconds to adjust last segment (+ pad, - trim).

    Returns:
        str: Path to the final processed segment file meeting the target duration, or None on failure.
    """
    target_precision_s = target_precision_ms / 1000.0
    
    # --- Apply Manual Adjustments for First/Last Segments ---
    # Strategy:
    # - For TRIM adjustments (negative values): adjust the extraction boundaries
    # - For PAD adjustments (positive values): add silence via FFmpeg filter, then stretch the combined audio
    
    adjusted_foreign_start = foreign_start
    adjusted_foreign_end = foreign_end
    prepend_silence_s = 0.0  # Silence to add BEFORE the extracted segment
    append_silence_s = 0.0   # Silence to add AFTER the extracted segment
    
    if is_first_segment and first_adjust_ms != 0.0:
        adjust_s = first_adjust_ms / 1000.0
        if adjust_s < 0:
            # NEGATIVE = TRIM from start (skip ahead in audio)
            logger.info(f"  -> First segment: Trimming {abs(adjust_s):.3f}s from start")
            adjusted_foreign_start -= adjust_s  # Move start forward (adjust_s is negative, so this subtracts negative = adds)
        else:
            # POSITIVE = PAD with silence at start
            logger.info(f"  -> First segment: Adding {adjust_s:.3f}s silence padding at start")
            prepend_silence_s = adjust_s
    
    if is_last_segment and last_adjust_ms != 0.0:
        adjust_s = last_adjust_ms / 1000.0
        if adjust_s < 0:
            # NEGATIVE = TRIM from end
            logger.info(f"  -> Last segment: Trimming {abs(adjust_s):.3f}s from end")
            adjusted_foreign_end += adjust_s  # Move end backward (adjust_s is negative)
        else:
            # POSITIVE = PAD with silence at end
            logger.info(f"  -> Last segment: Adding {adjust_s:.3f}s silence padding at end")
            append_silence_s = adjust_s
    
    # Calculate the base extracted duration (before padding)
    base_foreign_duration = adjusted_foreign_end - adjusted_foreign_start
    
    # Total foreign duration including any padding
    foreign_duration = base_foreign_duration + prepend_silence_s + append_silence_s
    
    # Basic validation
    if ref_duration <= 0 or base_foreign_duration <= 0 or foreign_duration <= 0:
        logger.warning(f"  -> Segment {segment_num}: Invalid duration (Ref={ref_duration:.3f}s, Base={base_foreign_duration:.3f}s, Total={foreign_duration:.3f}s)")
        return None

    # --- Initial setup ---
    # Initial speed factor estimate
    # IMPORTANT: `atempo` filter works inversely: tempo < 1 slows down, tempo > 1 speeds up.
    # So, we need foreign_duration / ref_duration
    initial_speed_factor = foreign_duration / ref_duration
    clamped_speed = max(MIN_ATEMPO, min(MAX_ATEMPO, initial_speed_factor))

    segment_output_path = os.path.join(temp_dir, f"segment_{segment_num:04d}_final.wav")
    best_segment_path = None
    best_duration_diff = float('inf')
    last_processed_duration = None

    logger.info(f"  -> Segment {segment_num}: Target={ref_duration:.3f}s, Base={base_foreign_duration:.3f}s, +Padding={foreign_duration:.3f}s. Initial speed={clamped_speed:.5f}x")

    # --- Iterative Refinement Loop ---
    for iteration in range(max_iterations):
        iteration_path = os.path.join(temp_dir, f"segment_{segment_num:04d}_iter{iteration}.wav")

        # Build FFmpeg filter chain with padding if needed
        if prepend_silence_s > 0 or append_silence_s > 0:
            # Need to add silence - use complex filter chain
            filter_parts = []
            input_labels = []
            
            # Generate prepend silence if needed
            if prepend_silence_s > 0:
                filter_parts.append(f"aevalsrc=0:d={prepend_silence_s:.8f}:s={DEFAULT_SAMPLE_RATE}:c={DEFAULT_CHANNELS}[pre_silence]")
                input_labels.append("[pre_silence]")
            
            # Extract and label the main audio segment
            filter_parts.append(f"[0:a]atrim=start={adjusted_foreign_start:.8f}:end={adjusted_foreign_end:.8f},asetpts=PTS-STARTPTS[main_audio]")
            input_labels.append("[main_audio]")
            
            # Generate append silence if needed
            if append_silence_s > 0:
                filter_parts.append(f"aevalsrc=0:d={append_silence_s:.8f}:s={DEFAULT_SAMPLE_RATE}:c={DEFAULT_CHANNELS}[post_silence]")
                input_labels.append("[post_silence]")
            
            # Concatenate all parts
            n_inputs = len(input_labels)
            concat_inputs = "".join(input_labels)
            filter_parts.append(f"{concat_inputs}concat=n={n_inputs}:v=0:a=1[padded]")
            
            # Apply atempo to the padded audio
            filter_parts.append(f"[padded]atempo={clamped_speed:.8f}")
            
            filter_complex = ";".join(filter_parts)
        else:
            # No padding needed - simple filter
            filter_complex = f"atrim=start={adjusted_foreign_start:.8f}:end={adjusted_foreign_end:.8f},asetpts=PTS-STARTPTS,atempo={clamped_speed:.8f}"
        
        process_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
            "-i", foreign_wav_full,
            "-filter_complex", filter_complex,
            "-c:a", "pcm_s16le", "-ar", str(DEFAULT_SAMPLE_RATE), "-ac", str(DEFAULT_CHANNELS),
            "-y", iteration_path
        ]

        if not run_ffmpeg(process_cmd, f"Process Segment {segment_num} (Iter {iteration+1}, Speed {clamped_speed:.5f}x)")[0]:
            logger.error(f"  -> Segment {segment_num}: Processing failed on iteration {iteration+1}")
            continue

        # Measure the actual duration of the processed segment
        processed_duration = get_file_duration(iteration_path, media_type='audio')
        if processed_duration is None:
            logger.warning(f"  -> Segment {segment_num}: Could not get duration for iteration {iteration+1}")
            continue

        duration_diff = processed_duration - ref_duration
        abs_duration_diff = abs(duration_diff)
        logger.info(f"    Iter {iteration+1}: Speed={clamped_speed:.5f}x -> Duration={processed_duration:.3f}s (Diff={duration_diff*1000:+.1f}ms)")

        # Keep track of the best result so far
        if abs_duration_diff < best_duration_diff:
            best_duration_diff = abs_duration_diff
            best_segment_path = iteration_path

        # Check if we've reached the target precision
        if abs_duration_diff <= target_precision_s:
            logger.info(f"   Segment {segment_num}: Achieved target precision ({abs_duration_diff*1000:.1f}ms <= {target_precision_ms}ms) on iteration {iteration+1}")
            break

        # --- Adjust speed factor for the next iteration ---
        if iteration < max_iterations - 1:
            if processed_duration <= 0:
                logger.warning(f"    Skipping speed adjustment for iter {iteration+1}: Processed duration is zero or negative.")
                continue

            ideal_correction = processed_duration / ref_duration
            dampening = 1.0 - min(0.7, abs(ideal_correction - 1.0) * 1.5)
            dampened_correction = (ideal_correction - 1.0) * dampening + 1.0
            next_speed = clamped_speed * dampened_correction
            clamped_speed = max(MIN_ATEMPO, min(MAX_ATEMPO, next_speed))
            logger.debug(f"    Adjusting speed: IdealCorr={ideal_correction:.6f}x, DampenedCorr={dampened_correction:.6f}x -> NextSpeed={clamped_speed:.6f}x")
            last_processed_duration = processed_duration

    # --- Post-Iteration Handling ---
    if best_segment_path is None:
         logger.error(f" Segment {segment_num}: No successful iteration completed.")
         return None

    # Check the duration of the best segment found
    final_processed_duration = get_file_duration(best_segment_path, media_type='audio')
    if final_processed_duration is None:
        logger.error(f" Segment {segment_num}: Could not get duration of best segment '{os.path.basename(best_segment_path)}'.")
        return None

    final_duration_gap = ref_duration - final_processed_duration

    # If the best result is still outside precision, perform final trim/pad
    if abs(final_duration_gap) > target_precision_s:
        logger.warning(f"  -> Segment {segment_num}: Best iteration ({final_processed_duration:.3f}s) still {final_duration_gap*1000:+.1f}ms off target. Applying final correction.")

        if final_duration_gap > 0: # Segment is too short, need to pad with silence
            silence_path = os.path.join(temp_dir, f"silence_{segment_num:04d}.wav")
            silence_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
                "-f", "lavfi", "-i", f"anullsrc=r={DEFAULT_SAMPLE_RATE}:cl={DEFAULT_CHANNELS}",
                "-t", f"{final_duration_gap:.8f}",
                "-c:a", "pcm_s16le", "-y", silence_path
            ]
            if not run_ffmpeg(silence_cmd, f"Create Silence Pad for Segment {segment_num} ({final_duration_gap:.3f}s)")[0]:
                 logger.error(f" Segment {segment_num}: Failed to create silence pad.")
                 return None

            concat_list_path = os.path.join(temp_dir, f"concat_list_{segment_num:04d}.txt")
            try:
                with open(concat_list_path, 'w', encoding='utf-8') as f_concat:
                    f_concat.write(f"file '{os.path.basename(best_segment_path)}'\n")
                    f_concat.write(f"file '{os.path.basename(silence_path)}'\n")
            except IOError as e:
                 logger.error(f" Segment {segment_num}: Failed to write concat list: {e}")
                 return None

            concat_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                "-y", segment_output_path
            ]
            if not run_ffmpeg(concat_cmd, f"Add Silence Pad to Segment {segment_num}")[0]:
                logger.error(f" Segment {segment_num}: Failed to concatenate silence pad.")
                return None
            logger.info(f"   Segment {segment_num}: Added {final_duration_gap*1000:.1f}ms silence pad for final correction.")
            return segment_output_path

        elif final_duration_gap < 0: # Segment is too long, need to trim
            trim_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
                "-i", best_segment_path,
                "-t", f"{ref_duration:.8f}",
                "-c", "copy",
                "-y", segment_output_path
            ]
            if not run_ffmpeg(trim_cmd, f"Trim Segment {segment_num} to {ref_duration:.3f}s")[0]:
                logger.error(f" Segment {segment_num}: Failed to trim segment.")
                return None
            logger.info(f"   Segment {segment_num}: Trimmed by {abs(final_duration_gap)*1000:.1f}ms for final correction.")
            return segment_output_path
    else:
        # Best iteration was already within precision
        logger.info(f"   Segment {segment_num}: Best iteration duration ({final_processed_duration:.3f}s) already within {target_precision_ms}ms of target.")
        try:
            shutil.copy2(best_segment_path, segment_output_path)
            return segment_output_path
        except Exception as e:
            logger.error(f" Segment {segment_num}: Failed to copy best segment to final path: {e}")
            return None

    logger.error(f" Segment {segment_num}: Failed to produce final segment after iterations and correction.")
    return None
def run_progressive_sync_iterative(args, visual_anchors_details, output_audio_path, temp_dir, db_threshold, min_segment_duration):
    """
    Audio sync stage using iterative refinement for precise segment durations.
    Filters anchors, processes each segment iteratively, concatenates, and applies delay.
    """
    logger.info("\n===== \1 =====")
    stage_start_time = time.time()

    # Define full paths for extracted audio
    ref_wav_full = os.path.join(temp_dir, "ref_audio_full.wav")
    foreign_wav_full = os.path.join(temp_dir, "foreign_audio_full.wav")

    # --- Step 1: Determine Audio Stream Indices ---
    logger.info(f"--- Finding Audio Streams (Ref: {args.ref_lang}, Foreign: {args.foreign_lang}) ---")
    ref_stream_idx = args.ref_stream_idx
    if ref_stream_idx is None:
        ref_streams = get_stream_info(args.ref_video)
        ref_stream_idx = find_audio_stream_index_by_lang(ref_streams, args.ref_lang)
        if ref_stream_idx is not None: logger.info(f"  -> Auto-detected Reference Stream Index: {ref_stream_idx}")
    else: logger.info(f"  -> Using Forced Reference Stream Index: {ref_stream_idx}")

    foreign_stream_idx = args.foreign_stream_idx
    if foreign_stream_idx is None:
        foreign_streams = get_stream_info(args.foreign_video)
        foreign_stream_idx = find_audio_stream_index_by_lang(foreign_streams, args.foreign_lang)
        if foreign_stream_idx is not None: logger.info(f"  -> Auto-detected Foreign Stream Index: {foreign_stream_idx}")
    else: logger.info(f"  -> Using Forced Foreign Stream Index: {foreign_stream_idx}")

    # Validate indices
    if ref_stream_idx is None:
        logger.error("-> Could not determine reference audio stream index. Cannot proceed.")
        return None, None
    if foreign_stream_idx is None:
        logger.error("-> Could not determine foreign audio stream index. Cannot proceed.")
        return None, None

    # --- Step 2: Extract Full Audio Tracks ---
    logger.info(f"--- Extracting Audio Tracks (Ref Index: {ref_stream_idx}, Foreign Index: {foreign_stream_idx}) ---")
    # Consistent resampling for quality and compatibility
    aresample_filter = f'aresample=resampler=soxr:precision=28:cutoff={0.99 if DEFAULT_SAMPLE_RATE >= 44100 else 0.90}'

    extract_cmd_ref = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats",
                       "-i", args.ref_video,
                       # Use absolute stream index mapping:
                       "-map", f"0:{ref_stream_idx}", # <<< CORRECTED MAPPING
                       "-vn",
                       "-c:a", "pcm_s16le", "-ar", str(DEFAULT_SAMPLE_RATE), "-ac", str(DEFAULT_CHANNELS),
                       "-af", aresample_filter, "-y", "-f", "wav", ref_wav_full]
    if not run_ffmpeg(extract_cmd_ref, f"Extract Reference Audio (Index {ref_stream_idx})")[0]:
        return None, None # Abort if extraction fails


    extract_cmd_foreign = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats",
                           "-i", args.foreign_video,
                           # Use absolute stream index mapping:
                           "-map", f"0:{foreign_stream_idx}", # <<< CORRECTED MAPPING
                           "-vn",
                           "-c:a", "pcm_s16le", "-ar", str(DEFAULT_SAMPLE_RATE), "-ac", str(DEFAULT_CHANNELS),
                           "-af", aresample_filter, "-y", "-f", "wav", foreign_wav_full]
    if not run_ffmpeg(extract_cmd_foreign, f"Extract Foreign Audio (Index {foreign_stream_idx})")[0]:
        return None, None # Abort if extraction fails

    # --- Step 3: Detect Audio Boundaries ---
    logger.info(f"--- Detecting Audio Content Boundaries (Threshold: {db_threshold} dB) ---")
    ref_start_s, ref_end_s = find_audio_start_end(ref_wav_full, db_threshold)
    foreign_start_s, foreign_end_s = find_audio_start_end(foreign_wav_full, db_threshold)

    if ref_start_s is None or foreign_start_s is None:
        logger.error("-> Failed to detect audio boundaries. Cannot proceed.")
        return None, None

    ref_delay_s = ref_start_s # The initial silence in the reference audio determines the final output delay
    ref_content_duration = ref_end_s - ref_start_s
    foreign_content_duration = foreign_end_s - foreign_start_s

    logger.info(f"  -> Reference Audio Content: {format_time(ref_start_s)} -> {format_time(ref_end_s)} (Duration: {ref_content_duration:.3f}s)")
    logger.info(f"  -> Foreign Audio Content:   {format_time(foreign_start_s)} -> {format_time(foreign_end_s)} (Duration: {foreign_content_duration:.3f}s)")
    logger.info(f"  -> Calculated Reference Delay (Padding): {ref_delay_s:.3f} seconds")

    # --- Step 4: Combine and Filter Anchor Points ---
    logger.info(f"--- Filtering Anchor Points (Min Ref Segment Duration: {min_segment_duration}s, Max Duration Diff: {MAX_ALLOWED_DURATION_PERCENT_DIFF}%) ---")
    # Start with audio boundaries as the absolute first and last anchors
    all_anchors = [(ref_start_s, foreign_start_s)]
    added_image_count = 0
    added_forced_count = 0
    skipped_outside = 0
    
    # Track forced sync point reference times - these bypass all filtering
    forced_ref_times = set()

    # Add visual anchors from Stage 1, ensuring they fall within the detected audio content times
    # EXCEPTION: Forced sync points (starting with 'FORCED_SYNC_') bypass boundary checks
    for ref_name, _, ref_img_time, foreign_img_time in visual_anchors_details:
        is_forced = ref_name.startswith('FORCED_SYNC_')
        
        if is_forced:
            # Forced sync points are always included (that's the point!)
            all_anchors.append((ref_img_time, foreign_img_time))
            forced_ref_times.add(ref_img_time)  # Track this as a forced anchor
            added_forced_count += 1
            logger.info(f"  > FORCED sync point added: Ref={format_time(ref_img_time)} -> Foreign={format_time(foreign_img_time)}")
        else:
            # Check if anchor falls within the content boundaries of BOTH reference and foreign audio
            is_within_ref = (ref_start_s <= ref_img_time <= ref_end_s)
            is_within_foreign = (foreign_start_s <= foreign_img_time <= foreign_end_s)
            if is_within_ref and is_within_foreign:
                all_anchors.append((ref_img_time, foreign_img_time))
                added_image_count += 1
            else:
                logger.debug(f"    Skipping visual anchor RefT={ref_img_time:.3f}/ForeignT={foreign_img_time:.3f} - outside audio bounds ({is_within_ref=}, {is_within_foreign=})")
                skipped_outside += 1

    all_anchors.append((ref_end_s, foreign_end_s)) # Add audio end boundary
    all_anchors.sort() # Sort chronologically by reference time

    logger.info(f"  > Started with {len(all_anchors)} total anchors (2 audio boundaries + {added_image_count} visual + {added_forced_count} forced).")
    if skipped_outside > 0: logger.info(f"  > Skipped {skipped_outside} visual anchors falling outside audio content boundaries.")

    # --- Filter 1: Minimum Reference Segment Duration ---
    # NOTE: Forced sync points (in forced_ref_times) bypass this filter
    min_ref_dur_filtered_anchors = []
    skipped_short_ref = 0
    if all_anchors:
        min_ref_dur_filtered_anchors.append(all_anchors[0]) # Always keep the first anchor (audio start)
        for i in range(1, len(all_anchors)):
            last_kept_ref_time, _ = min_ref_dur_filtered_anchors[-1]
            current_ref_time, _ = all_anchors[i]
            segment_ref_duration = current_ref_time - last_kept_ref_time
            
            # ALWAYS keep forced sync points regardless of segment duration
            is_forced = current_ref_time in forced_ref_times

            # Keep the current anchor if it's forced OR if the segment meets minimum duration
            if is_forced:
                min_ref_dur_filtered_anchors.append(all_anchors[i])
                logger.debug(f"    MinRefDur Filter: KEEPING forced anchor (RefT={current_ref_time:.3f}) despite segment duration ({segment_ref_duration:.3f}s)")
            elif segment_ref_duration >= min_segment_duration:
                min_ref_dur_filtered_anchors.append(all_anchors[i])
            elif i < len(all_anchors) - 1: # Don't count removal if it's the segment before the very last anchor
                logger.debug(f"    MinRefDur Filter: Removing anchor {i} (RefT={current_ref_time:.3f}) because segment duration ({segment_ref_duration:.3f}s) < {min_segment_duration}s")
                skipped_short_ref += 1
            # else: Anchor is the last one, but segment is too short - keep it anyway to preserve the endpoint

        # Ensure the final anchor point (audio end) is always included, even if the last segment is short
        if len(all_anchors) > 1 and min_ref_dur_filtered_anchors[-1] != all_anchors[-1]:
            last_kept_ref, _ = min_ref_dur_filtered_anchors[-1]
            actual_last_ref, _ = all_anchors[-1]
            last_segment_dur = actual_last_ref - last_kept_ref
            logger.warning(f"  -> Final segment ({last_segment_dur:.2f}s) is shorter than minimum ({min_segment_duration}s), but keeping final endpoint.") # Use argument here
            min_ref_dur_filtered_anchors.append(all_anchors[-1]) # Re-add the true last anchor

    if skipped_short_ref > 0: logger.info(f"  -> Filtered out {skipped_short_ref} anchors creating reference segments shorter than {min_segment_duration}s.") # Use argument here

    # --- Filter 2: Maximum Segment Duration Percentage Difference ---
    fully_filtered_anchors = []
    skipped_percent_diff = 0
    if min_ref_dur_filtered_anchors:
        fully_filtered_anchors.append(min_ref_dur_filtered_anchors[0]) # Always keep the start anchor
        for i in range(len(min_ref_dur_filtered_anchors) - 1):
            ref_start, foreign_start = min_ref_dur_filtered_anchors[i]
            ref_end, foreign_end = min_ref_dur_filtered_anchors[i+1] # Look ahead to the next anchor

            ref_seg_duration = ref_end - ref_start
            foreign_seg_duration = foreign_end - foreign_start
            
            # ALWAYS keep forced sync points regardless of duration difference
            is_end_forced = ref_end in forced_ref_times
            
            if is_end_forced:
                # Forced sync point - always keep
                fully_filtered_anchors.append(min_ref_dur_filtered_anchors[i+1])
                logger.debug(f"    MaxDiff Filter: KEEPING forced anchor (RefT={ref_end:.3f}) despite duration diff")
            # Avoid division by zero for zero-duration segments
            elif ref_seg_duration > 1e-6: # Use a small epsilon
                duration_diff_percent = abs(ref_seg_duration - foreign_seg_duration) / ref_seg_duration * 100
                if duration_diff_percent <= MAX_ALLOWED_DURATION_PERCENT_DIFF:
                    # If difference is acceptable, keep the *end* anchor of this valid segment
                    fully_filtered_anchors.append(min_ref_dur_filtered_anchors[i+1])
                else:
                    logger.debug(f"    MaxDiff Filter: Removing anchor {i+1} (RefT={ref_end:.3f}) - segment duration diff ({duration_diff_percent:.2f}%) > {MAX_ALLOWED_DURATION_PERCENT_DIFF}%")
                    skipped_percent_diff += 1
            elif abs(foreign_seg_duration) < 1e-6:
                 # Both ref and foreign durations are near zero, keep the anchor
                 fully_filtered_anchors.append(min_ref_dur_filtered_anchors[i+1])
            else:
                 # Reference duration is zero/negative, foreign is not. This indicates a problem. Remove.
                 logger.warning(f"    MaxDiff Filter: Removing anchor {i+1} (RefT={ref_end:.3f}) due to zero/negative ref duration ({ref_seg_duration:.3f}s) vs non-zero foreign duration ({foreign_seg_duration:.3f}s).")
                 skipped_percent_diff += 1

        # Ensure the very last anchor point is always present after filtering
        if min_ref_dur_filtered_anchors and fully_filtered_anchors[-1] != min_ref_dur_filtered_anchors[-1]:
            logger.info("  -> Re-adding final audio endpoint anchor after duration difference filtering.")
            fully_filtered_anchors.append(min_ref_dur_filtered_anchors[-1])
            # Remove potential duplicate if the last was already added and identical to second-last
            if len(fully_filtered_anchors) > 1 and fully_filtered_anchors[-1] == fully_filtered_anchors[-2]:
                 fully_filtered_anchors.pop()

    if skipped_percent_diff > 0: logger.info(f"  -> Filtered out {skipped_percent_diff} anchors creating segments with duration difference > {MAX_ALLOWED_DURATION_PERCENT_DIFF}%.")


    final_segment_anchors = fully_filtered_anchors # Use the fully filtered list
    num_segments = len(final_segment_anchors) - 1 # Number of segments is number of anchors - 1

    logger.info(f"  -> Using {len(final_segment_anchors)} final anchors, defining {num_segments} segments for processing.")
    if num_segments <= 0:
        logger.error("-> Need at least 2 final anchors (start and end) to define segments. Cannot proceed.")
        return None, None

    # --- Step 5: Write Segment Info to CSV (Optional) ---
    if args.output_csv: # Check if CSV output is requested
        logger.info(f"--- Writing segment information to CSV: {args.output_csv} ---")
        try:
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Define headers
                header = [
                    "Segment", "Ref Start Time", "Ref End Time", "Foreign Start Time", "Foreign End Time",
                    "Ref Duration (s)", "Foreign Duration (s)", "Duration Diff (%)",
                    "Initial Speed Factor" # Speed factor before iterative adjustment
                ]
                writer.writerow(header)

                # Write data for each segment
                for i in range(num_segments):
                    ref_start, foreign_start = final_segment_anchors[i]
                    ref_end, foreign_end = final_segment_anchors[i+1]
                    ref_dur = ref_end - ref_start
                    foreign_dur = foreign_end - foreign_start

                    # Calculate percentage difference and initial speed
                    percent_diff = float('inf')
                    initial_speed = 1.0
                    if ref_dur > 1e-6: # Avoid division by zero
                        percent_diff = abs(ref_dur - foreign_dur) / ref_dur * 100
                        initial_speed = foreign_dur / ref_dur
                    elif abs(foreign_dur) < 1e-6: # Both near zero
                         percent_diff = 0.0

                    writer.writerow([
                        i + 1,
                        format_time(ref_start), format_time(ref_end),
                        format_time(foreign_start), format_time(foreign_end),
                        f"{ref_dur:.3f}", f"{foreign_dur:.3f}",
                        f"{percent_diff:.2f}" if percent_diff != float('inf') else "N/A",
                        f"{initial_speed:.5f}"
                    ])
            logger.info(f"  -> Successfully wrote segment data to {args.output_csv}")
        except Exception as e:
            logger.error(f"-> Failed to write segment CSV file: {e}", exc_info=True)
            # Continue processing even if CSV writing fails
    else:
        logger.debug("Skipping CSV output (not requested)")

    # --- Step 6: Process Segments Iteratively ---
    logger.info(f"--- Processing {num_segments} Segments ---")
    process_start_time = time.time()
    processed_segment_files = [] # List to store paths of successfully processed segments
    total_processed_ref_duration = 0.0 # Sum of target durations for processed segments

    # Configure iterative processing parameters
    max_iterations = 3       # Max attempts per segment
    target_precision_ms = 5  # Target accuracy in milliseconds
    
    # Track corrections for summary
    segments_with_corrections = 0

    # Use progress bar for segment processing (visible on console)
    # Dynamic description shows current segment being processed
    pbar = tqdm(
        total=num_segments,
        desc="Processing",
        unit="seg",
        ncols=80,
        bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        file=sys.stdout,
        dynamic_ncols=False
    )

    for i in range(num_segments):
        segment_num = i + 1
        pbar.set_description(f"Segment {segment_num:3d}/{num_segments}")
        ref_start, foreign_start = final_segment_anchors[i]
        ref_end, foreign_end = final_segment_anchors[i+1]
        target_ref_duration = ref_end - ref_start # This is the target duration for the output segment

        logger.debug(f"Processing Segment {segment_num}/{num_segments}")

        # Call the iterative processing function for this segment
        segment_path = process_segment_iteratively(
            foreign_wav_full=foreign_wav_full,
            foreign_start=foreign_start,
            foreign_end=foreign_end,
            ref_duration=target_ref_duration, # Pass the target duration
            segment_num=segment_num,
            temp_dir=temp_dir,
            max_iterations=max_iterations,
            target_precision_ms=target_precision_ms,
            is_first_segment=(segment_num == 1),
            is_last_segment=(segment_num == num_segments),
            first_adjust_ms=args.first_segment_adjust,
            last_adjust_ms=args.last_segment_adjust
        )

        if segment_path and os.path.exists(segment_path):
            processed_segment_files.append(segment_path)
            total_processed_ref_duration += target_ref_duration # Add target duration to total
        else:
            pbar.close()
            logger.error(f"Failed to process segment {segment_num}. Aborting audio synchronization.")
            return None, None # Critical failure, stop processing
        
        pbar.update(1)

    pbar.close()
    
    process_elapsed_time = time.time() - process_start_time
    if not processed_segment_files:
         logger.error("No audio segments were successfully processed.")
         return None, None
    logger.info(f"  [OK] Processed {len(processed_segment_files)} segments in {process_elapsed_time:.1f}s")


    # --- Step 7: Concatenate Processed Segments ---
    logger.info("--- Concatenating Precisely-Timed Segments ---")
    concatenated_foreign_temp = os.path.join(temp_dir, "foreign_concatenated_temp.wav")
    concat_list_path = os.path.join(temp_dir, "concat_list_final.txt")

    try:
        with open(concat_list_path, 'w', encoding='utf-8') as f_concat:
            for seg_path in processed_segment_files:
                # Use relative paths within temp dir, ensure forward slashes for ffmpeg compatibility
                f_concat.write(f"file '{os.path.basename(seg_path)}'\n")
    except IOError as e:
        logger.error(f"-> Failed to create final concat list: {e}")
        return None, None

    # FFmpeg command for concatenation
    concat_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats",
        "-f", "concat", "-safe", "0", # Allow relative paths from list file
        "-i", concat_list_path,
        "-c", "copy", # Copy streams without re-encoding
        "-y", concatenated_foreign_temp
    ]

    if not run_ffmpeg(concat_cmd, "Concatenate All Processed Segments")[0]:
        logger.error("-> Failed to concatenate processed segments.")
        return None, None

    # --- Step 8: Verify Final Concatenated Duration ---
    final_duration = get_file_duration(concatenated_foreign_temp, media_type='audio')
    expected_total_duration = ref_content_duration # Should match the total duration of the reference content

    if final_duration is not None:
        duration_diff = final_duration - expected_total_duration
        logger.info(f"  -> Final Concatenated Audio Duration: {final_duration:.3f}s (Expected Reference Content Duration: {expected_total_duration:.3f}s)")
        if abs(duration_diff) > 0.1: # Check if difference is > 100ms
            logger.warning(f"  -> WARNING: Final duration differs from expected reference duration by {duration_diff:+.3f}s. Check segment processing logs.")
        else:
            logger.info(f"  -> SUCCESS: Final duration matches expected reference duration within {abs(duration_diff):.3f}s.")
    else:
        logger.warning("  -> Could not verify final concatenated audio duration using ffprobe.")


    # --- Step 9: Apply Start Delay Padding ---
    logger.info("--- Applying Start Delay Padding ---")
    if ref_delay_s >= MIN_DELAY_S: # Only pad if delay is significant
        delay_ms = int(ref_delay_s * 1000)
        logger.info(f"  Applying {delay_ms}ms start delay padding...")
        pad_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats",
            "-i", concatenated_foreign_temp, # Input is the concatenated audio
            "-af", f"adelay={delay_ms}|{delay_ms}", # Apply delay to all channels
            "-c:a", "pcm_s16le", # Keep WAV format for output
            "-ar", str(DEFAULT_SAMPLE_RATE), "-ac", str(DEFAULT_CHANNELS), # Maintain audio spec
            "-y", output_audio_path # Final output file path
        ]

        if not run_ffmpeg(pad_cmd, f"Apply {delay_ms}ms Padding")[0]:
            logger.error(f"-> Failed to apply start delay padding to final audio.")
            # Output without padding might still exist at concatenated_foreign_temp
            # Consider copying it to output_audio_path as a fallback?
            try:
                shutil.copy2(concatenated_foreign_temp, output_audio_path)
                logger.warning(f"  -> Copied unpadded audio to {output_audio_path} as padding failed.")
            except Exception as copy_err:
                logger.error(f"  -> Failed to copy unpadded audio as fallback: {copy_err}")
                return None, None # Indicate failure if padding fails and copy fails
        else:
             logger.info(f"  -> Successfully added {delay_ms}ms start delay padding.")
    else:
        # If delay is too small, just copy the concatenated file to the final output path
        logger.info(f"  -> Skipping delay padding (Reference delay {ref_delay_s:.3f}s < {MIN_DELAY_S:.3f}s).")
        try:
            shutil.copy2(concatenated_foreign_temp, output_audio_path)
            logger.info(f"  -> Copied concatenated audio directly to {output_audio_path}.")
        except Exception as e:
            logger.error(f"-> Failed to copy final audio (without padding): {e}")
            return None, None # Indicate failure if copy fails

    # --- Stage Completion ---
    stage_elapsed_time = time.time() - stage_start_time
    logger.info(f"---=== Audio Synchronization Stage Finished ({stage_elapsed_time:.2f}s) ===---")
    # Return the calculated delay and the final list of anchors used (for QC)
    return ref_delay_s, final_segment_anchors


# --- QC Image Generation ---
def _create_single_qc(ref_frame_path, foreign_frame_path, qc_output_path):
    """Creates a side-by-side comparison image from two frame paths."""
    try:
        img_ref = cv2.imread(ref_frame_path)
        img_foreign = cv2.imread(foreign_frame_path)

        if img_ref is None or img_foreign is None:
            logger.warning(f"QC Skip: Could not read images for QC: Ref='{os.path.basename(ref_frame_path)}', Foreign='{os.path.basename(foreign_frame_path)}'")
            return False

        h_ref, w_ref = img_ref.shape[:2]
        h_foreign, w_foreign = img_foreign.shape[:2]

        # Ensure images have valid dimensions
        if h_ref == 0 or w_ref == 0 or h_foreign == 0 or w_foreign == 0:
            logger.warning(f"QC Skip: Invalid image dimensions for Ref='{os.path.basename(ref_frame_path)}' or Foreign='{os.path.basename(foreign_frame_path)}'")
            return False

        target_h = QC_IMAGE_HEIGHT # Use constant for target height

        # Resize reference image maintaining aspect ratio
        ref_ratio = target_h / h_ref
        new_w_ref = int(w_ref * ref_ratio)
        if new_w_ref <= 0: return False # Check for invalid width
        img_ref_resized = cv2.resize(img_ref, (new_w_ref, target_h), interpolation=cv2.INTER_AREA)

        # Resize foreign image maintaining aspect ratio
        foreign_ratio = target_h / h_foreign
        new_w_foreign = int(w_foreign * foreign_ratio)
        if new_w_foreign <= 0: return False # Check for invalid width
        img_foreign_resized = cv2.resize(img_foreign, (new_w_foreign, target_h), interpolation=cv2.INTER_AREA)

        # Concatenate horizontally
        qc_image = cv2.hconcat([img_ref_resized, img_foreign_resized])

        # Save the QC image (use PNG for lossless quality)
        # Add compression level for PNG to potentially reduce size
        cv2.imwrite(qc_output_path, qc_image, [cv2.IMWRITE_PNG_COMPRESSION, 3]) # Compression level 0-9
        return True

    except Exception as e:
        logger.error(f"Error creating QC image '{os.path.basename(qc_output_path)}': {e}", exc_info=True)
        return False

def create_qc_images(visual_anchors_details, final_segment_anchors,
                     ref_extract_path, foreign_extract_path, qc_output_dir):
    """Generates side-by-side QC images for the anchor points used in the final segments."""
    # qc_output_dir check is done in main() before calling this
    if not visual_anchors_details:
         logger.warning("QC image generation skipped: No visual anchor details available.")
         return
    # Need at least start, one intermediate, and end anchor (3 total) to have intermediate points
    if not final_segment_anchors or len(final_segment_anchors) < 3:
        logger.warning("QC image generation skipped: Not enough final segment anchors (need >= 3) to generate QC for intermediate points.")
        return

    logger.info("\n===== \1 =====")
    logger.info(f"Saving QC images to: {qc_output_dir}")
    os.makedirs(qc_output_dir, exist_ok=True)
    start_time = time.time()
    qc_count = 0

    # Create a lookup map from the original visual anchors list: (ref_time, foreign_time) -> (ref_filename, foreign_filename)
    visual_lookup = {(r_t, f_t): (r_fn, f_fn) for r_fn, f_fn, r_t, f_t in visual_anchors_details}

    # We want QC images for the *internal* anchor points that defined the final segments.
    # Exclude the very first (audio start) and very last (audio end) anchors from QC generation,
    # as these might not correspond directly to visually matched frames.
    internal_segment_anchors = final_segment_anchors[1:-1]

    if not internal_segment_anchors:
         logger.warning("No internal anchor points found after filtering; cannot generate intermediate QC images.")
         return

    logger.info(f"Attempting to generate QC images for {len(internal_segment_anchors)} internal anchor points used in final segments.")

    for ref_ts, foreign_ts in tqdm(internal_segment_anchors, desc="  Generating QC", unit="image", ncols=100, leave=False):
        anchor_tuple = (ref_ts, foreign_ts)

        # Find the corresponding filenames using the lookup map
        if anchor_tuple in visual_lookup:
            ref_filename, foreign_filename = visual_lookup[anchor_tuple]
            ref_frame_path = os.path.join(ref_extract_path, ref_filename)
            foreign_frame_path = os.path.join(foreign_extract_path, foreign_filename)

            # Check if the actual frame image files exist
            if os.path.exists(ref_frame_path) and os.path.exists(foreign_frame_path):
                # Create a descriptive filename for the QC image
                ref_filename_base = os.path.splitext(ref_filename)[0] # e.g., "frame_000123"
                # Include reference timestamp in filename for easy identification
                qc_filename = f"qc_{ref_filename_base}_reft{ref_ts:.3f}s.png"
                qc_output_path = os.path.join(qc_output_dir, qc_filename)

                # Create the single QC image
                if _create_single_qc(ref_frame_path, foreign_frame_path, qc_output_path):
                    qc_count += 1
            else:
                logger.warning(f"QC Skip: Frame file missing for anchor (RefT={ref_ts:.3f}, ForeignT={foreign_ts:.3f}): Ref='{ref_filename}' or Foreign='{foreign_filename}'")
        else:
            # This might happen if an audio boundary point coincides exactly with a visual anchor time,
            # but generally internal points should come from the visual_anchors_details list.
            logger.warning(f"QC Skip: Could not find original filenames in visual anchor details for final segment anchor point (RefT={ref_ts:.3f}, ForeignT={foreign_ts:.3f})")

    elapsed_time = time.time() - start_time
    logger.info(f"  -> QC image generation complete. Created {qc_count} images ({elapsed_time:.2f}s).")


# --- Muxing Function ---

def run_muxing(args, ref_stream_idx, synced_subtitles=None):
    """
    Muxes the reference video with its original audio, the newly synced foreign audio, and synced subtitles.
    Handles deletion of temporary audio files.
    """
    if not args.output_video: # Check the required output video path
        logger.error("No output video path specified. Cannot mux.")
        return False
    if not os.path.exists(args.output_audio):
        logger.error(f"Muxing failed: Synced audio file '{args.output_audio}' not found.")
        return False

    logger.info("\n===== \1 =====")
    logger.info(f"  Reference Video Source: {os.path.basename(args.ref_video)}")
    logger.info(f"  Synced Foreign Audio:   {os.path.basename(args.output_audio)}")
    logger.info(f"  Output Muxed Video:     {os.path.basename(args.output_video)}") # Use output_video

    # Reference audio stream index should have been determined in the sync stage
    if ref_stream_idx is None:
         logger.warning("Reference stream index not provided to muxing stage, attempting to find again.")
         ref_streams = get_stream_info(args.ref_video)
         ref_stream_idx = find_audio_stream_index_by_lang(ref_streams, args.ref_lang)
         if ref_stream_idx is None:
              logger.error("Could not determine reference audio stream index for muxing. Aborting mux.")
              return False # Return False here so main knows muxing failed
    logger.info(f"  Using Reference Audio Stream Index: {ref_stream_idx}")

    # Construct FFmpeg command for muxing
    ffmpeg_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-stats',
        '-i', args.ref_video,              # Input 0: Reference video (contains video + original audio)
        '-i', args.output_audio,           # Input 1: Synced foreign audio (WAV)
    ]
    
    # Add subtitle inputs
    if synced_subtitles:
        for sub_info in synced_subtitles:
            ffmpeg_cmd.extend(['-i', sub_info['path']])
        logger.info(f"  Including {len(synced_subtitles)} synced subtitle stream(s)")
    
    # Map streams
    ffmpeg_cmd.extend([
        '-map', '0:v:0',                   # Map video stream from Input 0
        '-map', f'0:{ref_stream_idx}',     # Map reference audio
        '-map', '1:a:0',                   # Map synced foreign audio stream from Input 1
    ])
    
    # Map subtitle streams
    if synced_subtitles:
        for i, sub_info in enumerate(synced_subtitles):
            sub_input_idx = i + 2  # 0=ref video, 1=audio, 2+=subtitles
            ffmpeg_cmd.extend(['-map', f'{sub_input_idx}:s'])
    
    # Codec settings
    ffmpeg_cmd.extend([
        '-c:v', 'copy',                    # Copy video stream without re-encoding
        '-c:a:0', 'copy',                  # Copy original reference audio without re-encoding
        '-c:a:1', args.mux_foreign_codec,  # Codec for the second audio track (synced)
    ])

    # Add bitrate only if re-encoding
    if args.mux_foreign_codec != 'copy':
        ffmpeg_cmd.extend(['-b:a:1', args.mux_foreign_bitrate])
    
    # Add subtitle codec - use mov_text for MP4, srt for MKV
    if synced_subtitles:
        subtitle_codec = 'mov_text' if args.output_video.lower().endswith('.mp4') else 'srt'
        ffmpeg_cmd.extend(['-c:s', subtitle_codec])
        
        # Add language metadata for subtitles
        for i, sub_info in enumerate(synced_subtitles):
            ffmpeg_cmd.extend(['-metadata:s:s:{}'.format(i), f"language={sub_info['language']}"])

    # Metadata (REMOVED language tags as requested)
    ffmpeg_cmd.extend([
        # '-metadata:s:a:0', f'language={args.ref_lang}',  # Removed
        # '-metadata:s:a:0', 'title=Original', # Optional: Add titles if desired
        # '-disposition:s:a:0', 'default', # Optional: Set dispositions if desired
        # '-metadata:s:a:1', f'language={args.foreign_lang}', # Removed
        # '-metadata:s:a:1', 'title=Foreign Synced', # Optional
        # '-disposition:s:a:1', '0', # Optional
        '-y',
        args.output_video
    ])

    # Execute the muxing command
    success, _ = run_ffmpeg(ffmpeg_cmd, "Mux Final Video")

    # Determine if the WAV file was temporary (i.e., not specified by user)
    is_temp_wav = args.output_audio_original is None

    # Attempt to delete the audio file *if* it was temporary, regardless of muxing success
    if os.path.exists(args.output_audio) and is_temp_wav:
        try:
            os.remove(args.output_audio)
            logger.info(f"  -> Deleted temporary audio file: {args.output_audio}")
        except Exception as e:
            logger.warning(f"  Note: Could not delete temporary audio file '{args.output_audio}': {e}")
    elif not is_temp_wav:
        logger.info(f"  Keeping user-specified synchronized audio file: {args.output_audio}")


    if not success:
        logger.error(f"Muxing failed.")
        return False

    logger.info(f"---=== Muxing Stage Finished Successfully ===---")
    return True


# --- Main Execution ---
# --- SUBTITLE SYNCHRONIZATION FUNCTIONS ---
def seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt_time_to_seconds(srt_time):
    """Convert SRT timestamp format (HH:MM:SS,mmm) to seconds"""
    time_parts, millis = srt_time.split(',')
    h, m, s = map(int, time_parts.split(':'))
    return h * 3600 + m * 60 + s + int(millis) / 1000.0


def parse_srt_file(srt_path):
    """Parse SRT file into list of subtitle entries"""
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(srt_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read subtitle file {srt_path}: {e}")
            return None
    
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        timing_line = lines[1]
        timing_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timing_line)
        
        if not timing_match:
            continue
        
        start_time = srt_time_to_seconds(timing_match.group(1))
        end_time = srt_time_to_seconds(timing_match.group(2))
        text = '\n'.join(lines[2:])
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles


def write_srt_file(srt_path, subtitles):
    """Write subtitles to SRT file"""
    try:
        with open(srt_path, 'w', encoding='utf-8') as f:
            for idx, sub in enumerate(subtitles, 1):
                f.write(f"{idx}\n")
                f.write(f"{seconds_to_srt_time(sub['start'])} --> {seconds_to_srt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
        return True
    except Exception as e:
        logger.error(f"Failed to write subtitle file {srt_path}: {e}")
        return False


def adjust_subtitle_timing(subtitles, segment_anchors, ref_delay):
    """Adjust subtitle timing based on audio segments"""
    adjusted_subtitles = []
    num_segments = len(segment_anchors) - 1
    skipped_outside = 0
    
    logger.debug(f"\nSubtitle timing adjustment:")
    logger.debug(f"  Total subtitles: {len(subtitles)}")
    logger.debug(f"  Segments: {num_segments}")
    logger.debug(f"  Reference delay: {ref_delay:.3f}s (NOT added to subtitles - they sync to video)")
    logger.debug(f"  Segment ranges:")
    for i in range(num_segments):
        ref_s, foreign_s = segment_anchors[i]
        ref_e, foreign_e = segment_anchors[i + 1]
        logger.debug(f"    Seg {i+1}: foreign=[{foreign_s:.2f}-{foreign_e:.2f}s], ref=[{ref_s:.2f}-{ref_e:.2f}s]")
    
    for idx, sub in enumerate(subtitles):
        segment_idx = None
        for i in range(num_segments):
            foreign_start = segment_anchors[i][1]
            foreign_end = segment_anchors[i + 1][1]
            
            if foreign_start <= sub['start'] < foreign_end:
                segment_idx = i
                break
        
        if segment_idx is None:
            skipped_outside += 1
            if idx < 3 or idx >= len(subtitles) - 3:  # Log first and last few
                logger.debug(f"  Sub {idx+1} at {sub['start']:.2f}s: OUTSIDE all segments")
            continue
        
        ref_seg_start = segment_anchors[segment_idx][0]
        ref_seg_end = segment_anchors[segment_idx + 1][0]
        foreign_seg_start = segment_anchors[segment_idx][1]
        foreign_seg_end = segment_anchors[segment_idx + 1][1]
        
        ref_seg_duration = ref_seg_end - ref_seg_start
        foreign_seg_duration = foreign_seg_end - foreign_seg_start
        
        stretch_factor = ref_seg_duration / foreign_seg_duration if foreign_seg_duration > 0 else 1.0
        
        relative_start = sub['start'] - foreign_seg_start
        relative_end = sub['end'] - foreign_seg_start
        
        new_start = ref_seg_start + (relative_start * stretch_factor)
        new_end = ref_seg_start + (relative_end * stretch_factor)
        
        # NOTE: Do NOT add ref_delay here!
        # Subtitles sync to video timeline, which doesn't have the audio padding
        # Only the foreign audio gets ref_delay added to match reference audio
        
        if idx < 3:  # Log first few for debugging
            logger.debug(f"  Sub {idx+1}: {sub['start']:.2f}s -> {new_start:.2f}s (seg {segment_idx+1}, stretch {stretch_factor:.4f}x)")
        
        adjusted_subtitles.append({
            'start': max(0, new_start),
            'end': max(0, new_end),
            'text': sub['text']
        })
    
    if skipped_outside > 0:
        logger.info(f"  Note: Skipped {skipped_outside} subtitles outside segment boundaries")
    
    return adjusted_subtitles


def get_subtitle_streams(video_path):
    """Get list of subtitle streams from video file"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 's',
        '-show_entries', 'stream=index:stream_tags=language',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        
        subtitle_streams = []
        for stream in streams:
            stream_idx = stream.get('index')
            tags = stream.get('tags', {})
            lang = tags.get('language', 'unknown')
            subtitle_streams.append({'index': stream_idx, 'language': lang})
        
        return subtitle_streams
    except Exception as e:
        logger.debug(f"Failed to get subtitle streams: {e}")
        return []


def extract_subtitle_stream(video_path, stream_idx, output_path):
    """Extract subtitle stream to SRT file"""
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        '-i', video_path,
        '-map', f'0:{stream_idx}',
        '-c:s', 'srt',
        '-y', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def sync_subtitles(foreign_video, segment_anchors, ref_delay, temp_dir, foreign_lang):
    """Synchronize subtitle streams from foreign video"""
    logger.info("\n--- Subtitle Synchronization Stage ---")
    
    subtitle_streams = get_subtitle_streams(foreign_video)
    
    if not subtitle_streams:
        logger.info("> No subtitle streams found")
        return []
    
    logger.info(f"Found {len(subtitle_streams)} subtitle stream(s)")
    
    synced_subtitle_files = []
    
    for stream in subtitle_streams:
        stream_idx = stream['index']
        stream_lang = stream['language']
        
        logger.info(f"Processing subtitle stream {stream_idx} (lang: {stream_lang})")
        
        original_srt = os.path.join(temp_dir, f"subtitle_original_{stream_idx}.srt")
        if not extract_subtitle_stream(foreign_video, stream_idx, original_srt):
            logger.warning(f"  Failed to extract subtitle stream {stream_idx}")
            continue
        
        subtitles = parse_srt_file(original_srt)
        if not subtitles:
            logger.warning(f"  No subtitles parsed from stream {stream_idx}")
            continue
        
        logger.info(f"  Parsed {len(subtitles)} subtitle entries")
        
        adjusted_subtitles = adjust_subtitle_timing(subtitles, segment_anchors, ref_delay)
        logger.info(f"  Adjusted {len(adjusted_subtitles)} subtitle entries")
        
        synced_srt = os.path.join(temp_dir, f"subtitle_synced_{stream_idx}.srt")
        if write_srt_file(synced_srt, adjusted_subtitles):
            synced_subtitle_files.append({
                'path': synced_srt,
                'language': stream_lang,
                'original_index': stream_idx
            })
            logger.info(f"  [OK] Synced subtitle saved")
    
    if synced_subtitle_files:
        logger.info(f"[OK] Synced {len(synced_subtitle_files)} subtitle stream(s)")
    
    return synced_subtitle_files


def main():
    global FFMPEG_EXEC, FFPROBE_EXEC
    parser = argparse.ArgumentParser(
        description="AVSync: Synchronizes foreign audio to a reference video using visual anchors and precise audio timing.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Example Usage:
  # Basic usage (muxed video output is mandatory)
  python gs.py ref_video.mkv foreign_video.mkv output_video.mkv --ref_lang eng --foreign_lang spa

  # Specify audio streams by index instead of language (Use absolute stream indices shown)
  python gs.py ref_video.mkv foreign_video.mkv output_video.mkv --ref_stream_idx 1 --foreign_stream_idx 2

  # Set minimum segment duration for audio filtering to 10 seconds (default is 5)
  python gs.py ref_video.mkv foreign_video.mkv output_video.mkv --min_segment_duration 10

  # Keep the synchronized WAV file separately
  python gs.py ref_video.mkv foreign_video.mkv output_video.mkv --output_audio synced_audio.wav

  # Generate QC images and segment CSV along with the video
  python gs.py ref_video.mkv foreign_video.mkv output_video.mkv --qc_output_dir ./qc_images --output_csv segments.csv

Workflow:
1. Extracts scene change frames from both videos.
2. Matches frames between videos using template matching within a calculated time window (+/- {MATCH_WINDOW_PERCENT*100}% of ref duration).
3. Filters matches based on similarity and temporal consistency.
4. Extracts audio tracks based on language tags or specified absolute indices.
5. Determines audio content boundaries and filters anchor points based on minimum segment duration and duration difference.
6. Processes audio segments iteratively to match reference timing precisely.
7. Concatenates processed segments and applies start delay.
8. (Optional) Generates QC images and/or segment info CSV.
9. Muxes the reference video, original audio, and synchronized foreign audio into the final output video.
"""
    )
    # --- Input/Output Arguments ---
    parser.add_argument("ref_video", help="Path to the Reference video file (e.g., original language version).")
    parser.add_argument("foreign_video", help="Path to the Foreign video file (e.g., translated language version to be synced).")
    parser.add_argument("output_video", help="Path for the final muxed video file including reference video, reference audio, and synced foreign audio.")
    parser.add_argument("--output_audio", metavar="WAV_PATH", default=None, help="Optional: Path to save the synchronized audio as WAV file. If not specified, a temporary file will be used and deleted after muxing.")
    parser.add_argument("--output_csv", metavar="CSV_PATH", default=None, help="Optional: Path to save segment timing information in a CSV file. By default, no CSV is generated.")
    parser.add_argument("--no_subtitles", action='store_true', help="Skip subtitle synchronization (enabled by default)")

    parser.add_argument("--qc_output_dir", metavar="QC_DIR", default=None, help="Optional: Directory to save side-by-side QC images. By default, no QC images are generated.")

    # --- Image Pairing Arguments ---
    img_group = parser.add_argument_group('Image Pairing Parameters')
    img_group.add_argument("--scene_threshold", type=float, default=0.25, help="FFmpeg scene change detection threshold (0.0-1.0). Lower values detect more changes. (Default: 0.25)")
    img_group.add_argument("--match_threshold", type=float, default=0.7, help="OpenCV template matching score threshold (0.0-1.0) for considering frames a match. (Default: 0.7)")
    img_group.add_argument("--similarity_threshold", type=int, default=4, help="Perceptual hash (pHash) difference threshold for filtering similar reference frames. Lower values mean stricter filtering. Use -1 to disable. (Default: 4)")
    img_group.add_argument("--force_sync_points", type=str, default=None, metavar="SYNC_SPEC",
        help="Force sync points at specific timestamps where auto-detection fails. "
             "Format: 'HH:MM:SS:MS>HH:MM:SS:MS,HH:MM:SS:MS>HH:MM:SS:MS,...' "
             "Example: '00:00:10:500>00:00:10:800,00:02:00:300>00:02:01:000' creates sync points "
             "at ref 10.5s->foreign 10.8s and ref 120.3s->foreign 121.0s. "
             "These bypass scene detection and are guaranteed sync points.")
    # Note: Match search window is now calculated automatically, not a direct argument

    # --- Audio Processing Arguments ---
    audio_group = parser.add_argument_group('Audio Processing Parameters')
    audio_group.add_argument("--ref_lang", default=DEFAULT_REF_LANG, help=f"Reference audio language code (3-letter ISO 639-2/T) for stream selection. (Default: {DEFAULT_REF_LANG})")
    audio_group.add_argument("--foreign_lang", default=DEFAULT_FOREIGN_LANG, help=f"Foreign audio language code (3-letter ISO 639-2/T) for stream selection. (Default: {DEFAULT_FOREIGN_LANG})")
    audio_group.add_argument("--db_threshold", type=float, default=DEFAULT_DB_THRESHOLD, help=f"Audio detection threshold (dBFS) to find start/end of content. (Default: {DEFAULT_DB_THRESHOLD:.1f})")
    audio_group.add_argument("--min_segment_duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION, help=f"Minimum duration (seconds) for a reference audio segment to be kept during anchor filtering. (Default: {DEFAULT_MIN_SEGMENT_DURATION:.1f})") # New Argument
    audio_group.add_argument("--first_segment_adjust", type=float, default=0.0, help="Adjustment in milliseconds for the FIRST segment. Positive = add padding at start, Negative = trim from start. Applied BEFORE atempo processing (Default: 0.0ms).")
    audio_group.add_argument("--last_segment_adjust", type=float, default=0.0, help="Adjustment in milliseconds for the LAST segment. Positive = add padding at end, Negative = trim from end. Applied BEFORE atempo processing (Default: 0.0ms).")
    audio_group.add_argument("--ref_stream_idx", type=int, default=None, help="Force specific *absolute* audio stream index for reference video (e.g., 1, 2, ...). Overrides --ref_lang.")
    audio_group.add_argument("--foreign_stream_idx", type=int, default=None, help="Force specific *absolute* audio stream index for foreign video. Overrides --foreign_lang.")
    audio_group.add_argument("--auto_detect", action="store_true", help="Skip audio stream selection prompts and use auto-detection.")

    # --- Muxing Arguments ---
    mux_group = parser.add_argument_group('Muxing Parameters')
    mux_group.add_argument("--mux_foreign_codec", default=DEFAULT_MUX_ACODEC, help=f"Audio codec for the synced foreign track in the muxed output (e.g., 'aac', 'ac3', 'copy'). (Default: {DEFAULT_MUX_ACODEC})")
    mux_group.add_argument("--mux_foreign_bitrate", default=DEFAULT_MUX_ABITRATE, help=f"Audio bitrate for the synced foreign track if re-encoding (e.g., '192k', '320k'). (Default: {DEFAULT_MUX_ABITRATE})")


    # --- Caching Arguments ---
    cache_group = parser.add_argument_group('Caching Parameters')
    cache_group.add_argument("--use-cache", action="store_true", default=True, help="Use checkpoint cache for faster iterations (default: enabled)")
    cache_group.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable checkpoint caching")

    # --- Logging Arguments ---
    log_group = parser.add_argument_group('Logging Parameters')
    log_group.add_argument("--verbose", "-v", action="store_true", help="Show detailed output on console (default: minimal with progress bar)")
    log_group.add_argument("--show-warnings", action="store_true", help="Show warning messages on console (default: suppressed for clean progress)")
    log_group.add_argument("--log-file", type=str, default=None, metavar="PATH", help="Custom log file path. Default: <output_name>_avsync.log")
    log_group.add_argument("--no-log", action="store_true", help="Disable file logging (console only)")

    args = parser.parse_args()

    # --- Initialize Logging ---
    log_file_path = None
    # --- Set warning display preference before logging setup ---
    ConsoleFilter.show_warnings = args.show_warnings
    
    if not args.no_log:
        log_file_path = args.log_file if args.log_file else get_log_path(args.output_video)
    
    global logger
    logger, log_file_path = setup_logging(log_file=log_file_path, verbose=args.verbose)

    # --- Handle Temporary WAV File ---
    args.output_audio_original = args.output_audio # Store if user specified a path
    if args.output_audio is None:
        # Create a temporary file for the synchronized audio
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        args.output_audio = temp_audio_file.name
        temp_audio_file.close()  # Close but don't delete yet
        logger.info(f"Using temporary WAV file for processing: {args.output_audio}")
        # This file will be deleted by run_muxing

    # Assign output_video to mux_output for compatibility with any remaining internal refs if needed
    args.mux_output = args.output_video

    overall_start_time = time.time()
    
    # Professional startup banner
    logger.info("")
    logger.info("=" * 70)
    logger.info(" AVSync v12 - Audio/Video Synchronization Engine")
    logger.info("=" * 70)
    logger.info(f"Reference video : {args.ref_video}")
    logger.info(f"Foreign video   : {args.foreign_video}")
    logger.info(f"Output video    : {args.output_video}")
    if log_file_path:
        logger.info(f"Log file        : {log_file_path}")
    if args.output_audio_original:
        logger.info(f"Output audio    : {args.output_audio_original}")
    if args.output_csv:
        logger.info(f"Segment CSV     : {args.output_csv}")
    if args.qc_output_dir:
        logger.info(f"QC images       : {args.qc_output_dir}")
    logger.info(f"Caching         : {'Enabled' if args.use_cache else 'Disabled'}")
    logger.info("-"*70)


    # --- Find Executables ---
    FFMPEG_EXEC = find_executable("ffmpeg")
    FFPROBE_EXEC = find_executable("ffprobe")
    if not FFMPEG_EXEC or not FFPROBE_EXEC:
        logger.error("FATAL: Required 'ffmpeg' and/or 'ffprobe' executable not found in system PATH.")
        sys.exit(1)
    logger.info(f"Using ffmpeg: {FFMPEG_EXEC}")
    logger.info(f"Using ffprobe: {FFPROBE_EXEC}")

    # --- Validate Input/Output Paths ---
    if not os.path.isfile(args.ref_video): logger.error(f"Reference video not found: {args.ref_video}"); sys.exit(1)
    if not os.path.isfile(args.foreign_video): logger.error(f"Foreign video not found: {args.foreign_video}"); sys.exit(1)

    # --- Handle Audio Stream Selection ---
    # For reference video streams
    if args.ref_stream_idx is None and not args.auto_detect:
        # Prompt user, returns the absolute stream index if selected
        args.ref_stream_idx = prompt_user_for_audio_stream(args.ref_video, "reference")
        if args.ref_stream_idx is not None:
            logger.info(f"User selected Reference Stream Index: {args.ref_stream_idx}")

    # For foreign video streams
    if args.foreign_stream_idx is None and not args.auto_detect:
        # Prompt user, returns the absolute stream index if selected
        args.foreign_stream_idx = prompt_user_for_audio_stream(args.foreign_video, "foreign")
        if args.foreign_stream_idx is not None:
            logger.info(f"User selected Foreign Stream Index: {args.foreign_stream_idx}")

    # Log the final stream selection choices
    # Note: ref_stream_idx and foreign_stream_idx now hold the *absolute* index if specified or selected
    if args.ref_stream_idx is not None:
        logger.info(f"Ref Stream Index (Absolute): {args.ref_stream_idx} (User selected or forced)")
    else:
        logger.info(f"Ref Lang: {args.ref_lang} (Will auto-detect)")

    if args.foreign_stream_idx is not None:
        logger.info(f"Foreign Stream Index (Absolute): {args.foreign_stream_idx} (User selected or forced)")
    else:
        logger.info(f"Foreign Lang: {args.foreign_lang} (Will auto-detect)")

    logger.info(f"Audio Threshold: {args.db_threshold} dB")
    logger.info(f"Min Segment Duration: {args.min_segment_duration}s") # Log new parameter
    if args.first_segment_adjust != 0.0 or args.last_segment_adjust != 0.0:
        logger.info(f"First Segment Adjust: {args.first_segment_adjust:+.1f}ms, Last Segment Adjust: {args.last_segment_adjust:+.1f}ms")
    logger.info(f"Scene Threshold: {args.scene_threshold}, Match Threshold: {args.match_threshold}, Similarity Threshold: {args.similarity_threshold}")
    if args.force_sync_points:
        logger.info(f"Forced Sync Points: {args.force_sync_points}")
    logger.info(f"Frame Match Window: Calculated as {MATCH_WINDOW_PERCENT*100}% of reference video duration") # Log new behavior

    # Validate output directories are writable for all specified outputs
    output_paths_to_check = [args.output_video, args.output_audio, args.output_csv]
    for path in output_paths_to_check:
        if path: # Only check paths that are actually set
            try:
                out_dir = os.path.dirname(os.path.abspath(path))
                if not out_dir: # Handle case where path is just a filename in cwd
                    out_dir = '.'
                os.makedirs(out_dir, exist_ok=True) # Create dir if it doesn't exist
                if not os.access(out_dir, os.W_OK):
                    raise OSError(f"Output directory is not writable: {out_dir}")
                # Warn about overwriting files (except for the temp audio)
                if path != args.output_audio or args.output_audio_original: # Don't warn for default temp audio path
                    if os.path.exists(path) and os.path.isfile(path):
                        logger.warning(f"Output file '{os.path.basename(path)}' exists and will be overwritten.")
            except Exception as e:
                logger.error(f"Output path validation failed for '{path}': {e}")
                # Clean up temporary audio file if created and validation fails early
                if args.output_audio_original is None and args.output_audio and os.path.exists(args.output_audio):
                    try: os.remove(args.output_audio)
                    except Exception: pass
                sys.exit(1)
    # Validate QC dir separately if specified
    if args.qc_output_dir:
         try:
             qc_dir_abs = os.path.abspath(args.qc_output_dir)
             os.makedirs(qc_dir_abs, exist_ok=True)
             if not os.access(qc_dir_abs, os.W_OK):
                 raise OSError(f"QC output directory is not writable: {qc_dir_abs}")
             if os.path.exists(qc_dir_abs) and not os.path.isdir(qc_dir_abs):
                  raise OSError(f"QC output path exists but is not a directory: {qc_dir_abs}")
         except Exception as e:
             logger.error(f"QC output directory validation failed for '{args.qc_output_dir}': {e}")
             # Clean up temporary audio file if created
             if args.output_audio_original is None and args.output_audio and os.path.exists(args.output_audio):
                 try: os.remove(args.output_audio)
                 except Exception: pass
             sys.exit(1)


    # --- Main Process ---
    final_ref_delay = None
    audio_sync_success = False
    muxing_success = False
    final_ref_stream_idx = None # Store the absolute index used for sync/mux
    final_segment_anchors = None # Store anchors for QC
    temp_dir_obj = None # To hold the TemporaryDirectory object

    try:
        # Use a temporary directory for intermediate files (frames, wav segments)
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gsync_")
        temp_dir = temp_dir_obj.name # Get the path string
        logger.info(f"\nUsing temporary directory: {temp_dir}")

        # === Stage 1: Image Pairing ===
        # Try to load from cache if enabled
        visual_anchors_details = None
        if args.use_cache:
            cache_path = get_cache_path(args)
            checkpoint = load_checkpoint(cache_path)
            if checkpoint:
                visual_anchors_details = checkpoint.get('visual_anchors_details')
                if visual_anchors_details:
                    logger.info("[CACHE] Using cached visual anchors, skipping frame extraction and matching")
        
        # Run image pairing if not loaded from cache
        if visual_anchors_details is None:
            visual_anchors_details = run_image_pairing_stage(
                ref_video_path=args.ref_video,
                foreign_video_path=args.foreign_video,
                temp_dir=temp_dir,
                scene_threshold=args.scene_threshold,
                match_threshold=args.match_threshold,
                similarity_threshold=args.similarity_threshold
            )
            if visual_anchors_details is None:
                raise RuntimeError("Image Pairing Stage Failed: No visual anchors generated.")
            
            # Save checkpoint if caching is enabled
            if args.use_cache:
                cache_path = get_cache_path(args)
                save_checkpoint(cache_path, visual_anchors_details)

        # === Stage 1.5: Inject Forced Sync Points (if specified) ===
        forced_sync_points = parse_force_sync_points(args.force_sync_points)
        if forced_sync_points:
            visual_anchors_details = inject_forced_sync_points(visual_anchors_details, forced_sync_points)

        # === Stage 2: Audio Synchronization (Iterative Method) ===
        # This function now handles finding streams, filtering anchors, processing, concatenating, and padding.
        # It outputs to args.output_audio (which might be temporary)
        # It internally uses args.ref_stream_idx/foreign_stream_idx if set, or detects by lang.
        # It returns the delay and the final anchors used.
        final_ref_delay, final_segment_anchors = run_progressive_sync_iterative(
            args=args, # Pass all args
            visual_anchors_details=visual_anchors_details,
            output_audio_path=args.output_audio,
            temp_dir=temp_dir,
            db_threshold=args.db_threshold,
            min_segment_duration=args.min_segment_duration # Pass new argument
        )

        if final_ref_delay is None or final_segment_anchors is None:
            raise RuntimeError("Audio Synchronization Stage Failed.")

        audio_sync_success = True

        # Determine the *final* reference stream index used for muxing.
        # It might have been auto-detected within run_progressive_sync_iterative if not forced.
        final_ref_stream_idx = args.ref_stream_idx # Use forced/selected index if available
        if final_ref_stream_idx is None:
            # If it was auto-detected, find it again (necessary for muxing)
            temp_ref_streams = get_stream_info(args.ref_video)
            final_ref_stream_idx = find_audio_stream_index_by_lang(temp_ref_streams, args.ref_lang)

        if final_ref_stream_idx is None:
             # This is critical for muxing, raise error if still not found
             raise RuntimeError("Could not determine *final* reference audio stream index for muxing.")
        else:
             logger.info(f"Determined final absolute reference stream index for muxing: {final_ref_stream_idx}")


        # === Stage 2.5: Generate QC Images (Optional & Conditional) ===
        if args.qc_output_dir: # Check if requested
            if visual_anchors_details and final_segment_anchors:
                ref_extract_path = os.path.join(temp_dir, "Extracted_Reference")
                foreign_extract_path = os.path.join(temp_dir, "Extracted_Foreign")
                # Check if extract paths exist before calling QC function
                if os.path.isdir(ref_extract_path) and os.path.isdir(foreign_extract_path):
                    create_qc_images(
                        visual_anchors_details=visual_anchors_details,
                        final_segment_anchors=final_segment_anchors,
                        ref_extract_path=ref_extract_path,
                        foreign_extract_path=foreign_extract_path,
                        qc_output_dir=args.qc_output_dir
                    )
                else:
                    logger.warning("Skipping QC image generation: Frame extraction directories not found in temp dir.")
            else:
                 logger.warning("Skipping QC image generation: Missing anchor details or final anchors.")

        # === Stage 2.6: Subtitle Synchronization (Optional) ===
        synced_subtitles = []
        if not args.no_subtitles:
            synced_subtitles = sync_subtitles(
                foreign_video=args.foreign_video,
                segment_anchors=final_segment_anchors,
                ref_delay=final_ref_delay,
                temp_dir=temp_dir,
                foreign_lang=args.foreign_lang
            )

        # === Stage 3: Muxing (Now the default final step) ===
        if audio_sync_success:
            # Pass the determined *final* absolute ref_stream_idx to the muxing function
            muxing_success = run_muxing(args, final_ref_stream_idx, synced_subtitles)
            if not muxing_success:
                # Raise error if muxing failed, as it's the primary output
                raise RuntimeError("Muxing Stage Failed.")
        else:
             # This case should have been caught earlier, but defensive coding
             raise RuntimeError("Audio synchronization did not complete successfully, cannot mux.")

    except RuntimeError as e:
        logger.error(f"Process aborted due to error: {e}")
        # Clean up temporary audio file if it exists and wasn't user specified
        if args.output_audio_original is None and args.output_audio and os.path.exists(args.output_audio):
            try:
                os.remove(args.output_audio)
                logger.info(f"Cleaned up temporary audio file: {args.output_audio}")
            except Exception as del_e:
                logger.warning(f"Could not clean up temp audio file {args.output_audio}: {del_e}")
        if temp_dir_obj:
            try: temp_dir_obj.cleanup()
            except Exception as clean_e: logger.warning(f"Error cleaning up temp dir: {clean_e}")
            else: logger.info("Attempted temporary directory cleanup.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected critical error occurred:", exc_info=True)
        # Clean up temporary audio file if it exists and wasn't user specified
        if args.output_audio_original is None and args.output_audio and os.path.exists(args.output_audio):
            try:
                os.remove(args.output_audio)
                logger.info(f"Cleaned up temporary audio file: {args.output_audio}")
            except Exception as del_e:
                 logger.warning(f"Could not clean up temp audio file {args.output_audio}: {del_e}")
        if temp_dir_obj:
            try: temp_dir_obj.cleanup()
            except Exception as clean_e: logger.warning(f"Error cleaning up temp dir: {clean_e}")
            else: logger.info("Attempted temporary directory cleanup.")
        sys.exit(1)
    finally:
        # Ensure temporary directory is cleaned up even if errors occur after its creation but before muxing/final cleanup
        if temp_dir_obj:
            try: temp_dir_obj.cleanup()
            except Exception as clean_e: logger.warning(f"Final attempt to clean up temp dir failed: {clean_e}")
            # else: logger.info("Temporary directory cleaned up.") # Avoid duplicate message if already logged


    # --- Final Summary ---
    overall_elapsed_time = time.time() - overall_start_time
    logger.info("\n===== \1 =====")
    final_exit_code = 1 # Default to error

    # Core success means audio sync AND muxing worked
    if audio_sync_success and muxing_success:
        logger.info(f"-> Muxed Video Created:      {args.output_video}")
        logger.info(f"-> Calculated Reference Delay: {final_ref_delay:.3f} seconds")

        # Log optional outputs only if they were requested and presumably created
        if args.output_audio_original:
             if os.path.exists(args.output_audio_original):
                 logger.info(f"-> Synchronized Audio Saved: {args.output_audio_original}")
             else:
                  logger.warning(f"->Â  Expected audio file not found: {args.output_audio_original}") # Should not happen if sync succeeded

        if args.output_csv:
            if os.path.exists(args.output_csv):
                 logger.info(f"-> Segment CSV File Written: {args.output_csv}")
            else:
                 logger.warning(f"->Â  Expected CSV file not found: {args.output_csv}") # Might happen if writing failed but process continued

        if args.qc_output_dir:
             if os.path.isdir(args.qc_output_dir):
                 logger.info(f"-> QC Image Directory:       {args.qc_output_dir}")
             else:
                 logger.warning(f"->Â  Expected QC directory not found: {args.qc_output_dir}")

        logger.info(f"Total time elapsed  : {overall_elapsed_time:.1f}s")
        logger.info("===== \1 =====")
        final_exit_code = 0 # Success

    else:
        # Handle various failure modes
        if not audio_sync_success:
            logger.error("-> Audio Synchronization Stage Failed.")
        elif not muxing_success:
             logger.error("-> Muxing Stage Failed.") # Muxing is now essential

        logger.error("\nScript failed during processing.")
        logger.info(f"Total Elapsed Time: {overall_elapsed_time:.2f} seconds")
        logger.info("===== \1 =====")
        final_exit_code = 1 # Error

    sys.exit(final_exit_code)


if __name__ == "__main__":
    main()