#!/usr/bin/env python3
"""
Batch Video Synchronization Tool

Process multiple videos using configuration file with language detection support.
"""

import os
import sys
import json
import argparse
import glob
import re
import subprocess
from pathlib import Path
import logging
from tqdm import tqdm  # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("batch_sync")

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def verify_setup():
    """Verify that the environment is correctly set up"""
    issues = []
    
    # Check if AudioSynchronizer.py exists
    if not os.path.exists("AudioSynchronizer.py"):
        issues.append("AudioSynchronizer.py not found in the current directory")
    
    # Check if ffprobe and ffmpeg are available
    try:
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ffprobe not found in PATH (required for audio track detection)")
    
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ffmpeg not found in PATH (required for video processing)")
    
    # Check for necessary Python libraries
    try:
        import tqdm
    except ImportError:
        issues.append("tqdm Python library not installed (pip install tqdm)")
    
    return issues

def detect_language_track(video_path, config):
    """
    Detect the track index that matches the preferred language
    
    Parameters:
    video_path (str): Path to the video file
    config (dict): Configuration with language preferences
    
    Returns:
    int: Detected track index or None if not found
    """
    # Get language preferences
    primary_lang = config.get("language_preferences", {}).get("primary", "")
    fallback_lang = config.get("language_preferences", {}).get("fallback", "")
    language_map = config.get("language_map", {})
    
    # Get primary language codes/names to match
    primary_matches = language_map.get(primary_lang, [primary_lang.lower()])
    fallback_matches = language_map.get(fallback_lang, [fallback_lang.lower()])
    
    try:
        # Run ffprobe to get stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'a', video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # First pass: Look for exact primary language match
        for i, stream in enumerate(data.get('streams', [])):
            lang = stream.get('tags', {}).get('language', '').lower()
            title = stream.get('tags', {}).get('title', '').lower()
            
            # Check both language code and title
            for match in primary_matches:
                if match.lower() in lang or match.lower() in title:
                    logger.info(f"Found primary language '{primary_lang}' in track {i}")
                    return i
        
        # Second pass: Look for fallback language
        for i, stream in enumerate(data.get('streams', [])):
            lang = stream.get('tags', {}).get('language', '').lower()
            title = stream.get('tags', {}).get('title', '').lower()
            
            # Check both language code and title
            for match in fallback_matches:
                if match.lower() in lang or match.lower() in title:
                    logger.info(f"Found fallback language '{fallback_lang}' in track {i}")
                    return i
        
        # If no matches found, return the first audio track
        if data.get('streams'):
            logger.warning(f"No language match found. Using first audio track (index 0)")
            return 0
        else:
            logger.error(f"No audio streams found in {video_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error detecting language track: {e}")
        return None

def get_video_files(config):
    """
    Get video files to process based on configuration
    
    Parameters:
    config (dict): Configuration with batch process settings
    
    Returns:
    list: List of (reference_path, foreign_path, output_path) tuples
    """
    batch_config = config.get("batch_process", {})
    ref_dir = batch_config.get("reference_dir")
    foreign_dir = batch_config.get("foreign_dir")
    output_dir = batch_config.get("output_dir", "output")
    episode_range = batch_config.get("episode_range", [1, 100])
    file_pattern = batch_config.get("file_pattern", "*.mkv")
    file_format = batch_config.get("file_format", "{episode:03d}.mkv")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Use episode range and file format
    if episode_range and file_format:
        file_pairs = []
        
        for ep in range(episode_range[0], episode_range[1] + 1):
            episode_file = file_format.format(episode=ep)
            ref_path = os.path.join(ref_dir, episode_file)
            foreign_path = os.path.join(foreign_dir, episode_file)
            output_path = os.path.join(output_dir, episode_file)
            
            # Check if files exist
            if os.path.exists(ref_path) and os.path.exists(foreign_path):
                file_pairs.append((ref_path, foreign_path, output_path))
            else:
                logger.warning(f"Skipping episode {ep} - files not found")
        
        return file_pairs
    
    # Option 2: Use glob pattern matching
    # This is more flexible but requires consistent naming
    ref_files = sorted(glob.glob(os.path.join(ref_dir, file_pattern)))
    
    file_pairs = []
    for ref_path in ref_files:
        # Extract filename and find matching foreign file
        filename = os.path.basename(ref_path)
        foreign_path = os.path.join(foreign_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(foreign_path):
            file_pairs.append((ref_path, foreign_path, output_path))
        else:
            logger.warning(f"Skipping {filename} - no matching foreign file")
    
    return file_pairs

def build_command(ref_path, foreign_path, output_path, track_index, config):
    """
    Build the command to run AudioSynchronizer with the given configuration
    
    Parameters:
    ref_path (str): Path to reference video
    foreign_path (str): Path to foreign video
    output_path (str): Path to output video
    track_index (int): Audio track index to use
    config (dict): Configuration with default settings
    
    Returns:
    list: Command arguments list
    """
    settings = config.get("default_settings", {})
    
    # Start with the base command
    cmd = [
        "python", "AudioSynchronizer.py",
        ref_path, foreign_path, output_path
    ]
    
    # Add feature selection
    feature = settings.get("feature")
    if feature:
        cmd.extend(["--feature", feature])
    
    # Add sample mode
    sample_mode = settings.get("sample_mode")
    if sample_mode:
        cmd.extend(["--sample-mode", sample_mode])
    
    # Add numeric parameters
    if settings.get("max_frames") is not None:
        cmd.extend(["--max-frames", str(settings.get("max_frames"))])
    
    if settings.get("max_comparisons") is not None:
        cmd.extend(["--max-comparisons", str(settings.get("max_comparisons"))])
    
    if settings.get("scene_threshold") is not None:
        cmd.extend(["--scene-threshold", str(settings.get("scene_threshold"))])
    
    if settings.get("db_threshold") is not None:
        cmd.extend(["--db-threshold", str(settings.get("db_threshold"))])
    
    if settings.get("min_duration") is not None:
        cmd.extend(["--min-duration", str(settings.get("min_duration"))])
    
    if settings.get("min_silence") is not None:
        cmd.extend(["--min-silence", str(settings.get("min_silence"))])
    
    # Add boolean flags
    if settings.get("temporal_consistency"):
        cmd.append("--temporal-consistency")
    
    if settings.get("enhanced_preprocessing"):
        cmd.append("--enhanced-preprocessing")
    
    if settings.get("visualize"):
        cmd.append("--visualize")
    
    if settings.get("use_scene_detect"):
        cmd.append("--use-scene-detect")
    
    if settings.get("content_aware"):
        cmd.append("--content-aware")
    
    if settings.get("keep_temp"):
        cmd.append("--keep-temp")
    
    if settings.get("preserve_aspect_ratio"):
        cmd.append("--preserve-aspect-ratio")
    
    # Add detected audio track
    if track_index is not None:
        cmd.extend(["--force-track", str(track_index)])
    
    return cmd

def process_video(ref_path, foreign_path, output_path, config, dry_run=False):
    """
    Process a single video with language detection
    
    Parameters:
    ref_path (str): Path to reference video
    foreign_path (str): Path to foreign video
    output_path (str): Path to output video
    config (dict): Configuration
    dry_run (bool): If True, don't actually execute the command
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(ref_path)}")
        
        # Skip if output already exists
        if os.path.exists(output_path) and not config.get("overwrite", False):
            logger.info(f"Output already exists: {output_path} (skipping)")
            return True
        
        # Detect language track
        track_index = detect_language_track(foreign_path, config)
        if track_index is None:
            logger.error(f"Failed to detect audio track: {foreign_path}")
            return False
        
        # Build and run the command
        cmd = build_command(ref_path, foreign_path, output_path, track_index, config)
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # If dry run, don't actually execute the command
        if dry_run:
            logger.info("DRY RUN: Command would be executed here")
            return True
            
        # Execute the command - DIRECTLY using os.system
        command_str = " ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        logger.debug(f"Executing: {command_str}")
        
        # Run the command
        return_code = os.system(command_str)
        
        if return_code == 0:
            logger.info(f"Successfully processed: {os.path.basename(ref_path)}")
            return True
        else:
            logger.error(f"Command failed with return code: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False

def main():
    """Main entry point for batch processing"""
    # Verify setup
    issues = verify_setup()
    if issues:
        print("ERROR: Setup verification failed!")
        for issue in issues:
            print(f" - {issue}")
        print("\nPlease resolve these issues before continuing.")
        return 1
        
    parser = argparse.ArgumentParser(
        description='Batch process videos with audio synchronization and language detection'
    )
    
    parser.add_argument('--config', default='config.json',
                      help='Path to configuration file (default: config.json)')
    parser.add_argument('--episode', type=int, 
                      help='Process a specific episode number')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing output files')
    parser.add_argument('--list-only', action='store_true',
                      help='List files to be processed without processing them')
    parser.add_argument('--dry-run', action='store_true',
                      help='Build commands but do not execute them (for testing)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Add command-line args to config
    config["overwrite"] = args.overwrite
    config["dry_run"] = args.dry_run
    
    # Get video files
    file_pairs = get_video_files(config)
    
    if args.episode:
        # Filter for specific episode if requested
        episode_str = f"{args.episode:03d}"
        file_pairs = [pair for pair in file_pairs if episode_str in os.path.basename(pair[0])]
        
        if not file_pairs:
            logger.error(f"Episode {args.episode} not found")
            sys.exit(1)
    
    # Print summary
    logger.info(f"Found {len(file_pairs)} video pairs to process")
    
    if args.list_only:
        for ref, foreign, output in file_pairs:
            print(f"Ref: {os.path.basename(ref)}")
            print(f"Foreign: {os.path.basename(foreign)}")
            print(f"Output: {os.path.basename(output)}")
            print("-" * 40)
        return 0
    
    # Process videos
    success_count = 0
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Commands will be built but not executed")
    
    for ref_path, foreign_path, output_path in tqdm(file_pairs, desc="Processing videos"):
        success = process_video(ref_path, foreign_path, output_path, config, dry_run=args.dry_run)
        if success:
            success_count += 1
    
    # Print summary
    logger.info(f"Batch processing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(file_pairs)} videos")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
