import os
import sys
import subprocess
import argparse
import re
from typing import List

# --- Configuration ---
# The name of the main script to be called.
AVSYNC_SCRIPT_NAME = "AVSync_v14.py"

# A tuple of common video file extensions to look for (case-insensitive).
VIDEO_EXTENSIONS = ('.mkv', '.mp4', '.avi', '.mov', '.ts', '.webm')

def run_sync_process(command: List[str]) -> bool:
    """Runs a single AVSync process using subprocess and handles output."""
    try:
        # Use subprocess.run which is modern and straightforward.
        # We don't capture stdout/stderr here, allowing the child process
        # to print directly to the console in real-time.
        result = subprocess.run(command, check=False) # check=False to handle errors manually

        if result.returncode == 0:
            print("  -> SUCCESS: Process finished successfully.")
            return True
        else:
            print(f"  -> ERROR: Process failed with exit code {result.returncode}.")
            return False
    except FileNotFoundError:
        print(f"  -> FATAL ERROR: Could not find the script '{AVSYNC_SCRIPT_NAME}'. Check that the main script path is correct.")
        return False
    except Exception as e:
        print(f"  -> An unexpected error occurred while running the subprocess: {e}")
        return False

def parse_file_list(file_list, list_type = ""):
    """
    Parser that creates a dictionary {EpisodeId:Filename} from list of filenames.
    """
    pairs = {}
    for file in file_list:
        id = re.search(r'S\d{2}E\d{2}', file, re.IGNORECASE)
        if id:
            id = id.group().upper()
            if id in pairs.keys():
                print(f"\nFATAL ERROR: Duplicate episode ids in {list_type} directory.")
                sys.exit(1)
            else:
                pairs[id] = file
    return pairs

def main():
    """
    Main function to parse arguments and orchestrate the batch processing.
    """
    parser = argparse.ArgumentParser(
        description=f"Batch processor for '{AVSYNC_SCRIPT_NAME}'. Finds matching video files in two folders and runs the sync process on each pair. Links files with SxxExx regex.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:

# Basic usage (will skip files that already exist in the output folder):
python AVSync_batch_regex.py "C:\\path\\to\\ref_videos" "C:\\path\\to\\foreign_videos" "C:\\path\\to\\output"

# This version of script matches video files NOT based on name but episode and season number S00E00 using regex.

# Force overwrite of existing files in the output folder:
python AVSync_batch_regex.py ./ref ./foreign ./output --overwrite

# With pass-through arguments for AVSync_v14.py:
# (All arguments after the three folders are passed directly to the main script)
python AVSync_batch_regex.py ./ref ./foreign ./output --ref_lang eng --foreign_lang hin --foreign_tracks all

# Sync all foreign audio tracks with QC images:
python AVSync_batch_regex.py ./ref ./foreign ./output --foreign_lang jpn --foreign_tracks all --qc_output_dir ./qc_images

# Note: --auto_detect is always injected in batch mode to skip interactive prompts.
# A valid --foreign_lang is REQUIRED in batch mode (v14 enforces language metadata).
# If the foreign files have proper language tags in metadata, they will be used automatically.

# It is recommended to use quotes around paths, especially if they contain spaces.
"""
    )

    parser.add_argument("ref_dir", help="The input directory containing the reference video files.")
    parser.add_argument("foreign_dir", help="The input directory containing the foreign video files.")
    parser.add_argument("output_dir", help="The directory where the final muxed video files will be saved.")
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force the script to re-process and overwrite files that already exist in the output directory. Default is to skip them."
    )

    # This is the key to capturing all extra arguments for the child script
    args, unknown_args = parser.parse_known_args()

    # --- 1. Initial Validation ---
    print("--- AVSync Batch Processor ---")

    # Resolve the main script relative to this batch script (not just CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    avsync_script_path = os.path.join(script_dir, AVSYNC_SCRIPT_NAME)
    if not os.path.isfile(avsync_script_path):
        # Fallback: try current working directory
        avsync_script_path = os.path.abspath(AVSYNC_SCRIPT_NAME)
        if not os.path.isfile(avsync_script_path):
            print(f"\nFATAL ERROR: The main script '{AVSYNC_SCRIPT_NAME}' was not found.")
            print(f"Looked in: {script_dir}")
            print(f"Also tried: {os.getcwd()}")
            print(f"Please place '{AVSYNC_SCRIPT_NAME}' next to this batch script or in the current directory.")
            sys.exit(1)

    # Validate input directories
    if not os.path.isdir(args.ref_dir):
        print(f"\nFATAL ERROR: Reference directory not found: {args.ref_dir}")
        sys.exit(1)
    if not os.path.isdir(args.foreign_dir):
        print(f"\nFATAL ERROR: Foreign directory not found: {args.foreign_dir}")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Reference Folder: {os.path.abspath(args.ref_dir)}")
        print(f"Foreign Folder:   {os.path.abspath(args.foreign_dir)}")
        print(f"Output Folder:    {os.path.abspath(args.output_dir)}")
    except OSError as e:
        print(f"\nFATAL ERROR: Could not create output directory '{args.output_dir}': {e}")
        sys.exit(1)

    if args.overwrite:
        print("Mode: Overwriting existing files.")
    else:
        print("Mode: Skipping existing files (use --overwrite to force reprocessing).")
        
    if unknown_args:
        print(f"Pass-through args: {' '.join(unknown_args)}")

    # --- 2. Find and Match Files ---
    print("\n--- Scanning for video files ---")
    
    # Get a list of all files in the reference directory
    try:
        ref_files = sorted([f for f in os.listdir(args.ref_dir) if f.lower().endswith(VIDEO_EXTENSIONS)])
    except OSError as e:
        print(f"\nFATAL ERROR: Could not read reference directory: {e}")
        sys.exit(1)
        
    if not ref_files:
        print("No video files found in the reference directory. Exiting.")
        sys.exit(0)
    
    # Get a list of all files in the foreign directory
    try:
        for_files = sorted([f for f in os.listdir(args.foreign_dir) if f.lower().endswith(VIDEO_EXTENSIONS)])
    except OSError as e:
        print(f"\nFATAL ERROR: Could not read foreign directory: {e}")
        sys.exit(1)
        
    if not for_files:
        print("No video files found in the foreign directory. Exiting.")
        sys.exit(0)

    ref_dict = parse_file_list(ref_files, "reference")
    for_dict = parse_file_list(for_files, "foreign")

    # Files that contained an SxxExx code
    ref_with_id = len(ref_dict)
    for_with_id = len(for_dict)

    # Files that had no recognizable SxxExx code (silently ignored by the parser)
    ref_no_id = len(ref_files) - ref_with_id
    for_no_id = len(for_files) - for_with_id

    # Intersection = pairs we can process
    pairs = set(ref_dict.keys()) & set(for_dict.keys())
    unmatched_ref = sorted(set(ref_dict.keys()) - set(for_dict.keys()))
    unmatched_for = sorted(set(for_dict.keys()) - set(ref_dict.keys()))

    print(f"Reference videos with SxxExx: {ref_with_id}  (ignored {ref_no_id} without episode code)")
    print(f"Foreign videos with SxxExx:   {for_with_id}  (ignored {for_no_id} without episode code)")
    print(f"Matched pairs:                {len(pairs)}")
    if unmatched_ref:
        print(f"  Unmatched reference episodes: {', '.join(unmatched_ref)}")
    if unmatched_for:
        print(f"  Unmatched foreign episodes:   {', '.join(unmatched_for)}")

    # --- 3. Process Each Matched Pair ---
    print("\n--- Starting Batch Processing ---")
    
    success_count = 0
    failed_count = 0
    skipped_existing_count = 0

    for i, pair_id in enumerate(sorted(pairs), 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {pair_id}")
        
        ref_path = os.path.join(args.ref_dir, ref_dict.get(pair_id))
        foreign_path = os.path.join(args.foreign_dir, for_dict.get(pair_id))
        output_path = os.path.join(args.output_dir, ref_dict.get(pair_id))

        if not args.overwrite and os.path.isfile(output_path):
            print(f"  -> SKIPPING: Output file already exists.")
            print(f"     (Found: {output_path})")
            skipped_existing_count += 1
            continue
        
        # Construct the command to run AVSync_v14.py
        # Using sys.executable ensures the same python interpreter is used
        # --auto_detect is injected to ensure non-interactive batch operation
        command = [
            sys.executable,
            avsync_script_path,
            ref_path,
            foreign_path,
            output_path,
            "--auto_detect",
        ] + unknown_args
        
        print(f"  -> Executing: {' '.join(command)}")

        if run_sync_process(command):
            success_count += 1
        else:
            failed_count += 1

    # --- 4. Final Summary ---
    print("\n" + "="*30)
    print("--- Batch Processing Complete ---")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed:                 {failed_count}")
    print(f"  Skipped (already done): {skipped_existing_count}")
    print(f"  Unmatched ref episodes: {len(unmatched_ref)}")
    print(f"  Unmatched for episodes: {len(unmatched_for)}")
    print("="*30)
    
    if failed_count > 0:
        print("\nSome files failed to process. Please review the logs above for details.")
        sys.exit(1)
    else:
        print("\nBatch finished without processing errors.")
        sys.exit(0)


if __name__ == "__main__":
    main()