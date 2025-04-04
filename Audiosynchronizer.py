#!/usr/bin/env python3
"""
Video Synchronization CLI Tool
"""

import os
import argparse
import tempfile
import time
import shutil
from sync_driver import SyncDriver


def main():
    """
    Main entry point for the video synchronization tool
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Synchronize foreign audio with reference videos')
    
    parser.add_argument('reference', help='Path to reference video (high quality)')
    parser.add_argument('foreign', help='Path to foreign video (with foreign audio)')
    parser.add_argument('output', help='Path to output synchronized video')
    
    # Synchronization method
    parser.add_argument('--method', choices=['audio', 'visual', 'combined'], default='combined', 
                      help='Synchronization method (default: combined)')
    
    # Frame extraction options
    parser.add_argument('--max-frames', type=int, default=100, 
                      help='Maximum number of frames to extract for visual sync (default: 100)')
    parser.add_argument('--fps', type=float, default=1.0, 
                      help='Frames per second to extract for visual sync (default: 1.0)')
    parser.add_argument('--use-scene-detect', action='store_true',
                      help='Use scene change detection for frame extraction')
    parser.add_argument('--scene-threshold', type=float, default=0.6,
                      help='Threshold for scene change detection (default: 0.6)')
    parser.add_argument('--preserve-aspect-ratio', action='store_true',
                      help='Preserve aspect ratio when resizing frames (adds letterboxing/pillarboxing)')
    
    # Audio options
    parser.add_argument('--window-size', type=float, default=0.1, 
                      help='Window size in seconds for audio analysis (default: 0.1)')
    parser.add_argument('--ignore-offset', action='store_true',
                      help='Ignore audio offset and only apply time stretching')
    parser.add_argument('--content-aware', action='store_true',
                      help='Use content-aware audio synchronization (detect actual audio content)')
    
    # Content detection parameters
    parser.add_argument('--db-threshold', type=float, default=-40,
                      help='Audio detection threshold in dB (default: -40)')
    parser.add_argument('--min-duration', type=float, default=0.2,
                      help='Minimum duration in seconds for audio segments (default: 0.2s)')
    parser.add_argument('--min-silence', type=int, default=500,
                      help='Minimum silence duration in ms (default: 500)')
    
    # Visual sync options
    parser.add_argument('--feature', choices=['orb', 'sift', 'akaze'], default='akaze', 
                      help='Visual feature extraction method (default: akaze)')
    parser.add_argument('--sample-mode', choices=['uniform', 'sparse', 'adaptive', 'multi_pass'], default='multi_pass',
                      help='Frame sampling strategy for visual sync (default: multi_pass)')
    parser.add_argument('--max-comparisons', type=int, default=800,
                      help='Maximum number of frame comparisons for visual sync (default: 800)')
    parser.add_argument('--temporal-consistency', action='store_true',
                      help='Use temporal consistency checking for visual sync')
    parser.add_argument('--enhanced-preprocessing', action='store_true',
                      help='Use enhanced image preprocessing for better feature matching')
    
    # Audio track selection
    parser.add_argument('--force-track', type=int, 
                      help='Force using specific audio track number')
    
    # Output options
    parser.add_argument('--visualize', action='store_true', 
                      help='Create visualizations of the alignment process')
    parser.add_argument('--temp-dir', 
                      help='Directory for temporary files (overrides default location)')
    parser.add_argument('--keep-temp', action='store_true',
                      help='Keep temporary files after completion (default: delete)')
    parser.add_argument('--no-preserve-audio', action='store_true',
                      help='Do not preserve original audio tracks')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for path in [args.reference, args.foreign]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup temporary directory in the same location as the output file
    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        output_filename = os.path.splitext(os.path.basename(args.output))[0]
        temp_dir = os.path.join(output_dir, f"temp_{output_filename}")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    # Determine if we should cleanup temp files
    cleanup_temp = not args.keep_temp
    
    try:
        print("\n===== Video Synchronization Tool =====")
        print(f"Reference Video: {os.path.basename(args.reference)}")
        print(f"Foreign Video: {os.path.basename(args.foreign)}")
        print(f"Synchronization Method: {args.method}")
        print(f"Visual Feature Method: {args.feature}")
        print(f"Frame Sampling Mode: {args.sample_mode}")
        print(f"Temporary Directory: {temp_dir}")
        
        if args.content_aware:
            print("Using content-aware audio synchronization")
            print(f"Audio detection threshold: {args.db_threshold} dB")
            print(f"Minimum segment duration: {args.min_duration}s")
            print(f"Minimum silence duration: {args.min_silence}ms")
            
        if args.ignore_offset:
            print("Ignoring audio offset (using only time stretching)")
            
        if args.temporal_consistency:
            print("Using temporal consistency checking for visual sync")
            
        if args.enhanced_preprocessing:
            print("Using enhanced image preprocessing")
            
        if args.use_scene_detect:
            print(f"Using scene detection for frame extraction (threshold: {args.scene_threshold})")
            
        if args.preserve_aspect_ratio:
            print("Preserving aspect ratio with letterboxing/pillarboxing")
        
        # Create synchronizer
        start_time = time.time()
        sync = SyncDriver(
            window_size=args.window_size, 
            feature_method=args.feature,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            max_comparisons=args.max_comparisons
        )
        
        # Customize the audio_sync parameters to use dB-based detection
        sync.audio_sync.detect_audio_content = lambda audio_data, sr, **kwargs: \
            sync.audio_sync.__class__.detect_audio_content(
                sync.audio_sync, audio_data, sr, 
                threshold_db=args.db_threshold, 
                min_duration=args.min_duration,
                min_silence_ms=args.min_silence
            )
        
        # Perform synchronization
        sync_results = sync.synchronize(
            args.reference, 
            args.foreign, 
            args.output,
            method=args.method,
            visualize=args.visualize,
            temp_dir=temp_dir,
            selected_track=args.force_track,
            preserve_original_audio=not args.no_preserve_audio,
            ignore_offset=args.ignore_offset,
            content_aware=args.content_aware,
            temporal_consistency=args.temporal_consistency,
            enhanced_preprocessing=args.enhanced_preprocessing,
            use_scene_detect=args.use_scene_detect,
            scene_threshold=args.scene_threshold,
            preserve_aspect_ratio=args.preserve_aspect_ratio
        )
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n===== Synchronization Summary =====")
        print(f"Total processing time: {elapsed_time:.1f} seconds")
        
        if args.method == 'combined' and 'combined' in sync_results:
            print("\nCombined Synchronization:")
            combined_info = sync_results['combined']
            print(f"Dominant method: {combined_info.get('dominant_method', 'none')}")
            
            if 'offset_seconds' in combined_info and not args.ignore_offset:
                print(f"Applied offset: {combined_info['offset_seconds']:.2f} seconds")
                
            if 'frame_rate_ratio' in combined_info and abs(combined_info['frame_rate_ratio'] - 1.0) > 0.01:
                print(f"Applied frame rate ratio: {combined_info['frame_rate_ratio']:.4f}")
                
            # For content-aware synchronization, print additional info
            if combined_info.get('method') == 'content_aware' and 'content_start_points' in combined_info:
                ref_start = combined_info['content_start_points']['reference']
                for_start = combined_info['content_start_points']['foreign']
                print(f"First content detected at - Reference: {ref_start:.2f}s, Foreign: {for_start:.2f}s")
        
        # Print visual sync info if available
        if 'visual' in sync_results:
            visual_info = sync_results['visual']
            if 'match_count' in visual_info:
                print(f"\nVisual Synchronization:")
                print(f"Matched frames: {visual_info['match_count']}")
                if 'anchor_points' in visual_info and visual_info['anchor_points']:
                    print(f"Number of anchor points: {len(visual_info['anchor_points'])}")
        
        if args.visualize:
            print("\nVisualizations saved in:")
            print(f"- {os.path.dirname(args.output)}")
            if os.path.exists(os.path.join(temp_dir, "content_aware_alignment.png")):
                print(f"- Content-aware alignment: {os.path.join(temp_dir, 'content_aware_alignment.png')}")
            if os.path.exists(os.path.join(temp_dir, "visual_sync")):
                print(f"- Visual match visualizations: {os.path.join(temp_dir, 'visual_sync')}")
        
        print(f"\nOutput saved to: {args.output}")
        print("Synchronization complete!")
        
        return 0
        
    except Exception as e:
        print(f"\nError during synchronization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary directory if needed
        if cleanup_temp and os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files from: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary files deleted successfully")
            except Exception as e:
                print(f"Warning: Could not delete temporary files: {e}")
        else:
            print(f"\nTemporary files kept at: {temp_dir}")


if __name__ == "__main__":
    exit(main())
