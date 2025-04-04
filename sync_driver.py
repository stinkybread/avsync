"""
Optimized Synchronization Driver Module

This module provides the main synchronization driver focused on visual synchronization
with content-aware audio support.
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from av_utils import EnhancedAudioVideoUtils as AudioVideoUtils
from audio_sync import AudioSynchronizer
from visual_sync import EnhancedVisualSynchronizer as VisualSynchronizer


class SyncDriver:
    def __init__(self, window_size=0.1, feature_method='sift', max_frames=500,
                sample_mode='multi_pass', max_comparisons=5000):
        """
        Initialize the synchronization driver
        
        Parameters:
        window_size (float): Window size for audio sync in seconds
        feature_method (str): Feature extraction method ('sift' recommended)
        max_frames (int): Maximum number of frames to extract
        sample_mode (str): Frame sampling strategy (multi_pass recommended)
        max_comparisons (int): Maximum number of frame comparisons
        """
        self.window_size = window_size
        self.feature_method = feature_method
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.max_comparisons = max_comparisons
        
        self.audio_sync = AudioSynchronizer(window_size=window_size)
        self.visual_sync = VisualSynchronizer(
            feature_method=feature_method,
            sample_mode=sample_mode,
            max_comparisons=max_comparisons,
            temporal_consistency=True,        # Always use temporal consistency
            enhanced_preprocessing=True       # Always use enhanced preprocessing
        )
        self.av_utils = AudioVideoUtils
    
    def extract_audio(self, ref_path, foreign_path, temp_dir, selected_track=None):
        """
        Extract audio from reference and foreign videos
        
        Parameters:
        ref_path (str): Path to reference video
        foreign_path (str): Path to foreign video
        temp_dir (str): Directory for temporary files
        selected_track (int): Specific audio track to select
        
        Returns:
        dict: Audio data and metadata
        """
        print("\n=== Extracting Reference Audio ===")
        ref_audio_path = os.path.join(temp_dir, "reference_audio.wav")
        self.av_utils.extract_audio(ref_path, ref_audio_path)
        
        ref_audio, ref_sr = self.av_utils.load_audio(ref_audio_path)
        
        print("\n=== Extracting Foreign Audio ===")
        if selected_track is None:
            selected_track = self.av_utils.select_audio_track(foreign_path)
            
        foreign_audio_path = os.path.join(temp_dir, "foreign_audio.wav")
        self.av_utils.extract_audio(foreign_path, foreign_audio_path, track_index=selected_track)
        
        foreign_audio, foreign_sr = self.av_utils.load_audio(foreign_audio_path)
        
        audio_tracks = self.av_utils.get_audio_tracks(foreign_path)
        track_info = audio_tracks[selected_track] if selected_track < len(audio_tracks) else None
        
        return {
            'ref_audio': ref_audio,
            'ref_sr': ref_sr,
            'ref_audio_path': ref_audio_path,
            'foreign_audio': foreign_audio,
            'foreign_sr': foreign_sr,
            'foreign_audio_path': foreign_audio_path,
            'track_info': track_info,
            'selected_track': selected_track
        }
    
    def extract_frames(self, ref_path, foreign_path, max_frames, temp_dir=None, fps=1.0, 
                       use_scene_detect=True, scene_threshold=0.25, preserve_aspect_ratio=False):
        """
        Extract frames from reference and foreign videos
        
        Parameters:
        ref_path (str): Path to reference video
        foreign_path (str): Path to foreign video
        max_frames (int): Maximum number of frames to extract
        temp_dir (str): Directory for temporary files
        fps (float): Frames per second for uniform extraction
        use_scene_detect (bool): Whether to use scene detection
        scene_threshold (float): Threshold for scene detection
        preserve_aspect_ratio (bool): Whether to preserve aspect ratio
        
        Returns:
        dict: Frame data including timestamps
        """
        ref_frames_dir = None
        foreign_frames_dir = None
        
        if temp_dir:
            ref_frames_dir = os.path.join(temp_dir, "ref_frames")
            foreign_frames_dir = os.path.join(temp_dir, "foreign_frames")
            os.makedirs(ref_frames_dir, exist_ok=True)
            os.makedirs(foreign_frames_dir, exist_ok=True)
        
        print("\n=== Extracting Reference Video Frames ===")
        ref_frames, ref_timestamps = self.av_utils.extract_frames(
            ref_path, 
            output_dir=ref_frames_dir, 
            max_frames=max_frames, 
            fps=fps,
            use_scene_detect=use_scene_detect,
            scene_threshold=scene_threshold,
            preserve_aspect_ratio=preserve_aspect_ratio
        )
        
        print("\n=== Extracting Foreign Video Frames ===")
        ref_width = next(iter(ref_frames.values()))['original'].shape[1] if ref_frames else 640
        ref_height = next(iter(ref_frames.values()))['original'].shape[0] if ref_frames else 480
        ref_aspect_ratio = ref_width / ref_height if ref_height > 0 else 4/3
        
        foreign_frames, foreign_timestamps = self.av_utils.extract_frames(
            foreign_path, 
            output_dir=foreign_frames_dir, 
            max_frames=max_frames, 
            fps=fps,
            reference_aspect_ratio=ref_aspect_ratio,
            use_scene_detect=use_scene_detect,
            scene_threshold=scene_threshold,
            preserve_aspect_ratio=preserve_aspect_ratio
        )
        
        return {
            'ref_frames': ref_frames,
            'ref_timestamps': ref_timestamps,
            'foreign_frames': foreign_frames,
            'foreign_timestamps': foreign_timestamps
        }
    
    def audio_synchronization(self, audio_data, visualize=False, temp_dir=None, content_aware=True):
        """
        Perform content-aware audio synchronization
        
        Parameters:
        audio_data (dict): Audio data from extract_audio
        visualize (bool): Whether to create visualizations
        temp_dir (str): Directory for temporary files
        content_aware (bool): Whether to use content-aware synchronization
        
        Returns:
        tuple: (aligned_audio, audio_sync_info)
        """
        print("\n===== Performing Audio Synchronization =====")
        
        viz_path = None
        if visualize and temp_dir:
            viz_path = os.path.join(temp_dir, "content_aware_alignment.png")
        
        print("Using content-aware audio synchronization...")
        aligned_audio, audio_sync_info = self.audio_sync.content_aware_synchronize(
            audio_data['ref_audio'],
            audio_data['foreign_audio'],
            audio_data['ref_sr'],
            audio_data['foreign_sr'],
            method='auto',
            visualize=visualize,
            viz_path=viz_path
        )
        
        if temp_dir:
            audio_result_path = os.path.join(temp_dir, "audio_synchronized.wav")
            self.av_utils.save_audio(aligned_audio, audio_result_path, audio_data['foreign_sr'])
            audio_sync_info["output_path"] = audio_result_path
        
        print(f"\nAudio synchronization method: {audio_sync_info['method']}")
        if audio_sync_info['method'] == 'content_aware':
            print(f"Content-aware offset: {audio_sync_info['offset_seconds']:.3f} seconds")
            if 'content_start_points' in audio_sync_info:
                ref_start = audio_sync_info['content_start_points']['reference'] 
                for_start = audio_sync_info['content_start_points']['foreign']
                print(f"First content - Reference: {ref_start:.3f}s, Foreign: {for_start:.3f}s")
            if 'content_end_points' in audio_sync_info:
                ref_end = audio_sync_info['content_end_points']['reference']
                for_end = audio_sync_info['content_end_points']['foreign']
                print(f"Last content - Reference: {ref_end:.3f}s, Foreign: {for_end:.3f}s")
            print(f"Duration ratio: {audio_sync_info.get('duration_ratio', 1.0):.4f}")
        
        return aligned_audio, audio_sync_info
    
    def visual_synchronization(self, frame_data, visualize=False, temp_dir=None, 
                              temporal_consistency=True, enhanced_preprocessing=True):
        """
        Perform visual synchronization
        
        Parameters:
        frame_data (dict): Frame data from extract_frames
        visualize (bool): Whether to create visualizations
        temp_dir (str): Directory for temporary files
        temporal_consistency (bool): Whether to use temporal consistency checking
        enhanced_preprocessing (bool): Whether to use enhanced image preprocessing
        
        Returns:
        dict: Visual synchronization information
        """
        print("\n===== Performing Visual Synchronization =====")
        
        viz_dir = None
        if visualize and temp_dir:
            viz_dir = os.path.join(temp_dir, "visual_sync")
            os.makedirs(viz_dir, exist_ok=True)
        
        self.visual_sync.temporal_consistency = temporal_consistency
        self.visual_sync.enhanced_preprocessing = enhanced_preprocessing
        
        visual_sync_info = self.visual_sync.synchronize(
            frame_data['ref_frames'],
            frame_data['foreign_frames'],
            visualize=visualize,
            output_dir=viz_dir
        )
        
        print(f"\nVisual synchronization results:")
        print(f"Matched frames: {visual_sync_info['match_count']}")
        print(f"Global offset: {visual_sync_info['global_offset']:.3f} seconds")
        print(f"Frame rate ratio: {visual_sync_info['frame_rate_ratio']:.4f}")
        
        if 'anchor_points' in visual_sync_info and visual_sync_info['anchor_points']:
            print(f"Number of anchor points: {len(visual_sync_info['anchor_points'])}")
        
        return visual_sync_info
    
    def apply_synchronization(self, audio_data, sync_info, temp_dir=None, content_aware=True):
        """
        Apply visual synchronization parameters to foreign audio with content-aware support
        
        Parameters:
        audio_data (dict): Audio data
        sync_info (dict): Synchronization info
        temp_dir (str): Directory for temporary files
        content_aware (bool): Whether to use content-aware processing
        
        Returns:
        tuple: (synchronized audio, output path)
        """
        print("\n===== Applying Synchronization =====")
    
        # Get synchronization parameters
        offset = sync_info.get('global_offset', 0)
        frame_rate_ratio = sync_info.get('frame_rate_ratio', 1.0)
    
        # Get foreign audio
        foreign_audio = audio_data['foreign_audio']
        foreign_sr = audio_data['foreign_sr']
    
        # Always use content-aware processing (as per your command)
        if content_aware:
            print(f"Applying visual synchronization with content-aware processing")
        
            # Detect content in both audio tracks
            print("Detecting audio content in reference track...")
            ref_segments = self.audio_sync.detect_audio_content(audio_data['ref_audio'], audio_data['ref_sr'])
        
            print("Detecting audio content in foreign track...")
            foreign_segments = self.audio_sync.detect_audio_content(foreign_audio, foreign_sr)
        
            if ref_segments and foreign_segments:
                # Use the first segment for alignment
                ref_start = ref_segments[0][0]
                for_start = foreign_segments[0][0]
            
                print(f"First content - Reference: {ref_start:.3f}s, Foreign: {for_start:.3f}s")
            
                # Trim foreign audio at content start
                trim_samples = int(for_start * foreign_sr)
                if trim_samples > 0 and trim_samples < len(foreign_audio):
                    trimmed_audio = foreign_audio[trim_samples:]
                    print(f"Trimmed {for_start:.3f}s from beginning of foreign audio")
                
                    # Apply time stretching using the visual frame rate ratio
                    if abs(frame_rate_ratio - 1.0) > 0.01:
                        print(f"Applying time stretching with ratio: {frame_rate_ratio:.6f}")
                        stretched_audio = self.audio_sync.time_stretch_audio(trimmed_audio, frame_rate_ratio, foreign_sr)
                    else:
                        stretched_audio = trimmed_audio
                
                    # Add silence equal to reference content start
                    silence_samples = int(ref_start * foreign_sr)
                    print(f"Adding {ref_start:.3f}s of silence to preserve reference timing")
                    final_audio = np.concatenate((np.zeros(silence_samples), stretched_audio))
                
                    # Save the final audio
                    final_audio_path = None
                    if temp_dir:
                        final_audio_path = os.path.join(temp_dir, "final_synchronized.wav")
                        self.av_utils.save_audio(final_audio, final_audio_path, foreign_sr)
                        print(f"Final synchronized audio saved to: {final_audio_path}")
                
                    return final_audio, final_audio_path
                else:
                    print(f"Warning: Invalid trim point ({trim_samples} samples), using standard visual approach")
            else:
                print("Could not detect enough audio content, using standard visual approach")
    
        # Standard visual synchronization approach (fallback)
        print(f"Applying visual synchronization - offset: {offset:.3f}s, ratio: {frame_rate_ratio:.4f}")
    
        # Apply offset first
        print(f"Applying offset: {offset:.2f}s")
        offset_audio = self.audio_sync.apply_offset(foreign_audio, offset, foreign_sr)
    
        # Apply time stretching
        if abs(frame_rate_ratio - 1.0) > 0.01:
            print(f"Applying time-stretching with ratio: {frame_rate_ratio:.4f}")
        
            # Set stretch factor based on frame rate ratio
            if frame_rate_ratio > 1.0:
                stretch_factor = frame_rate_ratio
                operation = "speed up"
            else:
                # Invert ratio for slowing down
                stretch_factor = 1.0 / frame_rate_ratio
                operation = "slow down"
            
            print(f"This will {operation} the audio to match reference video length")
        
            original_duration = len(offset_audio) / foreign_sr
            stretched_audio = self.audio_sync.time_stretch_audio(offset_audio, stretch_factor, foreign_sr)
            stretched_duration = len(stretched_audio) / foreign_sr
        
            print(f"Original duration: {original_duration:.2f}s, Stretched duration: {stretched_duration:.2f}s")
        else:
            print("No time stretching needed (ratio near 1.0)")
            stretched_audio = offset_audio
    
        # Save the final audio
        final_audio_path = None
        if temp_dir:
            final_audio_path = os.path.join(temp_dir, "final_synchronized.wav")
            self.av_utils.save_audio(stretched_audio, final_audio_path, foreign_sr)
            print(f"Final synchronized audio saved to: {final_audio_path}")
    
        return stretched_audio, final_audio_path
    
    def visualize_sync(self, visual_sync_info, save_path):
        """
        Visualize the synchronization mapping
        
        Parameters:
        visual_sync_info (dict): Visual synchronization information
        save_path (str): Path to save visualization
        """
        if 'anchor_points' not in visual_sync_info or not visual_sync_info['anchor_points']:
            print("No anchor points available for visualization")
            return
        
        # Get parameters
        anchor_points = visual_sync_info['anchor_points']
        frame_rate_ratio = visual_sync_info['frame_rate_ratio']
        global_offset = visual_sync_info['global_offset']
        
        plt.figure(figsize=(10, 6))
        
        # Extract timestamps
        ref_times, for_times = zip(*anchor_points)
        
        # Create a range of reference times
        min_ref = min(ref_times)
        max_ref = max(ref_times)
        ref_range = np.linspace(min_ref, max_ref, 100)
        
        # Calculate corresponding foreign times using the frame rate ratio and offset
        for_range = (ref_range * frame_rate_ratio) + global_offset
        
        # Plot the matched timestamps
        plt.scatter(ref_times, for_times, c='r', marker='o', label='Matched Frames')
        
        # Plot the calculated mapping line
        plt.plot(ref_range, for_range, 'b-', linewidth=2, 
                label=f'Mapping (Ratio: {frame_rate_ratio:.4f}, Offset: {global_offset:.2f}s)')
        
        # Plot 1:1 reference line
        plt.plot([min_ref, max_ref], [min_ref, max_ref], 'g--', label='1:1 Reference')
        
        # Add labels and title
        plt.xlabel('Reference Video Time (s)')
        plt.ylabel('Foreign Video Time (s)')
        plt.title('Visual Synchronization Mapping')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save
        plt.savefig(save_path)
        plt.close()
        
        print(f"Synchronization visualization saved to: {save_path}")
    
    def synchronize(self, ref_path, foreign_path, output_path, 
                      visualize=False, temp_dir=None, selected_track=None,
                      preserve_original_audio=True, content_aware=True,
                      temporal_consistency=True, enhanced_preprocessing=True,
                      use_scene_detect=True, scene_threshold=0.25, preserve_aspect_ratio=False):
        """
        Perform visual synchronization with content-aware audio support
        
        Parameters:
        ref_path (str): Path to reference video
        foreign_path (str): Path to foreign video
        output_path (str): Path to output video
        visualize (bool): Whether to create visualizations
        temp_dir (str): Directory for temporary files
        selected_track (int): Specific audio track to use
        preserve_original_audio (bool): Whether to keep original audio tracks
        content_aware (bool): Whether to use content-aware processing
        temporal_consistency (bool): Whether to use temporal consistency checking
        enhanced_preprocessing (bool): Whether to use enhanced image preprocessing
        use_scene_detect (bool): Whether to use scene detection
        scene_threshold (float): Threshold for scene detection
        preserve_aspect_ratio (bool): Whether to preserve aspect ratio
        
        Returns:
        dict: Synchronization results
        """
        self.ref_path = ref_path
        self.foreign_path = foreign_path
    
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            print(f"Created temporary directory: {temp_dir}")
        else:
            os.makedirs(temp_dir, exist_ok=True)
    
        print("\n===== Extracting Audio =====")
        audio_data = self.extract_audio(ref_path, foreign_path, temp_dir, selected_track)
    
        sync_results = {}
    
        # Extract frames using scene detection and specified parameters
        frame_data = self.extract_frames(
            ref_path, foreign_path, self.max_frames, temp_dir,
            use_scene_detect=use_scene_detect, 
            scene_threshold=scene_threshold,
            preserve_aspect_ratio=preserve_aspect_ratio
        )
        
        # Perform visual synchronization
        visual_sync_info = self.visual_synchronization(
            frame_data, 
            visualize, 
            temp_dir,
            temporal_consistency=temporal_consistency,
            enhanced_preprocessing=enhanced_preprocessing
        )
        sync_results['visual'] = visual_sync_info
        
        # Apply synchronization parameters to audio using content-aware approach
        final_audio, final_audio_path = self.apply_synchronization(
            audio_data, visual_sync_info, temp_dir, content_aware=content_aware
        )
        
        if visualize:
            viz_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_sync_map.png")
            self.visualize_sync(visual_sync_info, viz_path)
    
        track_info = audio_data.get('track_info')
        audio_language = track_info.get('language', 'unknown') if track_info else 'unknown'
        audio_title = track_info.get('title', 'Foreign Audio') if track_info else 'Foreign Audio'
    
        print("\n===== Creating Final Video =====")
        mux_result = self.av_utils.mux_audio_with_video(
            ref_path, 
            final_audio_path, 
            output_path,
            audio_language=audio_language,
            audio_title=audio_title,
            preserve_original_audio=preserve_original_audio
        )
    
        if mux_result:
            print(f"\nSuccessfully created synchronized video: {output_path}")
        else:
            print(f"\nFailed to create synchronized video!")
    
        return sync_results