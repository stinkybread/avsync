import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from av_utils import EnhancedAudioVideoUtils as AudioVideoUtils
from audio_sync import AudioSynchronizer
from visual_sync import EnhancedVisualSynchronizer as VisualSynchronizer


class SyncDriver:
    def __init__(self, window_size=0.1, feature_method='akaze', max_frames=100,
                sample_mode='multi_pass', max_comparisons=800):
        self.window_size = window_size
        self.feature_method = feature_method
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.max_comparisons = max_comparisons
        
        self.audio_sync = AudioSynchronizer(window_size=window_size)
        self.visual_sync = VisualSynchronizer(
            feature_method=feature_method,
            sample_mode=sample_mode,
            max_comparisons=max_comparisons
        )
        self.av_utils = AudioVideoUtils
    
    def extract_audio(self, ref_path, foreign_path, temp_dir, selected_track=None):
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
                       use_scene_detect=True, scene_threshold=0.6, preserve_aspect_ratio=False):
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
    
    def audio_synchronization(self, audio_data, visualize=False, temp_dir=None, content_aware=False):
        print("\n===== Performing Audio Synchronization =====")
        
        viz_path = None
        if visualize and temp_dir:
            if content_aware:
                viz_path = os.path.join(temp_dir, "content_aware_alignment.png")
            else:
                viz_path = os.path.join(temp_dir, "audio_alignment.png")
        
        if content_aware:
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
        else:
            print("Using standard audio synchronization...")
            aligned_audio, audio_sync_info = self.audio_sync.synchronize(
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
        if audio_sync_info['method'] == 'global_offset':
            print(f"Offset: {audio_sync_info['offset_seconds']:.3f} seconds")
            print(f"Confidence: {audio_sync_info['correlation_confidence']:.2f}")
        elif audio_sync_info['method'] == 'content_aware':
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
            ref_segments = len(audio_sync_info.get('content_segments', {}).get('reference', []))
            for_segments = len(audio_sync_info.get('content_segments', {}).get('foreign', []))
            print(f"Detected segments - Reference: {ref_segments}, Foreign: {for_segments}")
        else:
            print(f"Stretched with ratio: {audio_sync_info.get('duration_ratio', 1.0):.4f}")
        
        return aligned_audio, audio_sync_info
    
    def visual_synchronization(self, frame_data, visualize=False, temp_dir=None, 
                              temporal_consistency=False, enhanced_preprocessing=False):
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
    
    def combine_sync_methods(self, audio_sync_info, visual_sync_info):
        print("\n===== Combining Synchronization Methods =====")
        
        combined_info = {'method': 'combined'}
        
        audio_confidence = audio_sync_info.get('correlation_confidence', 0)
        visual_confidence = min(5, visual_sync_info.get('match_count', 0) / 5)
        
        print(f"Audio confidence: {audio_confidence:.2f}")
        print(f"Visual confidence: {visual_confidence:.2f}")
        
        if audio_sync_info.get('method') == 'content_aware':
            print("Content-aware audio sync was used - favoring audio results")
            combined_info['method'] = 'content_aware'
            combined_info['dominant_method'] = 'audio'
            combined_info['offset_seconds'] = audio_sync_info['offset_seconds']
            
            if 'duration_ratio' in audio_sync_info and abs(audio_sync_info['duration_ratio'] - 1.0) > 0.01:
                combined_info['frame_rate_ratio'] = audio_sync_info['duration_ratio']
                print(f"Using content-based duration ratio: {combined_info['frame_rate_ratio']:.4f}")
            else:
                combined_info['frame_rate_ratio'] = visual_sync_info['frame_rate_ratio']
                print(f"Using visual frame rate ratio: {combined_info['frame_rate_ratio']:.4f}")
            
            if 'content_segments' in audio_sync_info:
                combined_info['content_segments'] = audio_sync_info['content_segments']
            
            if 'content_start_points' in audio_sync_info:
                combined_info['content_start_points'] = audio_sync_info['content_start_points']
                ref_start = audio_sync_info['content_start_points']['reference']
                for_start = audio_sync_info['content_start_points']['foreign']
                print(f"First content points - Reference: {ref_start:.3f}s, Foreign: {for_start:.3f}s")
                
            if 'content_end_points' in audio_sync_info:
                combined_info['content_end_points'] = audio_sync_info['content_end_points']
                ref_end = audio_sync_info['content_end_points']['reference']
                for_end = audio_sync_info['content_end_points']['foreign']
                print(f"Last content points - Reference: {ref_end:.3f}s, Foreign: {for_end:.3f}s")
            
            return combined_info
        
        if audio_confidence >= 4.0 and (audio_confidence > visual_confidence * 1.5):
            print("Using audio-dominant synchronization")
            combined_info['dominant_method'] = 'audio'
            
            audio_offset = audio_sync_info['offset_seconds']
            combined_info['offset_seconds'] = audio_offset
            
            frame_rate_ratio = visual_sync_info['frame_rate_ratio']
            
            if (0.9 <= frame_rate_ratio <= 1.1) or abs(frame_rate_ratio - 1.0) > 0.03:
                combined_info['frame_rate_ratio'] = frame_rate_ratio
                print(f"Incorporating frame rate ratio: {frame_rate_ratio:.4f}")
            else:
                combined_info['frame_rate_ratio'] = 1.0
                print("Using default frame rate ratio: 1.0")
            
        elif visual_confidence >= 3.0 and (visual_confidence > audio_confidence * 1.2):
            print("Using visual-dominant synchronization")
            combined_info['dominant_method'] = 'visual'
            
            combined_info['offset_seconds'] = visual_sync_info['global_offset']
            combined_info['frame_rate_ratio'] = visual_sync_info['frame_rate_ratio']
            
        else:
            print("Using weighted combination of audio and visual synchronization")
            combined_info['dominant_method'] = 'combined'
            
            total_confidence = audio_confidence + visual_confidence
            if total_confidence > 0:
                audio_weight = audio_confidence / total_confidence
                visual_weight = visual_confidence / total_confidence
            else:
                audio_weight = visual_weight = 0.5
            
            audio_offset = audio_sync_info.get('offset_seconds', 0)
            visual_offset = visual_sync_info.get('global_offset', 0)
            combined_offset = (audio_offset * audio_weight) + (visual_offset * visual_weight)
            
            combined_info['offset_seconds'] = combined_offset
            combined_info['frame_rate_ratio'] = visual_sync_info['frame_rate_ratio']
            combined_info['audio_weight'] = audio_weight
            combined_info['visual_weight'] = visual_weight
            
            print(f"Audio offset: {audio_offset:.3f}s, Visual offset: {visual_offset:.3f}s")
            print(f"Combined offset: {combined_offset:.3f}s with weights {audio_weight:.2f}/{visual_weight:.2f}")
        
        ref_duration = self.av_utils.get_video_duration(self.ref_path)
        foreign_duration = self.av_utils.get_video_duration(self.foreign_path)
        
        if ref_duration > 0 and foreign_duration > 0:
            duration_ratio = foreign_duration / ref_duration
            print(f"Video durations: Reference = {ref_duration:.2f}s, Foreign = {foreign_duration:.2f}s")
            print(f"Actual duration ratio: {duration_ratio:.4f}")
            
            if abs(duration_ratio - 1.0) > 0.01:
                print(f"Using actual duration ratio: {duration_ratio:.4f}")
                combined_info['frame_rate_ratio'] = duration_ratio
        
        return combined_info
    
    def apply_synchronization(self, audio_data, sync_info, temp_dir=None, ignore_offset=False, content_aware=False):
        """
        Apply synchronization parameters to foreign audio
    
        Parameters:
        audio_data (dict): Audio data
        sync_info (dict): Synchronization info
        temp_dir (str): Directory for temporary files
        ignore_offset (bool): Whether to ignore the offset and use only time stretching
        content_aware (bool): Whether to use content-aware processing even for visual sync
    
        Returns:
        tuple: (synchronized audio, output path)
        """
        print("\n===== Applying Synchronization =====")
    
        # Get synchronization parameters
        offset = 0 if ignore_offset else sync_info.get('offset_seconds', 0)
        frame_rate_ratio = sync_info.get('frame_rate_ratio', 1.0)
    
        # Get foreign audio
        foreign_audio = audio_data['foreign_audio']
        foreign_sr = audio_data['foreign_sr']
    
        # Handle visual method with optional content-aware processing
        if 'method' not in sync_info or sync_info.get('method') == 'visual':
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
        
            # Standard visual synchronization (no content-aware or fallback from content-aware)
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
        
        # For content-aware sync, ensure the reference content start is preserved
        if sync_info.get('method') == 'content_aware' and not ignore_offset:
            print("Using content-aware synchronization approach")
        
            # Get content start and end points
            if 'content_start_points' in sync_info and 'content_end_points' in sync_info:
                ref_start = sync_info['content_start_points']['reference']
                for_start = sync_info['content_start_points']['foreign']
                ref_end = sync_info['content_end_points']['reference']
                for_end = sync_info['content_end_points']['foreign']
            
                print(f"Content spans - Reference: {ref_start:.3f}s to {ref_end:.3f}s")
                print(f"Content spans - Foreign: {for_start:.3f}s to {for_end:.3f}s")
            
                # Calculate content durations
                ref_duration = ref_end - ref_start
                for_duration = for_end - for_start
            
                # Calculate stretch ratio
                content_ratio = for_duration / ref_duration
                print(f"Content duration ratio: {content_ratio:.6f}")
            
                # Trim foreign audio at content start
                trim_samples = int(for_start * foreign_sr)
                if trim_samples > 0 and trim_samples < len(foreign_audio):
                    trimmed_audio = foreign_audio[trim_samples:]
                    print(f"Trimmed {for_start:.3f}s from beginning of foreign audio")
                
                    # Apply time stretching
                    if abs(content_ratio - 1.0) > 0.001:
                        print(f"Applying time stretching with ratio: {content_ratio:.6f}")
                        stretched_audio = self.audio_sync.time_stretch_audio(trimmed_audio, content_ratio, foreign_sr)
                    
                        # Verify result
                        original_duration = len(trimmed_audio) / foreign_sr
                        stretched_duration = len(stretched_audio) / foreign_sr
                        expected_duration = original_duration / content_ratio
                        print(f"Original: {original_duration:.2f}s, Stretched: {stretched_duration:.2f}s, Expected: {expected_duration:.2f}s")
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
                    print(f"Warning: Invalid trim point ({trim_samples} samples), using standard approach")
    
        # Standard approach for regular sync or fallback
        if ignore_offset:
            print("Ignoring offset (using only time stretching)")
            offset_audio = foreign_audio
        else:
            print(f"Applying offset: {offset:.2f}s")
            offset_audio = self.audio_sync.apply_offset(foreign_audio, offset, foreign_sr)
    
        # Apply time stretching if needed
        if abs(frame_rate_ratio - 1.0) > 0.01:
            print(f"Applying time-stretching with ratio: {frame_rate_ratio:.4f}")
        
            if frame_rate_ratio > 1.0:
                stretch_factor = frame_rate_ratio
                operation = "speed up"
            else:
                stretch_factor = 1.0 / frame_rate_ratio
                operation = "slow down"
            
            print(f"This will {operation} the audio to match reference video length")
            print(f"Using librosa stretch factor: {stretch_factor:.4f}")
        
            original_duration = len(offset_audio) / foreign_sr
        
            stretched_audio = self.audio_sync.time_stretch_audio(offset_audio, stretch_factor, foreign_sr)
        
            stretched_duration = len(stretched_audio) / foreign_sr
            print(f"Original duration: {original_duration:.2f}s, Stretched duration: {stretched_duration:.2f}s")
        
            # Calculate the expected duration after stretching
            expected_duration = original_duration / stretch_factor
            print(f"Expected duration after stretching: {expected_duration:.2f}s")
        
            # Check if the result is significantly different from expected
            if abs(stretched_duration - expected_duration) > (expected_duration * 0.1):
                print("WARNING: Time stretching result differs significantly from expected duration!")
        else:
            print("No time stretching needed (ratio near 1.0)")
            stretched_audio = offset_audio
    
        # For content-aware sync, ensure the reference content start is preserved
        ref_content_start = None
        if sync_info.get('method') == 'content_aware' and 'content_start_points' in sync_info:
            # Get the timestamp where content starts in the reference audio
            ref_content_start = sync_info['content_start_points']['reference']
            print(f"Reference content starts at: {ref_content_start:.3f}s")
        
            # For content-aware sync, we want to preserve the reference audio start timing
            # This means we need to add silence at the beginning equal to the reference start time
            if not ignore_offset:
                print(f"Content-aware sync: Preserving reference timing by adding {ref_content_start:.3f}s silence")
    
        # Add silence for content-aware sync if needed
        if ref_content_start is not None and ref_content_start > 0:
            # Add silence at the beginning equal to when the content starts in the reference
            # This preserves the original timing of when audio starts in the reference video
            silence_samples = int(ref_content_start * foreign_sr)
            print(f"Adding {silence_samples} samples ({ref_content_start:.3f}s) of silence to preserve timing")
            final_audio = np.concatenate((np.zeros(silence_samples), stretched_audio))
            print(f"Final audio length: {len(final_audio)/foreign_sr:.2f}s")
        else:
            final_audio = stretched_audio
    
        # Save the final synchronized audio
        final_audio_path = None
        if temp_dir:
            final_audio_path = os.path.join(temp_dir, "final_synchronized.wav")
            self.av_utils.save_audio(final_audio, final_audio_path, foreign_sr)
            print(f"Final synchronized audio saved to: {final_audio_path}")
    
        return final_audio, final_audio_path
        
    def _save_audio(self, audio_data, sr, temp_dir):
        """Helper method to save audio to a temporary file"""
        if temp_dir:
            final_audio_path = os.path.join(temp_dir, "final_synchronized.wav")
            self.av_utils.save_audio(audio_data, final_audio_path, sr)
            print(f"Final synchronized audio saved to: {final_audio_path}")
            return final_audio_path
        return None

    def apply_regular_sync(self, foreign_audio, offset, frame_rate_ratio, foreign_sr):
        """Helper method for regular sync approach"""
        # Apply offset first
        print(f"Applying offset: {offset:.2f}s")
        offset_audio = self.audio_sync.apply_offset(foreign_audio, offset, foreign_sr)
    
        # Apply time stretching if needed
        if abs(frame_rate_ratio - 1.0) > 0.01:
            print(f"Applying time-stretching with ratio: {frame_rate_ratio:.4f}")
        
            if frame_rate_ratio > 1.0:
                stretch_factor = frame_rate_ratio
                operation = "speed up"
            else:
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
    
        return stretched_audio
    
    def visualize_combined_sync(self, audio_sync_info, visual_sync_info, combined_info, save_path):
        plt.figure(figsize=(10, 6))
        
        max_time = self.av_utils.get_video_duration(self.ref_path)
        if max_time <= 0:
            max_time = 600
        
        x_range = np.linspace(0, max_time, 100)
        
        audio_offset = audio_sync_info.get('offset_seconds', 0)
        audio_ratio = audio_sync_info.get('duration_ratio', 1.0)
        
        if audio_sync_info.get('method') == 'content_aware':
            plt.plot(x_range, (x_range * audio_ratio) + audio_offset, 'r-', 
                    label=f'Content-Aware Audio Sync (Offset: {audio_offset:.2f}s, Ratio: {audio_ratio:.4f})')
            
            if 'content_start_points' in audio_sync_info:
                ref_start = audio_sync_info['content_start_points']['reference']
                for_start = audio_sync_info['content_start_points']['foreign'] 
                plt.scatter([ref_start], [for_start], c='m', marker='*', s=100, 
                           label='First Content Match')
                
            if 'content_end_points' in audio_sync_info:
                ref_end = audio_sync_info['content_end_points']['reference']
                for_end = audio_sync_info['content_end_points']['foreign']
                plt.scatter([ref_end], [for_end], c='y', marker='*', s=100,
                           label='Last Content Match')
        else:
            plt.plot(x_range, x_range + audio_offset, 'r-', 
                    label=f'Audio Sync (Offset: {audio_offset:.2f}s)')
        
        visual_offset = visual_sync_info.get('global_offset', 0)
        visual_ratio = visual_sync_info.get('frame_rate_ratio', 1.0)
        plt.plot(x_range, (x_range * visual_ratio) + visual_offset, 
                'g-', label=f'Visual Sync (Offset: {visual_offset:.2f}s, Ratio: {visual_ratio:.4f})')
        
        combined_offset = combined_info.get('offset_seconds', 0)
        combined_ratio = combined_info.get('frame_rate_ratio', 1.0)
        plt.plot(x_range, (x_range * combined_ratio) + combined_offset, 
                'b-', linewidth=2.5, label=f'Combined Sync (Offset: {combined_offset:.2f}s, Ratio: {combined_ratio:.4f})')
        
        plt.plot(x_range, x_range, 'k--', alpha=0.5, label='1:1 Reference')
        
        anchor_points = visual_sync_info.get('anchor_points', [])
        if anchor_points:
            ref_points, for_points = zip(*anchor_points)
            plt.scatter(ref_points, for_points, c='g', marker='o', label='Visual Anchors')
        
        plt.xlabel('Reference Time (s)')
        plt.ylabel('Foreign Time (s)')
        plt.title('Combined Synchronization Map')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(save_path)
        plt.close()
        print(f"Combined synchronization visualization saved to: {save_path}")
    
    def synchronize(self, ref_path, foreign_path, output_path, method='combined', 
                      visualize=False, temp_dir=None, selected_track=None,
                      preserve_original_audio=True, ignore_offset=False, content_aware=False,
                      temporal_consistency=False, enhanced_preprocessing=False,
                      use_scene_detect=True, scene_threshold=0.6, preserve_aspect_ratio=False):
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
        audio_sync_info = None
        visual_sync_info = None
    
        if method in ['audio', 'combined']:
            aligned_audio, audio_sync_info = self.audio_synchronization(audio_data, visualize, temp_dir, content_aware)
            sync_results['audio'] = audio_sync_info
    
        if method in ['visual', 'combined']:
            frame_data = self.extract_frames(
                ref_path, foreign_path, self.max_frames, temp_dir,
                use_scene_detect=use_scene_detect, 
                scene_threshold=scene_threshold,
                preserve_aspect_ratio=preserve_aspect_ratio
            )
        
            visual_sync_info = self.visual_synchronization(
                frame_data, 
                visualize, 
                temp_dir,
                temporal_consistency=temporal_consistency,
                enhanced_preprocessing=enhanced_preprocessing
            )
            sync_results['visual'] = visual_sync_info
    
        if method == 'combined' and audio_sync_info and visual_sync_info:
            combined_info = self.combine_sync_methods(audio_sync_info, visual_sync_info)
            sync_results['combined'] = combined_info
        
            final_audio, final_audio_path = self.apply_synchronization(
                audio_data, combined_info, temp_dir, ignore_offset=ignore_offset, content_aware=content_aware
            )
        
            if visualize:
                viz_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_sync_map.png")
                self.visualize_combined_sync(audio_sync_info, visual_sync_info, combined_info, viz_path)
    
        elif method == 'audio' and audio_sync_info:
            combined_info = audio_sync_info
        
            final_audio, final_audio_path = self.apply_synchronization(
                audio_data, combined_info, temp_dir, ignore_offset=ignore_offset, content_aware=content_aware
            )
        
        elif method == 'visual' and visual_sync_info:
            combined_info = visual_sync_info
            # Add method field for processing
            combined_info['method'] = 'visual'
        
            final_audio, final_audio_path = self.apply_synchronization(
                audio_data, combined_info, temp_dir, ignore_offset=ignore_offset, content_aware=content_aware
            )
        
        else:
            print("\nWarning: No synchronization performed. Using original audio.")
            final_audio_path = audio_data['foreign_audio_path']
            combined_info = {'method': 'none', 'note': 'No synchronization performed'}
    
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