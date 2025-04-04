"""
Audio Synchronization Module
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal


class AudioSynchronizer:
    """Core class for synchronizing audio tracks"""
    
    def __init__(self, window_size=0.1):
        """
        Initialize the audio synchronizer
        """
        self.window_size = window_size
    
    def create_energy_fingerprint(self, audio_data, sr):
        """
        Create a simple fingerprint based on audio energy
        """
        # Calculate number of samples per window
        samples_per_window = int(self.window_size * sr)
        
        # Calculate number of windows
        n_windows = len(audio_data) // samples_per_window
        
        # Reshape audio data into windows
        windowed_audio = audio_data[:n_windows * samples_per_window].reshape(-1, samples_per_window)
        
        # Calculate RMS energy for each window
        energy = np.sqrt(np.mean(windowed_audio**2, axis=1))
        
        return energy
    
    def create_onset_fingerprint(self, audio_data, sr):
        """
        Create a fingerprint based on onset strength
        """
        # Compute onset strength envelope with higher sensitivity
        hop_length = int(self.window_size * sr)
        
        try:
            onset_env = librosa.onset.onset_strength(
                y=audio_data, sr=sr, hop_length=hop_length,
                fmax=8000  # Increase frequency range for better detection
            )
            
            # Normalize the onset envelope
            if np.max(onset_env) > 0:
                onset_env = onset_env / np.max(onset_env)
                
            return onset_env
            
        except Exception as e:
            print(f"Warning: Error creating onset fingerprint: {e}")
            print("Falling back to simple energy fingerprint")
            return self.create_energy_fingerprint(audio_data, sr)
    
    def find_global_offset(self, ref_fingerprint, foreign_fingerprint, window_size_seconds):
        """
        Find global time offset between two audio tracks using cross-correlation
        """
        # Normalize the fingerprints
        ref_norm = (ref_fingerprint - np.mean(ref_fingerprint)) / (np.std(ref_fingerprint) + 1e-10)
        foreign_norm = (foreign_fingerprint - np.mean(foreign_fingerprint)) / (np.std(foreign_fingerprint) + 1e-10)
        
        # Compute cross-correlation
        correlation = signal.correlate(ref_norm, foreign_norm, mode='full')
        
        # Find the best offset
        best_offset = np.argmax(correlation) - (len(foreign_norm) - 1)
        
        # Calculate correlation strength as a confidence measure
        max_corr = np.max(correlation)
        corr_mean = np.mean(correlation)
        corr_std = np.std(correlation)
        confidence_value = (max_corr - corr_mean) / (corr_std + 1e-10)
        
        # Convert from windows to seconds
        time_offset = best_offset * window_size_seconds
        
        return time_offset, correlation, confidence_value
    
    def apply_offset(self, audio_data, offset_seconds, sr):
        """
        Apply a global time offset to audio data
        """
        offset_samples = int(offset_seconds * sr)
        
        # Before applying, check if the offset is reasonable
        if abs(offset_samples) > len(audio_data) * 0.8:
            print(f"Warning: Very large offset ({offset_seconds:.2f}s, {offset_samples} samples) compared to audio length")
            # Cap the offset to avoid completely empty output
            max_offset = int(len(audio_data) * 0.8)
            if offset_samples > max_offset:
                offset_samples = max_offset
                print(f"Capped positive offset to {offset_samples / sr:.2f}s")
            elif offset_samples < -max_offset:
                offset_samples = -max_offset
                print(f"Capped negative offset to {offset_samples / sr:.2f}s")
        
        if offset_samples > 0:
            # Delay the audio by adding zeros at the beginning
            return np.concatenate((np.zeros(offset_samples), audio_data))
        elif offset_samples < 0:
            # Advance the audio by removing samples from the beginning
            if -offset_samples >= len(audio_data):
                # Avoid returning empty array
                print("Warning: Offset would remove entire audio, using minimal offset")
                return audio_data[-1000:]
            return audio_data[-offset_samples:]
        else:
            # No change needed
            return audio_data
        
    def time_stretch_audio(self, audio_data, content_ratio, sr):
        """
        Stretch audio to match reference content duration
        """
        if abs(content_ratio - 1.0) < 0.001:
            print("No time-stretching needed (ratio very close to 1.0)")
            return audio_data
    
        # Calculate correct atempo value
        if content_ratio > 1.0:
            # Foreign is longer, need to speed up (shorten it)
            atempo = content_ratio  # Use ratio directly
            operation = "speed up"
        else:
            # Foreign is shorter, need to slow down (lengthen it)
            atempo = content_ratio  # Use ratio directly
            operation = "slow down"
    
        print(f"Content ratio: {content_ratio:.6f}. Will {operation} the audio using atempo={atempo:.6f}")
    
        try:
            print(f"Stretching audio using FFmpeg with atempo={atempo:.6f}")
        
            # Save input to temp file
            import tempfile
            import soundfile as sf
            import subprocess
            import os
        
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                temp_in_path = temp_in.name
        
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
                temp_out_path = temp_out.name
        
            # Write input audio
            sf.write(temp_in_path, audio_data, sr)
        
            # Build atempo chain for values outside 0.5-2.0 range
            atempo_cmd = []
            remaining_factor = atempo
        
            while remaining_factor > 2.0:
                atempo_cmd.append("atempo=2.0")
                remaining_factor /= 2.0
        
            while remaining_factor < 0.5:
                atempo_cmd.append("atempo=0.5")
                remaining_factor /= 0.5
        
            # Add final stretch value
            atempo_cmd.append(f"atempo={remaining_factor:.6f}")
        
            print(f"FFmpeg atempo chain: {','.join(atempo_cmd)}")
        
            # Run FFmpeg
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_in_path,
                '-filter:a', ','.join(atempo_cmd),
                '-y', temp_out_path
            ]
        
            print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        
            # Read back the processed file
            stretched_audio, _ = librosa.load(temp_out_path, sr=sr)
        
            # Clean up
            os.unlink(temp_in_path)
            os.unlink(temp_out_path)
        
            # Verify result
            original_duration = len(audio_data) / sr
            stretched_duration = len(stretched_audio) / sr
            expected_duration = original_duration / content_ratio
        
            print(f"Original: {original_duration:.2f}s, Stretched: {stretched_duration:.2f}s, Expected: {expected_duration:.2f}s")
        
            return stretched_audio
        
        except Exception as e:
            print(f"FFmpeg stretching failed: {e}")
            print("Falling back to librosa stretching")
        
            try:
                stretched_audio = librosa.effects.time_stretch(audio_data, rate=content_ratio)
            
                # Verify result
                original_duration = len(audio_data) / sr
                stretched_duration = len(stretched_audio) / sr
                expected_duration = original_duration / content_ratio
            
                print(f"Original: {original_duration:.2f}s, Stretched: {stretched_duration:.2f}s, Expected: {expected_duration:.2f}s")
            
                return stretched_audio
            
            except Exception as e2:
                print(f"Librosa stretching also failed: {e2}")
                print("Returning original audio")
                return audio_data
    
    def detect_audio_content(self, audio_data, sr, threshold_db=-40, min_duration=0.2, min_silence_ms=500):
        """
        Detect segments of actual audio content (non-silence) using decibel thresholding
        """
        # Calculate chunk size (in samples) for 10ms chunks
        chunk_size = int(0.01 * sr)  # 10ms chunks
        hop_length = chunk_size  # No overlap
        
        # Calculate number of chunks
        n_chunks = (len(audio_data) - chunk_size) // hop_length + 1
        
        # Calculate dB for each chunk
        chunk_dbs = []
        for i in range(n_chunks):
            start = i * hop_length
            end = start + chunk_size
            chunk = audio_data[start:end]
            
            # Calculate dB (avoid log of zero/negative by adding small value)
            if np.sum(chunk**2) > 0:
                # Convert to dB scale: 20 * log10(RMS)
                rms = np.sqrt(np.mean(chunk**2))
                db = 20 * np.log10(max(rms, 1e-10))
            else:
                db = -100  # Very low dB for silence
            
            chunk_dbs.append(db)
        
        # Find segments above threshold
        segments = []
        in_segment = False
        segment_start = 0
        min_silence_chunks = min_silence_ms // (hop_length / sr * 1000)
        silence_count = 0
        
        for i, db in enumerate(chunk_dbs):
            if not in_segment and db > threshold_db:
                # Start of segment
                segment_start = i
                in_segment = True
                silence_count = 0
            elif in_segment:
                if db <= threshold_db:
                    silence_count += 1
                    if silence_count >= min_silence_chunks:
                        # End of segment
                        segment_end = i - silence_count
                        duration = (segment_end - segment_start) * hop_length / sr
                        
                        if duration >= min_duration:
                            start_sec = segment_start * hop_length / sr
                            end_sec = segment_end * hop_length / sr
                            segments.append((start_sec, end_sec))
                        
                        in_segment = False
                else:
                    silence_count = 0
        
        # Handle case where audio ends during a segment
        if in_segment:
            segment_end = n_chunks - 1
            duration = (segment_end - segment_start) * hop_length / sr
            
            if duration >= min_duration:
                start_sec = segment_start * hop_length / sr
                end_sec = segment_end * hop_length / sr
                segments.append((start_sec, end_sec))
        
        # Merge segments that are very close (< 300ms gap)
        if segments:
            merged_segments = [segments[0]]
            
            for seg in segments[1:]:
                prev_end = merged_segments[-1][1]
                cur_start = seg[0]
                
                if cur_start - prev_end < 0.3:  # 300ms gap
                    # Merge with previous segment
                    merged_segments[-1] = (merged_segments[-1][0], seg[1])
                else:
                    merged_segments.append(seg)
            
            segments = merged_segments
        
        # Detailed information about segments
        if segments:
            print(f"Detected {len(segments)} audio segments using dB thresholding (threshold: {threshold_db} dB):")
            print(f"First segment: {segments[0][0]:.3f}s - {segments[0][1]:.3f}s (duration: {segments[0][1]-segments[0][0]:.3f}s)")
            print(f"Last segment: {segments[-1][0]:.3f}s - {segments[-1][1]:.3f}s (duration: {segments[-1][1]-segments[-1][0]:.3f}s)")
            print(f"Total content duration: {segments[-1][1] - segments[0][0]:.3f}s")
        else:
            print(f"No audio segments detected with threshold {threshold_db} dB - try adjusting threshold")
        
        return segments
    
    def content_aware_synchronize(self, ref_audio, foreign_audio, ref_sr, foreign_sr, 
                                method='auto', visualize=False, viz_path=None):
        """
        Synchronize foreign audio to reference audio using multi-point content-aware anchoring
        """
        # Detect content segments in both audios
        print("Detecting audio content in reference track...")
        ref_segments = self.detect_audio_content(ref_audio, ref_sr)
        print(f"Found {len(ref_segments)} content segments in reference audio")
        
        print("Detecting audio content in foreign track...")
        foreign_segments = self.detect_audio_content(foreign_audio, foreign_sr)
        print(f"Found {len(foreign_segments)} content segments in foreign audio")
        
        # If no segments found, fall back to regular synchronization
        if not ref_segments or not foreign_segments:
            print("Not enough content segments found, falling back to standard sync")
            return self.synchronize(ref_audio, foreign_audio, ref_sr, foreign_sr, 
                                   method, visualize, viz_path)
        
        # Use the first segment for initial alignment
        ref_first_seg = ref_segments[0]
        foreign_first_seg = foreign_segments[0]
        
        # Use the last segment for end alignment (for duration calculation)
        ref_last_seg = ref_segments[-1]
        foreign_last_seg = foreign_segments[-1]
        
        print(f"Reference content spans from {ref_first_seg[0]:.3f}s to {ref_last_seg[1]:.3f}s")
        print(f"Foreign content spans from {foreign_first_seg[0]:.3f}s to {foreign_last_seg[1]:.3f}s")
        
        # Calculate the offset to align first detected sounds
        # This offset represents how much we need to shift the foreign audio
        # to align its first sound with the reference first sound
        start_offset = ref_first_seg[0] - foreign_first_seg[0]
        print(f"Start offset (first content alignment): {start_offset:.3f}s")
        
        # Calculate content duration ratio for stretching
        ref_content_duration = ref_last_seg[1] - ref_first_seg[0]
        foreign_content_duration = foreign_last_seg[1] - foreign_first_seg[0]
        
        if foreign_content_duration > 0 and ref_content_duration > 0:
            content_ratio = foreign_content_duration / ref_content_duration
            print(f"Content duration ratio: {content_ratio:.4f}")
            print(f"Reference content duration: {ref_content_duration:.3f}s")
            print(f"Foreign content duration: {foreign_content_duration:.3f}s")
        else:
            content_ratio = 1.0
            print("Could not calculate content ratio, using 1.0")
        
        # Apply offset to align the first sounds
        print(f"Applying content-based offset: {start_offset:.3f}s")
        offset_audio = self.apply_offset(foreign_audio, start_offset, foreign_sr)
        
        # Apply time-stretching if needed to match the content duration
        if abs(content_ratio - 1.0) > 0.01:
            if content_ratio > 1.0:
                # Foreign is longer than reference, speed it up
                stretch_factor = content_ratio
                print(f"Content is longer, speeding up with factor: {stretch_factor:.4f}")
            else:
                # Foreign is shorter than reference, slow it down
                stretch_factor = 1.0 / content_ratio
                print(f"Content is shorter, slowing down with factor: {stretch_factor:.4f}")
                
            aligned_audio = self.time_stretch_audio(offset_audio, stretch_factor, foreign_sr)
        else:
            print("No time-stretching needed (ratio near 1.0)")
            aligned_audio = offset_audio
        
        # Create a detailed sync info dictionary
        sync_info = {
            "method": "content_aware",
            "offset_seconds": start_offset,
            "content_segments": {
                "reference": ref_segments,
                "foreign": foreign_segments
            },
            "content_start_points": {
                "reference": ref_first_seg[0],
                "foreign": foreign_first_seg[0]
            },
            "content_end_points": {
                "reference": ref_last_seg[1],
                "foreign": foreign_last_seg[1]
            },
            "duration_ratio": content_ratio
        }
        
        # Visualize the alignment if requested
        if visualize and viz_path:
            self._visualize_content_alignment(ref_segments, foreign_segments, 
                                            start_offset, content_ratio, viz_path)
        
        return aligned_audio, sync_info
    
    def _visualize_content_alignment(self, ref_segments, foreign_segments, offset, ratio, save_path):
        """
        Visualize the content-aware alignment between audio tracks
        """
        plt.figure(figsize=(12, 8))
        
        # Plot reference segments
        for i, (start, end) in enumerate(ref_segments):
            plt.plot([start, end], [1, 1], 'r-', linewidth=4, alpha=0.6)
            
            # Label first and last segment
            if i == 0 or i == len(ref_segments)-1:
                plt.text(start, 1.1, f"{start:.2f}s", color='r', fontsize=9, ha='left')
                plt.text(end, 1.1, f"{end:.2f}s", color='r', fontsize=9, ha='right')
        
        # Plot foreign segments (adjusted by offset)
        for i, (start, end) in enumerate(foreign_segments):
            # Apply the offset to show where it will be after sync
            adj_start = (start * ratio) + offset
            adj_end = (end * ratio) + offset
            
            plt.plot([adj_start, adj_end], [0.5, 0.5], 'b-', linewidth=4, alpha=0.6)
            
            # Label first and last segment
            if i == 0 or i == len(foreign_segments)-1:
                plt.text(adj_start, 0.4, f"{start:.2f}s → {adj_start:.2f}s", 
                        color='b', fontsize=9, ha='left')
                plt.text(adj_end, 0.4, f"{end:.2f}s → {adj_end:.2f}s", 
                        color='b', fontsize=9, ha='right')
        
        # Add reference lines for important sync points
        if ref_segments and foreign_segments:
            first_ref = ref_segments[0][0]
            first_for = (foreign_segments[0][0] * ratio) + offset
            
            last_ref = ref_segments[-1][1]  
            last_for = (foreign_segments[-1][1] * ratio) + offset
            
            # Plot alignment lines
            plt.plot([first_ref, first_ref], [0.4, 1.2], 'k--', alpha=0.5)
            plt.plot([last_ref, last_ref], [0.4, 1.2], 'k--', alpha=0.5)
            
            # Annotations
            plt.annotate('Start Alignment', 
                        xy=(first_ref, 0.75), xytext=(first_ref+2, 0.75),
                        arrowprops=dict(arrowstyle='->'))
                        
            plt.annotate('End Alignment', 
                        xy=(last_ref, 0.75), xytext=(last_ref-2, 0.75),
                        arrowprops=dict(arrowstyle='->'))
        
        # Add labels and legend
        plt.title('Content-Aware Audio Alignment')
        plt.xlabel('Time (seconds)')
        plt.yticks([0.5, 1], ['Foreign Audio', 'Reference Audio'])
        plt.grid(True, alpha=0.3)
        
        # Add sync parameters as text
        plt.figtext(0.02, 0.02, 
                   f"Offset: {offset:.3f}s, Duration Ratio: {ratio:.4f}", 
                   ha="left", fontsize=10,
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Content alignment visualization saved to: {save_path}")
    
    def synchronize(self, ref_audio, foreign_audio, ref_sr, foreign_sr, method='auto', 
                  visualize=False, viz_path=None, duration_ratio=None):
        """
        Synchronize foreign audio to reference audio
        """
        print("Creating audio fingerprints...")
        ref_fingerprint = self.create_energy_fingerprint(ref_audio, ref_sr)
        foreign_fingerprint = self.create_energy_fingerprint(foreign_audio, foreign_sr)
        
        # Find global offset using cross-correlation
        print("Finding global offset...")
        global_offset, correlation, confidence_value = self.find_global_offset(
            ref_fingerprint, foreign_fingerprint, self.window_size
        )
        
        print(f"Detected global offset: {global_offset:.2f} seconds (confidence: {confidence_value:.2f})")
        
        # Method selection logic
        use_stretch = False
        
        if method == 'stretch':
            use_stretch = True
            print("Using time-stretching method (forced)...")
        elif method == 'offset':
            use_stretch = False
            print("Using global offset method (forced)...")
        elif confidence_value < 3.0:  # Low correlation confidence suggests need for stretching
            use_stretch = True
            print("Using automatic method selection: Stretching (low correlation confidence)")
        else:
            use_stretch = False
            print("Using automatic method selection: Global offset (good correlation)")
        
        # Apply the selected method
        if not use_stretch:
            # For simple cases, just apply the global offset
            print("Applying global offset...")
            
            # Double-check if the offset makes sense
            if abs(global_offset) > len(foreign_audio) / foreign_sr * 0.9:
                print(f"Warning: Very large offset detected ({global_offset:.2f}s), possibly incorrect")
                print("Attempting to refine offset calculation...")
                
                # Try with onset features instead of energy
                ref_onset = self.create_onset_fingerprint(ref_audio, ref_sr)
                foreign_onset = self.create_onset_fingerprint(foreign_audio, foreign_sr)
                refined_offset, _, refined_confidence = self.find_global_offset(ref_onset, foreign_onset, self.window_size)
                
                if refined_confidence > confidence_value:
                    print(f"Using refined offset: {refined_offset:.2f}s (confidence: {refined_confidence:.2f})")
                    global_offset = refined_offset
                    confidence_value = refined_confidence
                else:
                    print("Couldn't find better offset, continuing with original")
            
            # Apply padding or trimming based on the offset
            aligned_audio = self.apply_offset(foreign_audio, global_offset, foreign_sr)
            
            # Apply duration ratio if specified
            if duration_ratio is not None and abs(duration_ratio - 1.0) > 0.01:
                print(f"Applying specified duration ratio: {duration_ratio:.4f}")
                if duration_ratio > 1.0:
                    # Foreign is longer than reference, speed it up
                    stretch_factor = duration_ratio
                    operation = "speed up"
                else:
                    # Foreign is shorter than reference, slow it down
                    stretch_factor = 1.0 / duration_ratio
                    operation = "slow down"
                
                print(f"This will {operation} the audio to match reference length")
                aligned_audio = self.time_stretch_audio(aligned_audio, stretch_factor, foreign_sr)
            
            # Verify alignment result
            if len(aligned_audio) < foreign_sr * 3:  # Less than 3 seconds
                print("Warning: Resulting audio is very short, offset may be incorrect")
                print("Reverting to original audio with zero offset")
                aligned_audio = foreign_audio
            
            sync_info = {
                "method": "global_offset",
                "offset_seconds": global_offset,
                "correlation_confidence": confidence_value
            }
            
            if duration_ratio is not None:
                sync_info["duration_ratio"] = duration_ratio
                
        else:
            # For more complex cases, use time-stretching
            print("Using time-stretching method...")
            
            # Apply the offset first
            offset_audio = self.apply_offset(foreign_audio, global_offset, foreign_sr)
            
            # Determine the stretch factor
            if duration_ratio is None:
                # Calculate an approximate stretch factor based on audio lengths
                ref_duration = len(ref_audio) / ref_sr
                foreign_duration = len(foreign_audio) / foreign_sr
                duration_ratio = foreign_duration / ref_duration
            
            # Apply time stretching
            if duration_ratio > 1.0:
                # Foreign is longer than reference, speed it up
                stretch_factor = duration_ratio
                operation = "speed up"
            else:
                # Foreign is shorter than reference, slow it down
                stretch_factor = 1.0 / duration_ratio
                operation = "slow down"
            
            print(f"Foreign/reference duration ratio: {duration_ratio:.4f}")
            print(f"This will {operation} the audio to match reference length")
            
            aligned_audio = self.time_stretch_audio(offset_audio, stretch_factor, foreign_sr)
            
            sync_info = {
                "method": "time_stretch",
                "offset_seconds": global_offset,
                "duration_ratio": duration_ratio,
                "stretch_factor": stretch_factor
            }
        
        # Create visualization if requested
        if visualize and viz_path:
            self._visualize_alignment(ref_fingerprint, foreign_fingerprint, global_offset, correlation, viz_path)
        
        return aligned_audio, sync_info
    
    def _visualize_alignment(self, ref_fingerprint, foreign_fingerprint, offset, correlation, save_path):
        """
        Visualize the alignment between two audio tracks
        """
        plt.figure(figsize=(12, 10))
        
        # Plot reference fingerprint
        plt.subplot(3, 1, 1)
        plt.plot(ref_fingerprint)
        plt.title("Reference Audio Fingerprint")
        plt.xlabel("Time (windows)")
        plt.ylabel("Energy")
        
        # Plot foreign fingerprint
        plt.subplot(3, 1, 2)
        plt.plot(foreign_fingerprint)
        plt.title("Foreign Audio Fingerprint")
        plt.xlabel("Time (windows)")
        plt.ylabel("Energy")
        
        # Plot correlation
        plt.subplot(3, 1, 3)
        plt.plot(correlation)
        plt.axvline(x=len(foreign_fingerprint)-1, color='r', linestyle='--')
        plt.title(f"Cross-correlation (Offset: {offset:.2f}s)")
        plt.xlabel("Lag (windows)")
        plt.ylabel("Correlation")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Visualization saved to: {save_path}")
