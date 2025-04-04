"""
Optimized Audio and Video Utilities Module

This module provides essential utilities for working with audio and video files:
- Audio track extraction and information
- Video frame extraction with scene detection
- Duration calculations
- Muxing operations

Optimized to focus on the functionality needed for visual synchronization with multiple audio tracks.
"""

import os
import subprocess
import json
import tempfile
import numpy as np
import librosa
from scipy.io import wavfile
import cv2


class EnhancedAudioVideoUtils:
    """Essential utilities for audio and video processing"""
    
    @staticmethod
    def get_audio_tracks(video_path):
        """
        Get a list of audio tracks available in the video file
        
        Parameters:
        video_path (str): Path to the video file
        
        Returns:
        list: List of dictionaries containing track info (index, language, title)
        """
        try:
            # Run ffprobe to get stream information
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a', video_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse the JSON output
            data = json.loads(result.stdout)
            
            tracks = []
            
            # Extract information about each audio stream
            for i, stream in enumerate(data.get('streams', [])):
                track_info = {
                    'index': stream.get('index', i),  # This is the actual stream index
                    'codec': stream.get('codec_name', 'unknown'),
                    'language': stream.get('tags', {}).get('language', 'unknown'),
                    'title': stream.get('tags', {}).get('title', f'Track {i+1}'),
                    'channels': stream.get('channels', 2),
                    'sample_rate': stream.get('sample_rate', '48000')
                }
                tracks.append(track_info)
            
            return tracks
        
        except Exception as e:
            print(f"Error getting audio tracks: {e}")
            # Return a default track if we can't get the information
            return [{'index': 0, 'codec': 'unknown', 'language': 'unknown', 'title': 'Default Track'}]
    
    @staticmethod
    def select_audio_track(video_path, force_track=None):
        """
        Select an audio track from a video file
        
        Parameters:
        video_path (str): Path to the video file
        force_track (int): Force selection of a specific track
        
        Returns:
        int: Selected track index
        """
        tracks = EnhancedAudioVideoUtils.get_audio_tracks(video_path)
        
        if not tracks:
            print("No audio tracks found in the video file.")
            return None
        
        print(f"\nFound {len(tracks)} audio tracks in {os.path.basename(video_path)}:")
        
        for i, track in enumerate(tracks):
            lang = track.get('language', 'unknown')
            title = track.get('title', f'Track {i+1}')
            codec = track.get('codec', 'unknown')
            channels = track.get('channels', 2)
            sample_rate = track.get('sample_rate', '48000')
            print(f"{i}: {title} [{lang}] - {codec}, {channels}ch, {sample_rate}Hz (stream #{track['index']})")
        
        # Use forced track if specified
        if force_track is not None:
            if 0 <= force_track < len(tracks):
                track_info = tracks[force_track]
                print(f"\nUsing forced audio track: {force_track} - {track_info['title']} [{track_info['language']}]")
                return force_track
            else:
                print(f"\nWarning: Forced track {force_track} out of range. Available tracks: 0-{len(tracks)-1}")
        
        # If only one track or forced track out of range, use the first track
        if len(tracks) == 1:
            print(f"\nAutomatically selected the only available audio track: {tracks[0]['title']}")
            return 0
        
        # Ask user to select a track
        while True:
            try:
                selection = int(input("\nSelect audio track number to synchronize: "))
                if 0 <= selection < len(tracks):
                    return selection
                else:
                    print(f"Invalid selection. Please enter a number between 0 and {len(tracks)-1}.")
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nSelection cancelled. Using first track.")
                return 0
    
    @staticmethod
    def extract_audio(video_path, output_path=None, track_index=None, sample_rate=44100, channels=1):
        """
        Extract audio from video file using FFmpeg
        
        Parameters:
        video_path (str): Path to the video file
        output_path (str): Path to save the extracted audio (optional)
        track_index (int): Index of the audio track to extract (optional)
        sample_rate (int): Target sample rate
        channels (int): Number of channels (1=mono, 2=stereo)
        
        Returns:
        str: Path to the extracted audio file
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_audio.wav"
        
        ffmpeg_command = [
            'ffmpeg', '-i', video_path, '-vn',
            '-acodec', 'pcm_s16le',  # 16-bit PCM for best compatibility
            '-ar', str(sample_rate),  # Target sample rate
            '-ac', str(channels)      # Mono or stereo
        ]
        
        # If a specific track is selected, add the map option
        if track_index is not None:
            tracks = EnhancedAudioVideoUtils.get_audio_tracks(video_path)
            if 0 <= track_index < len(tracks):
                stream_index = tracks[track_index]['index']
                ffmpeg_command.extend(['-map', f'0:{stream_index}'])
                print(f"Mapping audio stream index {stream_index} (track {track_index})")
        
        # Add output file and overwrite flag
        ffmpeg_command.extend([output_path, '-y'])
        
        try:
            # Run FFmpeg with detailed error handling
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            if result.stderr and ('Error' in result.stderr or 'error' in result.stderr):
                print(f"FFmpeg warnings: {result.stderr}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            raise e
    
    @staticmethod
    def load_audio(audio_path, sample_rate=44100, normalize=True):
        """
        Load audio file and return the audio data
        
        Parameters:
        audio_path (str): Path to the audio file
        sample_rate (int): Target sample rate
        normalize (bool): Whether to normalize audio amplitude
        
        Returns:
        tuple: Audio data and sample rate
        """
        try:
            # Use librosa for high-quality loading
            audio_data, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            
            # Normalize if requested
            if normalize and np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            return audio_data, sr
        except Exception as e:
            print(f"Error loading audio with librosa: {e}")
            print("Falling back to scipy wavfile...")
            
            try:
                # Fallback to scipy
                sr, audio_data = wavfile.read(audio_path)
                
                # Convert to float and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                # Resample if needed
                if sr != sample_rate:
                    print(f"Resampling from {sr}Hz to {sample_rate}Hz")
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                    sr = sample_rate
                
                # Normalize if requested
                if normalize and np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                return audio_data, sr
            except Exception as e2:
                print(f"Error loading audio with scipy: {e2}")
                raise RuntimeError(f"Could not load audio file {audio_path}: {e2}")
    
    @staticmethod
    def extract_frames(video_path, output_dir=None, max_frames=500, fps=1.0, 
                      target_width=640, use_scene_detect=True, scene_threshold=0.25, 
                      reference_aspect_ratio=None, preserve_aspect_ratio=False):
        """
        Extract frames from a video optimized for scene detection
        
        Parameters:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        max_frames (int): Maximum number of frames to extract
        fps (float): Frames per second for uniform extraction
        target_width (int): Width to resize frames to
        use_scene_detect (bool): Whether to use scene detection for frame selection
        scene_threshold (float): Threshold for scene detection
        reference_aspect_ratio (float): Aspect ratio to match
        preserve_aspect_ratio (bool): Whether to preserve aspect ratio
        
        Returns:
        tuple: (frames dictionary, timestamps list)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_aspect_ratio = original_width / original_height
        
        if reference_aspect_ratio is None:
            aspect_ratio = original_aspect_ratio
        else:
            aspect_ratio = reference_aspect_ratio
        
        if preserve_aspect_ratio:
            target_height = int(target_width / aspect_ratio)
        else:
            target_height = int(target_width / aspect_ratio)
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.2f}s, FPS: {video_fps:.2f}, Total frames: {total_frames}")
        print(f"Original resolution: {original_width}x{original_height}, Resizing to: {target_width}x{target_height}")
        
        if use_scene_detect:
            frame_indices = []
            prev_frame = None
            scene_scores = []
            
            sample_step = max(1, total_frames // 300)
            for frame_idx in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                small_frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    score = np.mean(frame_diff) / 255.0
                    scene_scores.append((frame_idx, score))
                
                prev_frame = gray
            
            scene_scores.sort(key=lambda x: x[1], reverse=True)
            
            threshold_idx = int(len(scene_scores) * scene_threshold)
            if threshold_idx < len(scene_scores):
                min_score = scene_scores[threshold_idx][1]
            else:
                min_score = 0.05
            
            scenes = []
            for idx, score in scene_scores:
                if score >= min_score:
                    scenes.append(idx)
            
            if len(scenes) > max_frames:
                scenes = sorted(scenes)
                step = len(scenes) / max_frames
                frame_indices = [scenes[int(i * step)] for i in range(max_frames)]
            else:
                frame_indices = sorted(scenes)
                
            if len(frame_indices) < max_frames:
                remaining = max_frames - len(frame_indices)
                even_indices = np.linspace(0, total_frames-1, remaining+2, dtype=int)[1:-1]
                for idx in even_indices:
                    if idx not in frame_indices:
                        frame_indices.append(idx)
                
            frame_indices = sorted(frame_indices)
            print(f"Scene detection found {len(frame_indices)} frames")
        else:
            frame_interval = int(video_fps / fps)
            frame_interval = max(1, frame_interval)
            frames_to_extract = min(max_frames, (total_frames // frame_interval) + 1)
            frame_indices = [i * frame_interval for i in range(frames_to_extract)]
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        frames = {}
        timestamps = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = frame_idx / video_fps
            timestamps.append(timestamp)
            
            if preserve_aspect_ratio:
                h, w = frame.shape[:2]
                if w/h > aspect_ratio:
                    new_w = int(h * aspect_ratio)
                    start_x = (w - new_w) // 2
                    frame = frame[:, start_x:start_x+new_w]
                elif w/h < aspect_ratio:
                    new_h = int(w / aspect_ratio)
                    start_y = (h - new_h) // 2
                    frame = frame[start_y:start_y+new_h]
            
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized_frame = clahe.apply(gray_frame)
            
            filtered_frame = cv2.bilateralFilter(equalized_frame, 9, 75, 75)
            
            frames[timestamp] = {
                'original': resized_frame.copy() if output_dir else None,
                'gray': gray_frame,
                'equalized': equalized_frame,
                'filtered': filtered_frame
            }
            
            if output_dir:
                frame_path = os.path.join(output_dir, f"frame_{timestamp:.3f}.jpg")
                cv2.imwrite(frame_path, resized_frame)
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames")
        return frames, sorted(timestamps)
    
    @staticmethod
    def save_audio(audio_data, output_path, sr, bit_depth=16):
        """
        Save audio data to a file
        
        Parameters:
        audio_data (numpy.ndarray): Audio data
        output_path (str): Output file path
        sr (int): Sample rate
        bit_depth (int): Bit depth (16 or 24 or 32)
        """
        # Ensure audio is normalized
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        if bit_depth == 16:
            # Standard 16-bit PCM
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data_int = (audio_data * 32767).astype(np.int16)
            wavfile.write(output_path, sr, audio_data_int)
        elif bit_depth == 24:
            # For 24-bit, use ffmpeg since scipy doesn't support it directly
            # Save as 32-bit float first
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            wavfile.write(temp_file.name, sr, audio_data.astype(np.float32))
            
            # Convert to 24-bit using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', temp_file.name, 
                '-acodec', 'pcm_s24le', 
                '-ar', str(sr), 
                output_path, '-y'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
        else:
            # 32-bit float
            wavfile.write(output_path, sr, audio_data.astype(np.float32))
    
    @staticmethod
    def mux_audio_with_video(reference_video_path, audio_file_path, output_path, 
                             audio_language="Unknown", audio_title=None,
                             preserve_original_audio=True, video_codec="copy"):
        """
        Mux audio file with the reference video
        
        Parameters:
        reference_video_path (str): Path to the reference video
        audio_file_path (str): Path to the audio file to add
        output_path (str): Path to save the output video
        audio_language (str): Language code for the audio
        audio_title (str): Title for the audio track
        preserve_original_audio (bool): Whether to keep original audio tracks
        video_codec (str): Video codec to use ("copy" for no re-encoding)
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Default title to language if not specified
            if audio_title is None:
                audio_title = audio_language
            
            # Build the ffmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', reference_video_path,  # Input reference video
                '-i', audio_file_path,      # Input synchronized audio
                '-map', '0:v',              # Map video stream from reference
                '-map', '1:a'               # Map audio stream from synced audio
            ]
            
            # Optionally map original audio tracks
            if preserve_original_audio:
                # Get original audio tracks
                original_tracks = EnhancedAudioVideoUtils.get_audio_tracks(reference_video_path)
                for i in range(len(original_tracks)):
                    ffmpeg_cmd.extend(['-map', f'0:a:{i}'])
            
            # Add metadata for the synchronized audio track
            ffmpeg_cmd.extend([
                '-disposition:a:0', 'default',  # Make synced track the default
                '-metadata:s:a:0', f'title={audio_title}',
                '-metadata:s:a:0', f'language={audio_language}'
            ])
            
            # Add encoding parameters
            ffmpeg_cmd.extend([
                '-c:v', video_codec,    # Video codec (copy = no re-encoding)
                '-c:a', 'aac',          # Convert audio to AAC
                '-b:a', '192k',         # Audio bitrate
                output_path,
                '-y'                    # Overwrite if exists
            ])
            
            # Run the ffmpeg command with proper error handling
            result = subprocess.run(ffmpeg_cmd, check=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            
            print(f"Successfully created muxed video: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error muxing audio with video: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    @staticmethod
    def get_video_duration(video_path):
        """
        Get the duration of a video in seconds using FFprobe
        
        Parameters:
        video_path (str): Path to the video file
        
        Returns:
        float: Duration in seconds
        """
        try:
            # Use FFprobe for accurate duration
            result = subprocess.run([
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                video_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
            
        except Exception as e:
            print(f"Error using FFprobe: {e}")
            print("Falling back to OpenCV for duration calculation")
            
            # Fallback to OpenCV
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {video_path}")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate duration
                duration = frame_count / fps if fps > 0 else 0
                
                # Release the video capture
                cap.release()
                
                return duration
            except Exception as e2:
                print(f"Error getting video duration: {e2}")
                return 0