"""
Enhanced Visual Synchronization Module

This module provides improved functionality for synchronizing videos using visual features
with enhanced preprocessing, AKAZE feature detection, and multi-pass anchor point detection.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
import logging


class EnhancedVisualSynchronizer:
    """Enhanced class for synchronizing videos using visual features"""
    
    def __init__(self, 
                 feature_method='akaze',       # Default changed to AKAZE for better matching
                 matching_ratio=0.85,          # Increased from 0.75 to be more lenient
                 min_matches=4,                # Reduced from 6 to get more anchor points
                 ransac_threshold=8.0,         # Increased from 5.0 to be more forgiving
                 sample_mode='multi_pass',     # New default sampling mode
                 max_comparisons=800,          # Increased from 500
                 temporal_consistency=True,    # Check temporal consistency
                 multi_resolution=True,        # Use multi-resolution approach
                 enhanced_preprocessing=True   # Use enhanced preprocessing
                ):
        """
        Initialize the enhanced visual synchronizer
        
        Parameters:
        feature_method (str): Method used for feature extraction ('orb', 'sift', 'akaze')
        matching_ratio (float): Ratio threshold for feature matching
        min_matches (int): Minimum number of good matches required
        ransac_threshold (float): Maximum allowed reprojection error in RANSAC
        sample_mode (str): Frame sampling strategy
        max_comparisons (int): Maximum number of frame comparisons to perform
        temporal_consistency (bool): Whether to enforce temporal consistency
        multi_resolution (bool): Whether to use multi-resolution approach
        enhanced_preprocessing (bool): Whether to use enhanced preprocessing
        """
        self.feature_method = feature_method
        self.matching_ratio = matching_ratio
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.sample_mode = sample_mode
        self.max_comparisons = max_comparisons
        self.temporal_consistency = temporal_consistency
        self.multi_resolution = multi_resolution
        self.enhanced_preprocessing = enhanced_preprocessing
        
        # Logger for detailed diagnostics
        self.logger = logging.getLogger('visual_sync')
        
        # Initialize feature detector based on method
        self._initialize_feature_detector()
            
    def _initialize_feature_detector(self):
        """Initialize the feature detector based on the selected method"""
        if self.feature_method == 'sift':
            self.feature_detector = cv2.SIFT_create(
                nfeatures=0,      # Auto-determine number of features
                nOctaveLayers=5,  # More octave layers for better scale invariance
                contrastThreshold=0.03,  # Lower threshold to detect more features
                edgeThreshold=15
            )
            # For SIFT, use FLANN-based matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        elif self.feature_method == 'akaze':
            self.feature_detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,  # Auto size
                descriptor_channels=3,
                threshold=0.0008,  # Lower threshold to detect more features
                nOctaves=4,
                nOctaveLayers=4
            )
            # For AKAZE, use FLANN-based matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        else:  # Default to ORB with improved parameters
            self.feature_detector = cv2.ORB_create(
                nfeatures=2500,  # Increased from 2000
                scaleFactor=1.1,  # Reduced from 1.2 for more scale levels
                nlevels=10,      # Increased from 8
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31
            )
            self.feature_method = 'orb'
            # For ORB, use Brute Force Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _check_perspective_quality(self, H):
        """
        Check if a homography matrix represents a reasonable transformation
        
        Parameters:
        H (numpy.ndarray): Homography matrix
        
        Returns:
        float: Quality score between 0 and 1
        """
        # Create some points for testing the transformation
        h, w = 100, 100
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1, 1, 2)
        
        # Apply the homography transformation
        try:
            dst = cv2.perspectiveTransform(pts, H)
            
            # Calculate the area before and after transformation
            area_before = w * h
            
            # Calculate area after transformation using shoelace formula
            x = dst[:, 0, 0]
            y = dst[:, 0, 1]
            area_after = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # Check area ratio (more lenient)
            area_ratio = min(area_after / area_before, area_before / area_after)
            
            # Check angles
            angles_before = []
            angles_after = []
            
            for i in range(4):
                # Vectors for before transformation
                v1_before = pts[(i+1)%4, 0] - pts[i, 0]
                v2_before = pts[(i-1)%4, 0] - pts[i, 0]
                
                # Vectors for after transformation
                v1_after = dst[(i+1)%4, 0] - dst[i, 0]
                v2_after = dst[(i-1)%4, 0] - dst[i, 0]
                
                # Calculate angles (dot product formula)
                dot_before = np.dot(v1_before, v2_before)
                dot_after = np.dot(v1_after, v2_after)
                
                mag_before = np.linalg.norm(v1_before) * np.linalg.norm(v2_before)
                mag_after = np.linalg.norm(v1_after) * np.linalg.norm(v2_after)
                
                if mag_before > 0 and mag_after > 0:
                    cos_before = min(1.0, max(-1.0, dot_before / mag_before))
                    cos_after = min(1.0, max(-1.0, dot_after / mag_after))
                    
                    angles_before.append(np.arccos(cos_before))
                    angles_after.append(np.arccos(cos_after))
            
            # Calculate angle differences
            angle_diffs = []
            for i in range(len(angles_before)):
                angle_diffs.append(abs(angles_before[i] - angles_after[i]))
            
            max_angle_diff = max(angle_diffs) if angle_diffs else np.pi
            
            # Be more lenient with angle differences
            angle_score = max(0, 1 - (max_angle_diff / (np.pi * 0.8)))
            
            # Combined quality score with more weight on area ratio
            quality_score = 0.8 * area_ratio + 0.2 * angle_score
            
            # More lenient minimum quality threshold
            return max(0.3, quality_score)
            
        except Exception as e:
            self.logger.debug(f"Perspective quality check failed: {str(e)}")
            return 0.0
    
    def enhance_frame(self, frame):
        """
        Apply enhanced preprocessing to a frame
        
        Parameters:
        frame (numpy.ndarray): Input grayscale frame
        
        Returns:
        numpy.ndarray: Enhanced frame
        """
        if not self.enhanced_preprocessing:
            return frame
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clip limit
        enhanced = clahe.apply(frame)
        
        # Apply mild Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply Bilateral filter to preserve edges while reducing noise
        enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)  # Adjusted parameters
        
        return enhanced
    
    def compute_features(self, frames):
        """
        Compute features for a set of frames with enhanced preprocessing
        
        Parameters:
        frames (dict): Dictionary of frame data {timestamp: frame_data}
        
        Returns:
        dict: Dictionary of features {timestamp: features}
        """
        features = {}
        print(f"Computing {self.feature_method.upper()} features for {len(frames)} frames")
        
        for timestamp, frame_data in frames.items():
            # Get preprocessed frame
            gray_frame = frame_data['gray']
            
            # Apply enhanced processing if enabled
            if self.enhanced_preprocessing:
                processed_frame = self.enhance_frame(gray_frame)
            else:
                processed_frame = frame_data['equalized']  # Use the existing equalized frame
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.feature_detector.detectAndCompute(processed_frame, None)
            
            # Store features if valid
            if descriptors is not None and len(keypoints) > 0:
                features[timestamp] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'frame': processed_frame  # Store processed frame for visualization
                }
                
                # Log detail about number of keypoints found
                self.logger.debug(f"Frame at {timestamp:.2f}s: {len(keypoints)} keypoints detected")
        
        print(f"Computed features for {len(features)} frames")
        return features
    
    def select_frame_pairs_multi_pass(self, ref_timestamps, foreign_timestamps):
        """
        Multi-pass frame pairing strategy - start with keyframes, then add more
        based on initial matches
        
        Parameters:
        ref_timestamps (list): List of reference timestamps
        foreign_timestamps (list): List of foreign timestamps
        
        Returns:
        list: List of (ref_ts, for_ts) pairs to compare
        """
        pairs = []
        
        # PASS 1: Start with key frames (beginning, middle segments, end)
        # This helps establish a rough alignment
        ref_keyframes = []
        segments = 7  # Increased from 5 for more segments
        
        for i in range(segments + 1):
            idx = int(i * (len(ref_timestamps) - 1) / segments)
            ref_keyframes.append(ref_timestamps[idx])
        
        foreign_keyframes = []
        for i in range(segments + 1):
            idx = int(i * (len(foreign_timestamps) - 1) / segments)
            foreign_keyframes.append(foreign_timestamps[idx])
        
        # Add combinations of these key frames with neighbors
        for ref_ts in ref_keyframes:
            ref_idx = ref_timestamps.index(ref_ts)
            
            for for_ts in foreign_keyframes:
                for_idx = foreign_timestamps.index(for_ts)
                
                # Add the key point and more neighbors
                for r_delta in [-2, -1, 0, 1, 2]:  # Expanded delta
                    for f_delta in [-2, -1, 0, 1, 2]:  # Expanded delta
                        r_idx = ref_idx + r_delta
                        f_idx = for_idx + f_delta
                        
                        if (0 <= r_idx < len(ref_timestamps) and 
                            0 <= f_idx < len(foreign_timestamps)):
                            pairs.append((ref_timestamps[r_idx], foreign_timestamps[f_idx]))
        
        # PASS 2: Add adaptive combinations based on possible offsets and ratios
        # This is a refined version of the original adaptive sampling
        min_ratio = 0.85  # Adjusted from 0.9
        max_ratio = 1.15  # Adjusted from 1.1
        
        # Estimate possible global offsets (in seconds)
        possible_offsets = [0]  # Start with no offset
        
        # Add offsets based on video lengths
        ref_duration = ref_timestamps[-1]
        for_duration = foreign_timestamps[-1]
        possible_offsets.append(for_duration - ref_duration)  # Trailing difference
        
        # Add more percentage-based offsets
        for pct in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
            possible_offsets.append(ref_duration * pct)
        
        # More granular ratios for better coverage
        ratios = [min_ratio, 0.92, 0.96, 1.0, 1.04, 1.08, max_ratio]  # Added more ratios
        
        # Calculate frames to sample for each combo
        max_per_combo = self.max_comparisons // (len(ratios) * len(possible_offsets) * 3)
        sample_count = min(max_per_combo, len(ref_timestamps) // 5)
        
        # Sample throughout the timeline, focusing on beginning and end
        sample_indices = []
        
        # Beginning frames (higher density)
        begin_indices = np.linspace(0, len(ref_timestamps)//4, sample_count//3, dtype=int)
        sample_indices.extend(begin_indices)
        
        # Middle frames
        mid_indices = np.linspace(len(ref_timestamps)//4, 3*len(ref_timestamps)//4, sample_count//3, dtype=int)
        sample_indices.extend(mid_indices)
        
        # End frames (higher density)
        end_indices = np.linspace(3*len(ref_timestamps)//4, len(ref_timestamps)-1, sample_count//3, dtype=int)
        sample_indices.extend(end_indices)
        
        # Generate pairs for each ratio/offset
        for ratio in ratios:
            for offset in possible_offsets:
                for idx in sample_indices:
                    ref_ts = ref_timestamps[idx]
                    
                    # Calculate expected foreign timestamp
                    expected_for_ts = (ref_ts * ratio) + offset
                    
                    # Find closest foreign timestamp
                    closest_idx = 0
                    closest_diff = float('inf')
                    
                    for j, for_ts in enumerate(foreign_timestamps):
                        diff = abs(for_ts - expected_for_ts)
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_idx = j
                    
                    # Add the closest match and more neighbors for denser sampling
                    for delta in [-6, -4, -2, 0, 2, 4, 6]:  # Expanded delta range
                        neighbor_idx = closest_idx + delta
                        if 0 <= neighbor_idx < len(foreign_timestamps):
                            pairs.append((ref_ts, foreign_timestamps[neighbor_idx]))
        
        # Remove duplicates
        pairs = list(set(pairs))
        
        # Limit to max comparisons if needed
        if len(pairs) > self.max_comparisons:
            # Sort pairs by absolute timestamp difference to prioritize likely matches
            sorted_pairs = sorted(pairs, key=lambda p: abs(p[0] - p[1]))
            pairs = sorted_pairs[:self.max_comparisons]
        
        print(f"Multi-pass sampling: {len(pairs)} frame pairs selected")
        return pairs
    
    def select_frame_pairs(self, ref_timestamps, foreign_timestamps):
        """
        Select frame pairs to compare using the selected sampling strategy
        
        Parameters:
        ref_timestamps (list): List of reference timestamps
        foreign_timestamps (list): List of foreign timestamps
        
        Returns:
        list: List of (ref_ts, for_ts) pairs to compare
        """
        # Use multi-pass strategy if selected
        if self.sample_mode == 'multi_pass':
            return self.select_frame_pairs_multi_pass(ref_timestamps, foreign_timestamps)
        
        # Otherwise, use the original strategies from the base class
        pairs = []
        
        if self.sample_mode == 'uniform':
            # Select evenly spaced frames from both videos
            # This assumes similar timing (with possible offset/stretching)
            ref_count = len(ref_timestamps)
            for_count = len(foreign_timestamps)
            
            # Determine how many frames to sample from each video
            # to keep total comparisons under the limit
            sample_size = int(np.sqrt(self.max_comparisons))
            sample_size = min(sample_size, min(ref_count, for_count))
            
            # Select evenly spaced frames
            ref_indices = np.linspace(0, ref_count-1, sample_size, dtype=int)
            for_indices = np.linspace(0, for_count-1, sample_size, dtype=int)
            
            ref_samples = [ref_timestamps[i] for i in ref_indices]
            for_samples = [foreign_timestamps[i] for i in for_indices]
            
            # Generate all combinations of sampled frames
            for ref_ts in ref_samples:
                for for_ts in for_samples:
                    pairs.append((ref_ts, for_ts))
            
            print(f"Uniform sampling: {len(pairs)} frame pairs selected")
        
        elif self.sample_mode == 'sparse':
            # Sparse sampling - start with very few frames, then add more if needed
            # Good for when we have strong confidence in offset and ratio
            # Start with keyframes (beginning, middle, end)
            ref_keyframes = [ref_timestamps[0], 
                           ref_timestamps[len(ref_timestamps)//2], 
                           ref_timestamps[-1]]
            
            for_keyframes = [foreign_timestamps[0], 
                           foreign_timestamps[len(foreign_timestamps)//2], 
                           foreign_timestamps[-1]]
            
            # Add some combinations of these keyframes
            for ref_ts in ref_keyframes:
                for for_ts in for_keyframes:
                    pairs.append((ref_ts, for_ts))
                    
            # If we have room for more comparisons, add some random frames
            remaining = self.max_comparisons - len(pairs)
            if remaining > 0:
                # Take random samples
                ref_samples = np.random.choice(ref_timestamps, 
                                            size=min(10, len(ref_timestamps)), 
                                            replace=False)
                for_samples = np.random.choice(foreign_timestamps, 
                                             size=min(10, len(foreign_timestamps)), 
                                             replace=False)
                
                # Add pairs until we reach the limit
                for ref_ts in ref_samples:
                    for for_ts in for_samples:
                        if (ref_ts, for_ts) not in pairs:
                            pairs.append((ref_ts, for_ts))
                            remaining -= 1
                            if remaining <= 0:
                                break
                    if remaining <= 0:
                        break
            
            print(f"Sparse sampling: {len(pairs)} frame pairs selected")
            
        elif self.sample_mode == 'adaptive':
            # Adaptive sampling - assume similar timing with possible stretching/offset
            # Try to match nearby frames in timeline
            max_offset_factor = 0.3  # How far to look from the expected position
            
            # Calculate possible frame rate ratio range
            min_ratio = 0.9  # Slowest playback (e.g., NTSC to PAL)
            max_ratio = 1.1  # Fastest playback (e.g., PAL to NTSC)
            
            # Estimate possible global offsets (in seconds)
            possible_offsets = []
            
            # Try no offset (matching alignment)
            possible_offsets.append(0)
            
            # Try some offsets based on video lengths
            ref_duration = ref_timestamps[-1]
            for_duration = foreign_timestamps[-1]
            possible_offsets.append(for_duration - ref_duration)  # Trailing difference
            possible_offsets.append(for_duration * 0.05)  # Small positive offset (5%)
            possible_offsets.append(-ref_duration * 0.05)  # Small negative offset (5%)
            
            # Explore different combinations of ratio and offset
            ratios = [min_ratio, 1.0, max_ratio]
            
            # Calculate how many frames to sample based on max comparisons
            frames_per_combo = max(1, min(20, self.max_comparisons // (len(ratios) * len(possible_offsets))))
            sample_indices = np.linspace(0, len(ref_timestamps)-1, frames_per_combo, dtype=int)
            
            # Generate pairs for each ratio/offset combination
            for ratio in ratios:
                for offset in possible_offsets:
                    for idx in sample_indices:
                        ref_ts = ref_timestamps[idx]
                        
                        # Calculate expected foreign timestamp
                        expected_for_ts = (ref_ts * ratio) + offset
                        
                        # Find closest foreign timestamp
                        closest_idx = 0
                        closest_diff = float('inf')
                        
                        for j, for_ts in enumerate(foreign_timestamps):
                            diff = abs(for_ts - expected_for_ts)
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_idx = j
                        
                        # Add the closest match and some neighbors
                        for delta in [-2, 0, 2]:  # Try the closest and ±2 frames
                            neighbor_idx = closest_idx + delta
                            if 0 <= neighbor_idx < len(foreign_timestamps):
                                pairs.append((ref_ts, foreign_timestamps[neighbor_idx]))
            
            # Remove duplicates
            pairs = list(set(pairs))
            print(f"Adaptive sampling: {len(pairs)} frame pairs selected")
        
        else:  # Default to brute force (all combinations)
            # Classic brute force - all combinations
            # Use this only for small sets of frames
            for ref_ts in ref_timestamps:
                for for_ts in foreign_timestamps:
                    pairs.append((ref_ts, for_ts))
            
            # If too many pairs, take a random subset
            if len(pairs) > self.max_comparisons:
                print(f"Warning: Too many frame pairs ({len(pairs)}), limiting to {self.max_comparisons}")
                np.random.shuffle(pairs)
                pairs = pairs[:self.max_comparisons]
        
        return pairs
    
    def _match_descriptors(self, ref_desc, for_desc):
        """
        Match descriptors between two frames with appropriate method
        
        Parameters:
        ref_desc (numpy.ndarray): Reference descriptors
        for_desc (numpy.ndarray): Foreign descriptors
        
        Returns:
        list: List of good matches
        """
        if self.feature_method in ['sift', 'akaze']:
            try:
                # For SIFT and AKAZE, use knnMatch with ratio test
                raw_matches = self.matcher.knnMatch(ref_desc, for_desc, k=2)
                
                # Apply Lowe's ratio test (using the customized ratio)
                good_matches = []
                for m, n in raw_matches:
                    if m.distance < self.matching_ratio * n.distance:
                        good_matches.append(m)
                
                return good_matches
            
            except Exception as e:
                self.logger.debug(f"knnMatch failed: {str(e)}")
                # Fall back to regular match
                try:
                    matches = self.matcher.match(ref_desc, for_desc)
                    matches = sorted(matches, key=lambda x: x.distance)
                    return matches[:100]  # Return more top matches (increased from 50)
                except Exception as e2:
                    self.logger.debug(f"Fallback match failed too: {str(e2)}")
                    return []
        else:
            # For ORB with Hamming distance
            try:
                raw_matches = self.matcher.match(ref_desc, for_desc)
                
                # Sort by distance
                raw_matches = sorted(raw_matches, key=lambda x: x.distance)
                
                # Take top matches (up to 150, increased from 100)
                max_matches = min(len(raw_matches), 150)
                return raw_matches[:max_matches]
            except Exception as e:
                self.logger.debug(f"Match failed: {str(e)}")
                return []
    
    def match_frames(self, ref_features, foreign_features):
        """
        Match features between reference and foreign frames with improved matching
        
        Parameters:
        ref_features (dict): Reference features {timestamp: features}
        foreign_features (dict): Foreign features {timestamp: features}
        
        Returns:
        list: List of matched frame pairs with quality scores
        """
        matches = []
        
        # Use a sample of frames if there are too many to process efficiently
        ref_timestamps = sorted(ref_features.keys())
        foreign_timestamps = sorted(foreign_features.keys())
        
        # Select frame pairs to compare
        frame_pairs = self.select_frame_pairs(ref_timestamps, foreign_timestamps)
        
        print(f"Matching features between {len(ref_features)} reference and {len(foreign_features)} foreign frames")
        print(f"Using {self.sample_mode} sampling strategy: {len(frame_pairs)} frame comparisons")
        
        # Create a progress indicator for the user
        total_comparisons = len(frame_pairs)
        last_percentage = -1
        
        for i, (ref_ts, for_ts) in enumerate(frame_pairs):
            # Update progress every 5%
            percentage = int((i / total_comparisons) * 100)
            if percentage % 5 == 0 and percentage != last_percentage:
                print(f"Progress: {percentage}% ({i}/{total_comparisons})")
                last_percentage = percentage
            
            # Skip if either timestamp doesn't have features (could happen if feature extraction failed)
            if ref_ts not in ref_features or for_ts not in foreign_features:
                continue
            
            # Get features for this pair
            ref_kp = ref_features[ref_ts]['keypoints']
            ref_desc = ref_features[ref_ts]['descriptors']
            for_kp = foreign_features[for_ts]['keypoints']
            for_desc = foreign_features[for_ts]['descriptors']
            
            # Skip if too few keypoints - lowered threshold to 3
            if len(ref_kp) < 3 or len(for_kp) < 3:
                continue
            
            # Match descriptors with improved error handling
            try:
                good_matches = self._match_descriptors(ref_desc, for_desc)
                
                # Check if we have enough good matches
                if len(good_matches) >= self.min_matches:
                    # Convert keypoints to numpy arrays for homography estimation
                    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    for_pts = np.float32([for_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Find homography with improved RANSAC parameters
                    try:
                        H, mask = cv2.findHomography(
                            ref_pts, for_pts, 
                            method=cv2.RANSAC, 
                            ransacReprojThreshold=self.ransac_threshold,
                            maxIters=3000,  # More iterations for better results
                            confidence=0.98  # Higher confidence
                        )
                        
                        if H is not None:
                            # Calculate inlier ratio
                            inliers = np.sum(mask) if mask is not None else 0
                            inlier_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0
                            
                            # Enhanced score calculation
                            # Prioritize frames with high inlier ratios
                            match_score = inlier_ratio * len(good_matches) * (0.5 + inlier_ratio)
                            
                            # Additional verification using perspective transformation
                            perspective_score = self._check_perspective_quality(H)
                            
                            # Only accept reasonable transformations (lowered threshold)
                            if perspective_score > 0.3:
                                matches.append({
                                    'ref_timestamp': ref_ts,
                                    'foreign_timestamp': for_ts,
                                    'num_matches': len(good_matches),
                                    'inliers': inliers,
                                    'inlier_ratio': inlier_ratio,
                                    'score': match_score * perspective_score,
                                    'homography': H,
                                    'ref_keypoints': ref_kp,
                                    'for_keypoints': for_kp,
                                    'good_matches': good_matches,
                                    'mask': mask
                                })
                            else:
                                self.logger.debug(f"Frame pair {ref_ts}/{for_ts} rejected due to poor perspective (score={perspective_score:.2f})")
                        
                    except Exception as e:
                        self.logger.debug(f"Homography estimation failed for {ref_ts}/{for_ts}: {str(e)}")
                        # Don't add this match if homography failed
            
            except Exception as e:
                self.logger.debug(f"Feature matching failed for {ref_ts}/{for_ts}: {str(e)}")
# Sort matches by score (descending)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Found {len(matches)} potential matches")
        return matches
    
    def filter_matches_temporal(self, matches, max_outliers=0.5):  # Increased from 0.3
        """
        Filter matches with temporal consistency checking
    
        Parameters:
        matches (list): List of matched frame pairs
        max_outliers (float): Maximum ratio of outliers to tolerate
    
        Returns:
        list: Filtered list of matched frame pairs
        tuple: (global_offset, frame_rate_ratio)
        list: List of anchor points (ref_ts, for_ts)
        """
        if not matches:
            return [], (0, 1.0), []
    
        # Get top matches for initial estimation
        top_matches = matches[:min(40, len(matches))]  # Increased from 30
    
        # Extract timestamps
        timestamps = [(m['ref_timestamp'], m['foreign_timestamp']) for m in top_matches]
    
        # Sort by reference timestamp
        timestamps.sort(key=lambda x: x[0])
    
        # If we have enough points, try to filter outliers using RANSAC
        if len(timestamps) >= 3:  # Reduced from 5
            try:
                # Convert to numpy arrays for RANSAC
                ref_times = np.array([t[0] for t in timestamps])
                for_times = np.array([t[1] for t in timestamps])
            
                # Fit a line using RANSAC to handle outliers
                # Model: for_time = a * ref_time + b
                from sklearn.linear_model import RANSACRegressor
            
                ransac = RANSACRegressor(
                    random_state=42,
                    min_samples=3,  # Reduced from default
                    residual_threshold=5.0,  # More lenient residual threshold
                    max_trials=1000  # More trials for better fit
                )
                ransac.fit(ref_times.reshape(-1, 1), for_times)
            
                # Get inliers and their indices
                inlier_mask = ransac.inlier_mask_
                inliers = np.where(inlier_mask)[0]
            
                # Check if we have enough inliers
                if len(inliers) >= 3:  # Reduced from 4
                    # Extract inlier timestamps
                    inlier_timestamps = [timestamps[i] for i in inliers]
                
                    # Get the RANSAC model parameters
                    slope = ransac.estimator_.coef_[0]  # This is our frame_rate_ratio
                    intercept = ransac.estimator_.intercept_  # This is our global_offset
                
                    print(f"RANSAC fitted model: for_time = {slope:.4f} * ref_time + {intercept:.2f}")
                    print(f"Inliers: {len(inliers)}/{len(timestamps)} ({len(inliers)/len(timestamps)*100:.1f}%)")
                
                    # Filter the original matches using the inlier mask
                    filtered_matches = [top_matches[i] for i in inliers]
                
                    return filtered_matches, (intercept, slope), inlier_timestamps
        
            except Exception as e:
                print(f"RANSAC fitting failed: {e}")
                # Continue with the standard approach if RANSAC fails
    
        # Standard approach if RANSAC not applicable or failed
        # Check if timestamps exhibit a clear trend
        if len(timestamps) >= 2:
            # Fit a line to the timestamp pairs
            ref_times, for_times = zip(*timestamps)
        
            # Calculate correlation coefficient to check linearity
            corr = np.corrcoef(ref_times, for_times)[0, 1]
        
            if corr > 0.8:  # Reduced from 0.9 for more tolerance
                # Calculate slopes for each adjacent pair
                slopes = []
                for i in range(1, len(timestamps)):
                    delta_ref = timestamps[i][0] - timestamps[i-1][0]
                    delta_for = timestamps[i][1] - timestamps[i-1][1]
                    if delta_ref > 0:
                        slopes.append(delta_for / delta_ref)
            
                # Check if slopes are consistent
                if slopes:
                    median_slope = np.median(slopes)
                    frame_rate_ratio = median_slope
                
                    # If we deviate too much from the median slope, the matches might be incorrect
                    outliers = sum(abs(s - median_slope) / median_slope > 0.15 for s in slopes)  # Increased from 0.1
                    if outliers / len(slopes) <= max_outliers:
                        print(f"Consistent matches found! Frame rate ratio: {frame_rate_ratio:.4f}")
                    
                        # Calculate global offset
                        offsets = [for_ts - (ref_ts * frame_rate_ratio) for ref_ts, for_ts in timestamps]
                        global_offset = np.median(offsets)
                    
                        print(f"Estimated global offset: {global_offset:.2f} seconds")
                    
                        # Return filtered matches, frame rate info, and anchor points
                        return top_matches, (global_offset, frame_rate_ratio), timestamps
    
        # If we reach here, either too few matches or inconsistent ones
        # Use the top match as the only reliable point and assume ratio=1
        if matches:
            best_match = matches[0]
            ref_ts = best_match['ref_timestamp']
            for_ts = best_match['foreign_timestamp']
            global_offset = for_ts - ref_ts
            print(f"Using single best match. Offset: {global_offset:.2f}s, Ratio: 1.0")
            return [best_match], (global_offset, 1.0), [(ref_ts, for_ts)]
        else:
            print("No reliable matches found")
            return [], (0, 1.0), []
    
    def filter_matches(self, matches, max_outliers=0.5):  # Increased from 0.3
        """
        Filter matches to keep only consistent ones, with option for temporal consistency
        
        Parameters:
        matches (list): List of matched frame pairs
        max_outliers (float): Maximum ratio of outliers to tolerate
        
        Returns:
        list: Filtered list of matched frame pairs
        tuple: (global_offset, frame_rate_ratio)
        list: List of anchor points (ref_ts, for_ts)
        """
        # Use temporal consistency filtering if enabled
        if self.temporal_consistency:
            try:
                from sklearn.linear_model import RANSACRegressor
                return self.filter_matches_temporal(matches, max_outliers)
            except ImportError:
                print("Warning: sklearn not available. Falling back to standard filtering.")
                # Fall back to standard filtering
        
        # Standard filtering (from original implementation)
        if not matches:
            return [], (0, 1.0), []
            
        # Get top matches by score
        top_matches = matches[:min(20, len(matches))]  # Increased from 10
        
        # Extract timestamps
        timestamps = [(m['ref_timestamp'], m['foreign_timestamp']) for m in top_matches]
        
        # Sort by reference timestamp
        timestamps.sort(key=lambda x: x[0])
        
        # Check if timestamps exhibit a clear trend
        if len(timestamps) >= 2:
            # Fit a line to the timestamp pairs
            ref_times, for_times = zip(*timestamps)
            
            # Calculate correlation coefficient to check linearity
            corr = np.corrcoef(ref_times, for_times)[0, 1]
            
            if corr > 0.8:  # Reduced from 0.9 for more lenient matching
                # Calculate slopes for each adjacent pair
                slopes = []
                for i in range(1, len(timestamps)):
                    delta_ref = timestamps[i][0] - timestamps[i-1][0]
                    delta_for = timestamps[i][1] - timestamps[i-1][1]
                    if delta_ref > 0:
                        slopes.append(delta_for / delta_ref)
                
                # Check if slopes are consistent
                if slopes:
                    median_slope = np.median(slopes)
                    frame_rate_ratio = median_slope
                    
                    # More lenient deviation check
                    outliers = sum(abs(s - median_slope) / median_slope > 0.15 for s in slopes)  # Increased from 0.1
                    if outliers / len(slopes) <= max_outliers:
                        print(f"Consistent matches found! Frame rate ratio: {frame_rate_ratio:.4f}")
                        
                        # Calculate global offset
                        offsets = [for_ts - (ref_ts * frame_rate_ratio) for ref_ts, for_ts in timestamps]
                        global_offset = np.median(offsets)
                        
                        print(f"Estimated global offset: {global_offset:.2f} seconds")
                        
                        # Return filtered matches, frame rate info, and anchor points
                        return top_matches, (global_offset, frame_rate_ratio), timestamps
        
        # If we reach here, either too few matches or inconsistent ones
        # Use the top match as the only reliable point and assume ratio=1
        if matches:
            best_match = matches[0]
            ref_ts = best_match['ref_timestamp']
            for_ts = best_match['foreign_timestamp']
            global_offset = for_ts - ref_ts
            print(f"Using single best match. Offset: {global_offset:.2f}s, Ratio: 1.0")
            return [best_match], (global_offset, 1.0), [(ref_ts, for_ts)]
        else:
            print("No reliable matches found")
            return [], (0, 1.0), []
        
    def visualize_match(self, ref_frame, for_frame, ref_kp, for_kp, good_matches, match_info, ref_ts, for_ts, save_path=None):
        """
        Visualize a matched frame pair with enhanced information
        
        Parameters:
        ref_frame (numpy.ndarray): Reference frame
        for_frame (numpy.ndarray): Foreign frame
        ref_kp (list): Reference keypoints
        for_kp (list): Foreign keypoints
        good_matches (list): List of good matches
        match_info (dict): Match information
        ref_ts (float): Reference timestamp
        for_ts (float): Foreign timestamp
        save_path (str): Path to save visualization (optional)
        """
        # Draw matches between the frames
        if match_info.get('mask') is not None:
            # If we have a mask from homography, use it to separate inliers/outliers
            mask = match_info['mask'].ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),  # Green for inliers
                singlePointColor=(0, 0, 255),  # Blue for outliers
                matchesMask=mask,  # Highlight inliers
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
            )
            match_img = cv2.drawMatches(ref_frame, ref_kp, for_frame, for_kp, 
                                       good_matches, None, **draw_params)
        else:
            # Draw matches between the frames - simple version
            match_img = cv2.drawMatches(ref_frame, ref_kp, for_frame, for_kp, good_matches[:50], None,
                                       matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        
        # Add match information
        num_matches = match_info['num_matches']
        inliers = match_info.get('inliers', 0)
        score = match_info['score']
        
        # Add text with match info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Ref: {ref_ts:.2f}s, For: {for_ts:.2f}s"
        text2 = f"Matches: {num_matches}, Inliers: {inliers}, Score: {score:.1f}"
        cv2.putText(match_img, text, (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(match_img, text2, (10, 60), font, 0.8, (0, 255, 0), 2)
        
        # Save or display the visualization
        if save_path:
            cv2.imwrite(save_path, match_img)
            print(f"Match visualization saved to: {save_path}")
        else:
            cv2.imshow("Match Visualization", match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def visualize_timestamps(self, timestamps, frame_rate_ratio, global_offset, save_path=None):
        """
        Visualize the timestamp mapping with enhanced information
        
        Parameters:
        timestamps (list): List of matched timestamp pairs
        frame_rate_ratio (float): Frame rate ratio
        global_offset (float): Global offset
        save_path (str): Path to save visualization
        """
        # Extract timestamps
        ref_times, for_times = zip(*timestamps)
        
        # Create a range of reference times
        min_ref = min(ref_times)
        max_ref = max(ref_times)
        ref_range = np.linspace(min_ref, max_ref, 100)
        
        # Calculate corresponding foreign times using the frame rate ratio and offset
        for_range = (ref_range * frame_rate_ratio) + global_offset
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the matched timestamps
        plt.scatter(ref_times, for_times, c='r', marker='o', label='Matched Frames')
        
        # Plot the calculated mapping line
        plt.plot(ref_range, for_range, 'b-', linewidth=2, 
                label=f'Mapping (Ratio: {frame_rate_ratio:.4f}, Offset: {global_offset:.2f}s)')
        
        # Plot 1:1 reference line
        plt.plot([min_ref, max_ref], [min_ref, max_ref], 'g--', label='1:1 Reference')
        
        # Add confidence interval if multiple points
        if len(timestamps) > 2:
            # Calculate the mean absolute error between the model and actual points
            errors = [(for_time - ((ref_time * frame_rate_ratio) + global_offset)) 
                     for ref_time, for_time in timestamps]
            mae = np.mean(np.abs(errors))
            
            # Plot confidence interval (roughly 2 * MAE)
            plt.fill_between(
                ref_range, 
                for_range - 2*mae, 
                for_range + 2*mae, 
                color='b', alpha=0.1, 
                label=f'Confidence Interval (±{2*mae:.2f}s)'
            )
        
        # Add labels and title
        plt.xlabel('Reference Video Time (s)')
        plt.ylabel('Foreign Video Time (s)')
        plt.title('Video Time Mapping')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            print(f"Timestamp visualization saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def synchronize(self, ref_frames, foreign_frames, visualize=False, output_dir=None):
        """
        Synchronize videos based on visual features with enhanced quality
        
        Parameters:
        ref_frames (dict): Reference video frames
        foreign_frames (dict): Foreign video frames
        visualize (bool): Whether to create visualizations
        output_dir (str): Directory to save visualizations
        
        Returns:
        dict: Synchronization results
        """
        # Create output directory if needed
        if visualize and output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Compute features for all frames
        print("Computing features for reference frames...")
        ref_features = self.compute_features(ref_frames)
        
        print("Computing features for foreign frames...")
        foreign_features = self.compute_features(foreign_frames)
        
        # Match frames
        print("Matching frames...")
        all_matches = self.match_frames(ref_features, foreign_features)
        
        # Filter matches
        print("Filtering matches...")
        filtered_matches, time_mapping, anchor_points = self.filter_matches(all_matches)
        
        # Extract mapping parameters
        global_offset, frame_rate_ratio = time_mapping
        
        # Create visualizations if requested
        if visualize and output_dir:
            # Visualize best matches
            if filtered_matches:
                # Create a subdirectory for match visualizations if we have many
                matches_dir = os.path.join(output_dir, "match_visualizations")
                if len(filtered_matches) > 3:
                    os.makedirs(matches_dir, exist_ok=True)
                
                # Visualize top matches (up to 5)
                for i, match in enumerate(filtered_matches[:5]):
                    ref_ts = match['ref_timestamp']
                    for_ts = match['foreign_timestamp']
                    
                    # Get frames and features
                    if 'ref_keypoints' in match:  # Use stored keypoints if available
                        ref_kp = match['ref_keypoints']
                        for_kp = match['for_keypoints']
                        good_matches = match['good_matches']
                    else:
                        # Otherwise, get from features
                        ref_kp = ref_features[ref_ts]['keypoints']
                        for_kp = foreign_features[for_ts]['keypoints']
                        
                        # Get good matches (need to recalculate)
                        ref_desc = ref_features[ref_ts]['descriptors']
                        for_desc = foreign_features[for_ts]['descriptors']
                        good_matches = self._match_descriptors(ref_desc, for_desc)
                    
                    # Get original frames
                    if 'original' in ref_frames[ref_ts]:
                        ref_frame = ref_frames[ref_ts]['original']
                        for_frame = foreign_frames[for_ts]['original']
                    else:
                        # Fallback to grayscale frames
                        ref_frame = ref_frames[ref_ts]['gray']
                        for_frame = foreign_frames[for_ts]['gray']
                        # Convert to BGR for drawing color matches
                        if len(ref_frame.shape) == 2:  # If grayscale
                            ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_GRAY2BGR)
                            for_frame = cv2.cvtColor(for_frame, cv2.COLOR_GRAY2BGR)
                    
                    # Set visualization path
                    if len(filtered_matches) > 3:
                        match_viz_path = os.path.join(matches_dir, f"match_{i+1}.jpg")
                    else:
                        match_viz_path = os.path.join(output_dir, f"match_{i+1}.jpg")
                    
                    # Create visualization
                    self.visualize_match(ref_frame, for_frame, ref_kp, for_kp, good_matches, 
                                         match, ref_ts, for_ts, save_path=match_viz_path)
                
                # Also save the best match as "best_match.jpg" in the root output directory
                best_match = filtered_matches[0]
                ref_ts = best_match['ref_timestamp']
                for_ts = best_match['foreign_timestamp']
                
                # Get frames and features for best match
                if 'ref_keypoints' in best_match:
                    ref_kp = best_match['ref_keypoints']
                    for_kp = best_match['for_keypoints']
                    good_matches = best_match['good_matches']
                else:
                    ref_kp = ref_features[ref_ts]['keypoints']
                    for_kp = foreign_features[for_ts]['keypoints']
                    ref_desc = ref_features[ref_ts]['descriptors']
                    for_desc = foreign_features[for_ts]['descriptors']
                    good_matches = self._match_descriptors(ref_desc, for_desc)
                
                # Get original frames
                if 'original' in ref_frames[ref_ts]:
                    ref_frame = ref_frames[ref_ts]['original']
                    for_frame = foreign_frames[for_ts]['original']
                else:
                    ref_frame = ref_frames[ref_ts]['gray']
                    for_frame = foreign_frames[for_ts]['gray']
                    if len(ref_frame.shape) == 2:
                        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_GRAY2BGR)
                        for_frame = cv2.cvtColor(for_frame, cv2.COLOR_GRAY2BGR)
                
                best_match_viz_path = os.path.join(output_dir, "best_match.jpg")
                self.visualize_match(ref_frame, for_frame, ref_kp, for_kp, good_matches, 
                                     best_match, ref_ts, for_ts, save_path=best_match_viz_path)
            
            # Visualize timestamp mapping
            if anchor_points:
                ts_viz_path = os.path.join(output_dir, "time_mapping.png")
                self.visualize_timestamps(anchor_points, frame_rate_ratio, global_offset, save_path=ts_viz_path)
        
        # Create result dictionary
        sync_info = {
            "global_offset": global_offset,
            "frame_rate_ratio": frame_rate_ratio,
            "match_count": len(filtered_matches),
            "anchor_points": anchor_points
        }
        
        return sync_info
