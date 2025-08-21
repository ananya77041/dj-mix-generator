#!/usr/bin/env python3
"""
Beat alignment utilities for precise DJ mixing
"""

import numpy as np
import librosa
from typing import Tuple, List
try:
    from .models import Track
except ImportError:
    # Fallback for direct execution - use full path to avoid conflicts
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from core.models import Track


class BeatAligner:
    """Handles precise beat and downbeat alignment for seamless transitions"""
    
    def __init__(self, interactive_beats: bool = False):
        self.interactive_beats = interactive_beats
    
    def find_optimal_transition_points(self, track1: Track, track2: Track, 
                                     transition_duration: float) -> Tuple[int, int]:
        """
        Find optimal transition points where track1 ends and track2 begins
        Returns (track1_end_sample, track2_start_sample) with perfect downbeat alignment
        """
        # Convert transition duration to samples
        transition_samples = int(transition_duration * track1.sr)
        
        # Find the best downbeat in track1 to end the transition
        track1_end_downbeat = self._find_outro_downbeat(track1, transition_samples)
        track1_end_sample = librosa.frames_to_samples(track1_end_downbeat, hop_length=512)
        
        # Find the first downbeat in track2 to bring it in
        track2_start_downbeat = self._find_intro_downbeat(track2)
        track2_start_sample = librosa.frames_to_samples(track2_start_downbeat, hop_length=512)
        
        # CRITICAL: Ensure perfect downbeat alignment in the transition
        # The end of track1's outro must align exactly with start of track2's intro
        aligned_start_sample = self._align_downbeats_for_transition(
            track1, track2, track1_end_sample, track2_start_sample, transition_samples
        )
        
        return track1_end_sample, aligned_start_sample
    
    def _align_downbeats_for_transition(self, track1: Track, track2: Track, 
                                      track1_end_sample: int, track2_start_sample: int, 
                                      transition_samples: int) -> int:
        """
        Ensure perfect downbeat alignment between tracks during transition.
        Both tracks' downbeats must hit simultaneously throughout the transition.
        """
        if len(track1.downbeats) == 0 or len(track2.downbeats) == 0:
            print("  Warning: Missing downbeats, using original alignment")
            return track2_start_sample
        
        # Find track1's downbeats that occur during the transition
        track1_outro_start = track1_end_sample - transition_samples
        track1_downbeats_samples = librosa.frames_to_samples(track1.downbeats, hop_length=512)
        
        # Get track1's downbeats within the transition window
        track1_transition_downbeats = track1_downbeats_samples[
            (track1_downbeats_samples >= track1_outro_start) & 
            (track1_downbeats_samples <= track1_end_sample)
        ]
        
        if len(track1_transition_downbeats) == 0:
            print("  Warning: No track1 downbeats in transition window, using original alignment")
            return track2_start_sample
        
        # Find track2's first suitable downbeat
        min_intro_time = 2.0  # Skip first 2 seconds
        min_intro_frame = librosa.time_to_frames(min_intro_time, sr=track2.sr, hop_length=512)
        
        suitable_downbeats = track2.downbeats[track2.downbeats >= min_intro_frame]
        if len(suitable_downbeats) == 0:
            suitable_downbeats = track2.downbeats  # Use any downbeat if none after 2s
        
        first_downbeat_frame = suitable_downbeats[0]
        first_downbeat_sample = librosa.frames_to_samples(first_downbeat_frame, hop_length=512)
        
        # Choose a track1 downbeat to align with (prefer one in the middle of transition)
        target_track1_downbeat = track1_transition_downbeats[len(track1_transition_downbeats)//2]
        
        # Calculate track2 start position so its downbeat coincides with track1's downbeat
        # We want: track2_start + first_downbeat_sample = target_track1_downbeat
        # Therefore: track2_start = target_track1_downbeat - first_downbeat_sample
        optimal_track2_start = target_track1_downbeat - first_downbeat_sample
        
        # Ensure valid position
        optimal_track2_start = max(0, optimal_track2_start)
        optimal_track2_start = min(optimal_track2_start, len(track2.audio) - transition_samples)
        
        # Calculate the timing adjustment
        adjustment_seconds = (optimal_track2_start - track2_start_sample) / track2.sr
        target_time = target_track1_downbeat / track1.sr
        
        print(f"  Simultaneous downbeat alignment:")
        print(f"    Track1 downbeat at: {target_time:.3f}s")
        print(f"    Track2 adjusted by: {adjustment_seconds:.3f}s")
        print(f"    Downbeats will hit simultaneously during transition")
        
        return optimal_track2_start
    
    def _find_outro_downbeat(self, track: Track, transition_samples: int) -> int:
        """
        Find the optimal downbeat in track1 to end the transition
        Should be late enough in the track but allow for the full transition
        """
        if len(track.downbeats) == 0:
            # Fallback to regular beats if no downbeats detected
            if len(track.beats) > 0:
                return track.beats[-4] if len(track.beats) >= 4 else track.beats[-1]
            else:
                return len(track.audio) - transition_samples
        
        # Find downbeats that are late enough for a good outro
        # but leave room for the transition
        min_outro_position = len(track.audio) * 0.7  # Start looking after 70% of track
        max_outro_position = len(track.audio) - transition_samples * 1.1  # Leave room for transition
        
        # Convert to frame numbers for comparison with downbeats
        min_outro_frame = librosa.samples_to_frames(min_outro_position, hop_length=512)
        max_outro_frame = librosa.samples_to_frames(max_outro_position, hop_length=512)
        
        # Find suitable downbeats
        suitable_downbeats = track.downbeats[
            (track.downbeats >= min_outro_frame) & 
            (track.downbeats <= max_outro_frame)
        ]
        
        if len(suitable_downbeats) > 0:
            # Choose the last suitable downbeat for maximum track play
            return suitable_downbeats[-1]
        else:
            # Fallback: use the last downbeat if it's reasonable
            if len(track.downbeats) > 0 and track.downbeats[-1] <= max_outro_frame:
                return track.downbeats[-1]
            else:
                # Ultimate fallback: calculate a position manually
                return librosa.samples_to_frames(max_outro_position, hop_length=512)
    
    def _find_intro_downbeat(self, track: Track) -> int:
        """
        Find the first good downbeat in track2 to bring it in
        Should skip any intro/silence but catch the first strong downbeat
        """
        if len(track.downbeats) == 0:
            # Fallback to first beat
            return track.beats[0] if len(track.beats) > 0 else 0
        
        # Skip very early downbeats that might be in silence/intro
        min_intro_time = 2.0  # Skip first 2 seconds
        min_intro_frame = librosa.time_to_frames(min_intro_time, sr=track.sr, hop_length=512)
        
        # Find the first downbeat after the intro period
        suitable_downbeats = track.downbeats[track.downbeats >= min_intro_frame]
        
        if len(suitable_downbeats) > 0:
            return suitable_downbeats[0]
        else:
            # Fallback: use first downbeat even if it's early
            return track.downbeats[0] if len(track.downbeats) > 0 else 0
    
    def align_beats_for_transition(self, track1: Track, track2: Track, 
                                 track1_end_sample: int, track2_start_sample: int,
                                 transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and align audio segments for perfect beat-matched transition
        Returns (track1_outro, track2_intro) with precise beat-by-beat alignment
        """
        transition_samples = int(transition_duration * track1.sr)
        
        # Extract outro from track1 (leading up to the downbeat)
        outro_start = max(0, track1_end_sample - transition_samples)
        track1_outro = track1.audio[outro_start:track1_end_sample]
        
        # Extract intro from track2 (starting from the aligned position)
        intro_end = min(len(track2.audio), track2_start_sample + transition_samples)
        track2_intro = track2.audio[track2_start_sample:intro_end]
        
        # Ensure both segments are the same length for crossfade
        min_length = min(len(track1_outro), len(track2_intro))
        if min_length < transition_samples * 0.5:  # If segments are too short
            print(f"  Warning: Short transition segments ({min_length/track1.sr:.1f}s)")
            print(f"  Track1 outro: {len(track1_outro)/track1.sr:.1f}s, Track2 intro: {len(track2_intro)/track2.sr:.1f}s")
        
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Apply perfect beat-by-beat alignment throughout the transition
        track1_outro_aligned, track2_intro_aligned = self._apply_beat_by_beat_alignment(
            track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample, transition_duration
        )
        
        # Verify alignment quality
        self._verify_perfect_alignment(track1, track2, track1_end_sample, track2_start_sample, len(track1_outro_aligned))
        
        return track1_outro_aligned, track2_intro_aligned
    
    def _apply_beat_by_beat_alignment(self, track1: Track, track2: Track, 
                                    track1_outro: np.ndarray, track2_intro: np.ndarray,
                                    outro_start: int, track2_start_sample: int, transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perfect beat-by-beat alignment throughout the transition period.
        Uses micro-stretching and phase alignment to ensure every beat matches.
        """
        try:
            # Check if interactive alignment is requested via environment variable or manual mode
            use_interactive_alignment = self._should_use_interactive_alignment()
            
            if use_interactive_alignment:
                return self._apply_interactive_alignment(
                    track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample, transition_duration
                )
            else:
                return self._apply_automatic_alignment(
                    track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample
                )
                
        except Exception as e:
            print(f"  Warning: Beat-by-beat alignment failed: {e}")
            print("  Using original audio segments")
            return track1_outro, track2_intro
    
    def _should_use_interactive_alignment(self) -> bool:
        """Determine if interactive beatgrid alignment should be used"""
        return self.interactive_beats
    
    def _apply_interactive_alignment(self, track1: Track, track2: Track, 
                                   track1_outro: np.ndarray, track2_intro: np.ndarray,
                                   outro_start: int, track2_start_sample: int, 
                                   transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply interactive beatgrid alignment using GUI"""
        try:
            try:
                from ..gui.beatgrid.simple_gui import align_beatgrids_interactive
            except ImportError:
                from gui.beatgrid.simple_gui import align_beatgrids_interactive
            
            print("  Opening interactive beatgrid alignment...")
            print(f"  Track 1 BPM: {track1.bpm:.1f}")
            print(f"  Track 2 BPM: {track2.bpm:.1f}")
            bpm_diff = abs(track1.bpm - track2.bpm)
            if bpm_diff < 0.1:
                print("  ✓ Tracks are tempo-matched for optimal user experience")
            else:
                print(f"  ⚠ Warning: Tracks have different tempos (diff: {bpm_diff:.1f} BPM)")
                print("  This should not happen - tracks should be tempo-matched before interactive alignment")
            
            # Get user's alignment offset
            offset = align_beatgrids_interactive(
                track1, track2, track1_outro, track2_intro,
                outro_start, track2_start_sample, transition_duration
            )
            
            if offset != 0.0:
                print(f"  Applying user alignment: {offset:.3f}s offset")
                
                # Apply the offset to track2_intro by time-shifting
                track2_intro_aligned = self._apply_time_offset(track2_intro, offset, track1.sr)
                
                # Ensure both tracks are the same length
                min_length = min(len(track1_outro), len(track2_intro_aligned))
                return track1_outro[:min_length], track2_intro_aligned[:min_length]
            else:
                print("  No offset applied, using automatic alignment")
                return self._apply_automatic_alignment(
                    track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample
                )
                
        except ImportError:
            print("  Interactive alignment not available, using automatic alignment")
            return self._apply_automatic_alignment(
                track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample
            )
        except Exception as e:
            print(f"  Interactive alignment failed: {e}")
            print("  Falling back to automatic alignment")
            return self._apply_automatic_alignment(
                track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample
            )
    
    def _apply_time_offset(self, audio: np.ndarray, offset_seconds: float, sr: int) -> np.ndarray:
        """Apply a time offset to audio by shifting and padding/cropping"""
        offset_samples = int(offset_seconds * sr)
        
        if offset_samples == 0:
            return audio
        elif offset_samples > 0:
            # Positive offset: delay the audio (pad beginning, crop end)
            padding = np.zeros(offset_samples)
            shifted_audio = np.concatenate([padding, audio])
            return shifted_audio[:len(audio)]  # Keep original length
        else:
            # Negative offset: advance the audio (crop beginning, pad end)
            offset_samples = abs(offset_samples)
            if offset_samples < len(audio):
                cropped_audio = audio[offset_samples:]
                padding = np.zeros(offset_samples)
                return np.concatenate([cropped_audio, padding])
            else:
                # Offset larger than audio length, return silence
                return np.zeros_like(audio)
    
    def _apply_automatic_alignment(self, track1: Track, track2: Track, 
                                 track1_outro: np.ndarray, track2_intro: np.ndarray,
                                 outro_start: int, track2_start_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply automatic beat-by-beat alignment (original method)"""
        try:
            print("  Applying automatic beat-by-beat alignment...")
            
            # Convert beat positions to samples relative to the segments
            track1_downbeats_samples = librosa.frames_to_samples(track1.downbeats, hop_length=512)
            track2_downbeats_samples = librosa.frames_to_samples(track2.downbeats, hop_length=512)
            track1_beats_samples = librosa.frames_to_samples(track1.beats, hop_length=512)
            track2_beats_samples = librosa.frames_to_samples(track2.beats, hop_length=512)
            
            # Find beats within the transition segments
            track1_segment_beats = track1_beats_samples[
                (track1_beats_samples >= outro_start) & 
                (track1_beats_samples <= outro_start + len(track1_outro))
            ] - outro_start  # Make relative to segment start
            
            track2_segment_beats = track2_beats_samples[
                (track2_beats_samples >= track2_start_sample) & 
                (track2_beats_samples <= track2_start_sample + len(track2_intro))
            ] - track2_start_sample  # Make relative to segment start
            
            if len(track1_segment_beats) == 0 or len(track2_segment_beats) == 0:
                print("    Warning: No beats found in transition segments, using original audio")
                return track1_outro, track2_intro
            
            # Create ideal beat grid based on track1's tempo (the reference)
            beats_per_second = track1.bpm / 60.0
            samples_per_beat = track1.sr / beats_per_second
            
            # Generate ideal beat positions for the transition duration
            ideal_beat_positions = []
            
            # Start from the first detected beat position for phase consistency
            if len(track1_segment_beats) > 0:
                first_beat_offset = track1_segment_beats[0]
                current_beat_pos = first_beat_offset
                
                while current_beat_pos < len(track1_outro):
                    ideal_beat_positions.append(current_beat_pos)
                    current_beat_pos += samples_per_beat
            
            ideal_beat_positions = np.array(ideal_beat_positions)
            
            if len(ideal_beat_positions) == 0:
                print("    Warning: No ideal beats generated, using original audio")
                return track1_outro, track2_intro
            
            print(f"    Track1 beats in segment: {len(track1_segment_beats)}")
            print(f"    Track2 beats in segment: {len(track2_segment_beats)}")
            print(f"    Ideal beat grid: {len(ideal_beat_positions)} beats")
            
            # Apply intelligent beat alignment to both tracks for perfect synchronization
            print("    Applying intelligent beat shifting for perfect alignment...")
            track1_outro_aligned, track2_intro_aligned = self._apply_intelligent_beat_shifting(
                track1_outro, track2_intro, track1_segment_beats, track2_segment_beats, 
                ideal_beat_positions, track1.sr
            )
            
            # Ensure both tracks are exactly the same length
            min_length = min(len(track1_outro_aligned), len(track2_intro_aligned))
            track1_outro_final = track1_outro_aligned[:min_length]
            track2_intro_final = track2_intro_aligned[:min_length]
            
            print(f"    Final alignment: {min_length} samples ({min_length/track1.sr:.3f}s)")
            
            return track1_outro_final, track2_intro_final
            
        except Exception as e:
            print(f"    Warning: Automatic alignment failed: {e}")
            print("    Using original audio segments")
            return track1_outro, track2_intro
    
    def _apply_intelligent_beat_shifting(self, track1_outro: np.ndarray, track2_intro: np.ndarray,
                                        track1_beats: np.ndarray, track2_beats: np.ndarray, 
                                        ideal_beats: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligently shift beats in both tracks to achieve perfect alignment while preserving 
        total transition length and maintaining continuity at both edges.
        
        Enhanced with artifact prevention and conservative adjustments.
        """
        try:
            print("      Calculating optimal beat alignment strategy...")
            
            # Check if alignment is really necessary
            if len(track1_beats) == 0 and len(track2_beats) == 0:
                print("      No beats detected, using original audio")
                return track1_outro, track2_intro
            
            # Calculate alignment necessity - skip if tracks are already well aligned
            if len(track1_beats) > 0 and len(track2_beats) > 0:
                # Sample a few beat positions to check alignment
                sample_size = min(3, len(track1_beats), len(track2_beats))
                total_offset = 0
                valid_comparisons = 0
                
                for i in range(sample_size):
                    if i < len(track1_beats) and i < len(track2_beats):
                        offset = abs(track1_beats[i] - track2_beats[i])
                        if offset < sr * 0.5:  # Within 0.5 seconds
                            total_offset += offset
                            valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_offset = total_offset / valid_comparisons
                    offset_ms = (avg_offset / sr) * 1000
                    
                    if offset_ms < 50:  # Less than 50ms average offset
                        print(f"      Tracks already well-aligned ({offset_ms:.1f}ms avg offset), skipping beat shifting")
                        return track1_outro, track2_intro
            
            # Find common beat grid by analyzing both tracks
            target_beats = self._calculate_optimal_beat_grid(
                track1_beats, track2_beats, ideal_beats, len(track1_outro), sr
            )
            
            if len(target_beats) < 2:
                print("      Insufficient beats for intelligent alignment, using original audio")
                return track1_outro, track2_intro
            
            print(f"      Target beat grid: {len(target_beats)} beats")
            
            # Apply intelligent beat shifting with conservative approach
            track1_aligned = self._apply_intelligent_stretch(
                track1_outro, track1_beats, target_beats, sr, "Track1"
            )
            
            track2_aligned = self._apply_intelligent_stretch(
                track2_intro, track2_beats, target_beats, sr, "Track2"
            )
            
            # Quality check - if too much change, revert to original
            track1_change = abs(len(track1_aligned) - len(track1_outro)) / len(track1_outro)
            track2_change = abs(len(track2_aligned) - len(track2_intro)) / len(track2_intro)
            
            if track1_change > 0.02 or track2_change > 0.02:  # More than 2% length change
                print(f"      Beat alignment caused excessive length change ({track1_change:.1%}, {track2_change:.1%})")
                print("      Reverting to original audio for quality preservation")
                return track1_outro, track2_intro
            
            # Ensure both tracks maintain exact original length (critical for mix continuity)
            track1_aligned = self._preserve_segment_length(track1_aligned, len(track1_outro))
            track2_aligned = self._preserve_segment_length(track2_aligned, len(track2_intro))
            
            print(f"      Intelligent alignment complete: {len(track1_aligned)} samples")
            return track1_aligned, track2_aligned
            
        except Exception as e:
            print(f"      Warning: Intelligent beat shifting failed: {e}")
            print("      Using original audio segments")
            return track1_outro, track2_intro
    
    def _calculate_optimal_beat_grid(self, track1_beats: np.ndarray, track2_beats: np.ndarray, 
                                   ideal_beats: np.ndarray, segment_length: int, sr: int) -> np.ndarray:
        """Calculate optimal beat positions that minimize adjustments for both tracks"""
        try:
            # Start with ideal beat grid as base
            target_beats = []
            
            for ideal_beat in ideal_beats:
                if ideal_beat >= segment_length:
                    break
                    
                # Find closest beats in both tracks
                track1_distances = np.abs(track1_beats - ideal_beat) if len(track1_beats) > 0 else [float('inf')]
                track2_distances = np.abs(track2_beats - ideal_beat) if len(track2_beats) > 0 else [float('inf')]
                
                track1_closest_idx = np.argmin(track1_distances) if len(track1_beats) > 0 else None
                track2_closest_idx = np.argmin(track2_distances) if len(track2_beats) > 0 else None
                
                # Calculate compromise position (weighted average favoring closer beats)
                positions = [ideal_beat]  # Always include ideal as fallback
                
                if track1_closest_idx is not None and track1_distances[track1_closest_idx] < sr * 0.3:
                    positions.append(track1_beats[track1_closest_idx])
                    
                if track2_closest_idx is not None and track2_distances[track2_closest_idx] < sr * 0.3:
                    positions.append(track2_beats[track2_closest_idx])
                
                # Use weighted average with preference for actual beats over ideal
                if len(positions) == 1:
                    target_beat = positions[0]
                else:
                    # Weight actual beats more than ideal, average of actual beats
                    actual_beats = positions[1:] if len(positions) > 1 else []
                    if actual_beats:
                        target_beat = np.mean(actual_beats)
                    else:
                        target_beat = ideal_beat
                
                target_beats.append(target_beat)
            
            return np.array(target_beats)
            
        except Exception as e:
            print(f"        Warning: Beat grid calculation failed: {e}")
            return ideal_beats[:len(ideal_beats)//2] if len(ideal_beats) > 2 else np.array([])
    
    def _apply_intelligent_stretch(self, audio: np.ndarray, detected_beats: np.ndarray, 
                                 target_beats: np.ndarray, sr: int, track_name: str) -> np.ndarray:
        """Apply minimal stretching to align detected beats with target beats"""
        try:
            if len(detected_beats) == 0 or len(target_beats) == 0:
                return audio
            
            # Match detected beats to target beats (find best alignment)
            beat_pairs = []
            used_targets = set()
            
            for detected_beat in detected_beats:
                if detected_beat >= len(audio):
                    continue
                    
                # Find closest unused target beat
                available_targets = [i for i in range(len(target_beats)) if i not in used_targets]
                if not available_targets:
                    break
                    
                distances = [abs(target_beats[i] - detected_beat) for i in available_targets]
                closest_target_idx = available_targets[np.argmin(distances)]
                
                if distances[np.argmin(distances)] < sr * 0.4:  # Within 0.4 seconds
                    beat_pairs.append((detected_beat, target_beats[closest_target_idx]))
                    used_targets.add(closest_target_idx)
            
            if len(beat_pairs) < 2:
                print(f"        {track_name}: Insufficient beat pairs, using original audio")
                return audio
                
            beat_pairs.sort(key=lambda x: x[0])  # Sort by detected beat position
            print(f"        {track_name}: Aligning {len(beat_pairs)} beat pairs")
            
            # Apply piecewise intelligent stretching
            return self._piecewise_intelligent_stretch(audio, beat_pairs, sr)
            
        except Exception as e:
            print(f"        Warning: {track_name} stretch failed: {e}")
            return audio
    
    def _piecewise_intelligent_stretch(self, audio: np.ndarray, beat_pairs: List[Tuple[float, float]], sr: int) -> np.ndarray:
        """Apply piecewise stretching with smooth transitions and artifact prevention"""
        try:
            # Use larger segments to reduce artifacts and improve quality
            if len(beat_pairs) < 2:
                print("          Insufficient beat pairs for smooth stretching, using original audio")
                return audio
            
            # Group beat pairs to create larger segments (reduces artifacts)
            segment_size = max(2, len(beat_pairs) // 8)  # Create ~8 segments max
            grouped_pairs = []
            
            for i in range(0, len(beat_pairs), segment_size):
                group = beat_pairs[i:i + segment_size]
                if group:
                    # Use first and last pair in group for segment boundaries
                    start_pair = group[0]
                    end_pair = group[-1]
                    grouped_pairs.append((start_pair[0], start_pair[1], end_pair[0], end_pair[1]))
            
            stretched_segments = []
            overlap_samples = int(0.02 * sr)  # 20ms overlap for smooth transitions
            
            # Process each grouped segment
            for i, (seg_start_orig, seg_start_target, seg_end_orig, seg_end_target) in enumerate(grouped_pairs):
                
                # Extract segment with overlap
                segment_start = max(0, int(seg_start_orig) - (overlap_samples if i > 0 else 0))
                segment_end = min(len(audio), int(seg_end_orig) + (overlap_samples if i < len(grouped_pairs)-1 else 0))
                
                if segment_end <= segment_start:
                    continue
                    
                segment = audio[segment_start:segment_end]
                
                # Calculate target length (with overlap compensation)
                target_start = int(seg_start_target) - (overlap_samples if i > 0 else 0)
                target_end = int(seg_end_target) + (overlap_samples if i < len(grouped_pairs)-1 else 0)
                target_length = target_end - target_start
                
                if target_length <= 0:
                    continue
                
                # Calculate stretch ratio with safety limits
                stretch_ratio = len(segment) / target_length
                
                # Apply stretching with conservative limits to prevent artifacts
                if 0.92 <= stretch_ratio <= 1.08 and abs(len(segment) - target_length) > sr * 0.025:  # >25ms difference
                    # Use high-quality stretching for minimal artifacts
                    try:
                        stretched_segment = librosa.effects.time_stretch(
                            segment, 
                            rate=stretch_ratio,
                            # Use higher hop length for better quality
                            hop_length=max(512, len(segment) // 64)
                        )
                        
                        # Fine-tune length with minimal padding/cropping
                        length_diff = len(stretched_segment) - target_length
                        if abs(length_diff) > 0:
                            if length_diff > 0:
                                # Crop from edges symmetrically to maintain phase
                                crop_each_side = length_diff // 2
                                stretched_segment = stretched_segment[crop_each_side:crop_each_side + target_length]
                            else:
                                # Pad with fade-in/out silence
                                padding_needed = -length_diff
                                pad_each_side = padding_needed // 2
                                left_pad = np.zeros(pad_each_side)
                                right_pad = np.zeros(padding_needed - pad_each_side)
                                stretched_segment = np.concatenate([left_pad, stretched_segment, right_pad])
                                
                    except Exception as stretch_error:
                        print(f"          Stretch failed for segment {i}, using original: {stretch_error}")
                        stretched_segment = segment
                        # Adjust to target length with less aggressive methods
                        if len(segment) != target_length:
                            if len(segment) > target_length:
                                # Crop from center
                                excess = len(segment) - target_length
                                start_crop = excess // 2
                                stretched_segment = segment[start_crop:start_crop + target_length]
                            else:
                                # Repeat end samples to extend
                                needed = target_length - len(segment)
                                if len(segment) > 0:
                                    extension = np.tile(segment[-min(1024, len(segment)):], 
                                                      (needed // min(1024, len(segment)) + 1))[:needed]
                                    stretched_segment = np.concatenate([segment, extension])
                                else:
                                    stretched_segment = np.zeros(target_length)
                else:
                    # No stretching needed or ratio too extreme - use original with length adjustment
                    stretched_segment = segment
                    if len(segment) != target_length:
                        if len(segment) > target_length:
                            # Crop from center to preserve important content
                            excess = len(segment) - target_length
                            start_crop = excess // 2
                            stretched_segment = segment[start_crop:start_crop + target_length]
                        else:
                            # Extend with fade-out of original content
                            needed = target_length - len(segment)
                            if len(segment) > 0:
                                # Use last 10% of segment for extension
                                extend_source = segment[int(len(segment) * 0.9):]
                                extension = np.tile(extend_source, (needed // len(extend_source) + 1))[:needed]
                                # Apply fade to extension to prevent clicks
                                fade_length = min(len(extension), int(0.005 * sr))  # 5ms fade
                                extension[:fade_length] *= np.linspace(0.8, 0.2, fade_length)
                                stretched_segment = np.concatenate([segment, extension])
                            else:
                                stretched_segment = np.zeros(target_length)
                
                # Apply overlap-add for smooth transitions between segments
                if i > 0 and len(stretched_segments) > 0 and overlap_samples > 0:
                    # Cross-fade with previous segment
                    prev_segment = stretched_segments[-1]
                    if len(prev_segment) >= overlap_samples and len(stretched_segment) >= overlap_samples:
                        # Extract overlapping regions
                        prev_overlap = prev_segment[-overlap_samples:]
                        curr_overlap = stretched_segment[:overlap_samples]
                        
                        # Create crossfade
                        fade_out = np.linspace(1.0, 0.0, overlap_samples)
                        fade_in = np.linspace(0.0, 1.0, overlap_samples)
                        
                        # Apply crossfade
                        mixed_overlap = prev_overlap * fade_out + curr_overlap * fade_in
                        
                        # Update segments
                        stretched_segments[-1] = prev_segment[:-overlap_samples]
                        stretched_segment = np.concatenate([mixed_overlap, stretched_segment[overlap_samples:]])
                
                stretched_segments.append(stretched_segment)
            
            # Handle final segment (from last beat to end) if needed
            if len(beat_pairs) > 0:
                final_segment_start = int(beat_pairs[-1][0])
                if final_segment_start < len(audio):
                    final_segment = audio[final_segment_start:]
                    if len(final_segment) > 0:
                        stretched_segments.append(final_segment)
            
            # Concatenate all segments
            if stretched_segments:
                result = np.concatenate(stretched_segments)
                print(f"          Smooth piecewise stretch complete: {len(audio)} -> {len(result)} samples")
                return result
            else:
                return audio
                
        except Exception as e:
            print(f"          Warning: Smooth piecewise stretch failed: {e}")
            print(f"          Falling back to original audio")
            return audio
    
    def _preserve_segment_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Ensure audio segment maintains exact target length for mix continuity"""
        if len(audio) == target_length:
            return audio
        elif len(audio) > target_length:
            # Crop to exact length
            return audio[:target_length]
        else:
            # Pad with zeros to exact length
            padding = np.zeros(target_length - len(audio))
            return np.concatenate([audio, padding])

    def _micro_stretch_to_beat_grid(self, audio: np.ndarray, detected_beats: np.ndarray, 
                                  ideal_beats: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply micro-stretching to align detected beats with ideal beat grid.
        Uses piecewise time-stretching between beat markers.
        """
        if len(detected_beats) == 0 or len(ideal_beats) == 0:
            return audio
        
        # Match detected beats to ideal beats (find closest pairs)
        beat_pairs = []
        for ideal_beat in ideal_beats:
            if len(detected_beats) > 0:
                distances = np.abs(detected_beats - ideal_beat)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < sr * 0.5:  # Within 0.5 seconds
                    beat_pairs.append((detected_beats[closest_idx], ideal_beat))
        
        if len(beat_pairs) < 2:
            print("    Not enough beat pairs for micro-stretching, using original audio")
            return audio
        
        beat_pairs.sort(key=lambda x: x[0])  # Sort by detected beat position
        
        # Apply piecewise stretching between beat pairs
        stretched_segments = []
        
        # Process audio in segments between beats
        for i in range(len(beat_pairs)):
            detected_pos, ideal_pos = beat_pairs[i]
            
            if i == 0:
                # First segment: from start to first beat
                segment_start = 0
                segment_end = int(detected_pos)
                target_start = 0
                target_end = int(ideal_pos)
            else:
                # Segments between beats
                prev_detected, prev_ideal = beat_pairs[i-1]
                segment_start = int(prev_detected)
                segment_end = int(detected_pos)
                target_start = int(prev_ideal)
                target_end = int(ideal_pos)
            
            # Extract segment
            if segment_end > segment_start and segment_end <= len(audio):
                segment = audio[segment_start:segment_end]
                
                # Calculate stretch ratio
                original_length = segment_end - segment_start
                target_length = target_end - target_start
                
                if original_length > 0 and target_length > 0:
                    stretch_ratio = target_length / original_length
                    
                    # Only stretch if the ratio is reasonable (between 0.8 and 1.25)
                    if 0.8 <= stretch_ratio <= 1.25:
                        try:
                            stretched_segment = librosa.effects.time_stretch(segment, rate=1/stretch_ratio)
                            # Ensure target length
                            if len(stretched_segment) > target_length:
                                stretched_segment = stretched_segment[:target_length]
                            elif len(stretched_segment) < target_length:
                                # Pad with silence if too short
                                padding = target_length - len(stretched_segment)
                                stretched_segment = np.pad(stretched_segment, (0, padding), 'constant')
                        except:
                            # Fallback: just resize
                            stretched_segment = np.resize(segment, target_length)
                    else:
                        # Stretch ratio too extreme, just resize
                        stretched_segment = np.resize(segment, target_length)
                    
                    stretched_segments.append(stretched_segment)
        
        # Handle final segment after last beat
        if len(beat_pairs) > 0:
            last_detected, last_ideal = beat_pairs[-1]
            if int(last_detected) < len(audio):
                final_segment = audio[int(last_detected):]
                stretched_segments.append(final_segment)
        
        # Combine all stretched segments
        if stretched_segments:
            aligned_audio = np.concatenate(stretched_segments)
            print(f"    Micro-stretching applied: {len(beat_pairs)} beat alignment points")
            return aligned_audio
        else:
            return audio
    
    def _verify_perfect_alignment(self, track1: Track, track2: Track, 
                                track1_end_sample: int, track2_start_sample: int, 
                                aligned_length: int):
        """Enhanced verification for perfect beat-by-beat alignment"""
        try:
            print("  Verifying perfect beat alignment...")
            
            # This is a more thorough verification than the previous method
            outro_start = max(0, track1_end_sample - aligned_length)
            
            # Find all beats during the transition
            track1_beats_samples = librosa.frames_to_samples(track1.beats, hop_length=512)
            track2_beats_samples = librosa.frames_to_samples(track2.beats, hop_length=512)
            
            # Beats in track1's outro section
            track1_transition_beats = track1_beats_samples[
                (track1_beats_samples >= outro_start) & 
                (track1_beats_samples <= track1_end_sample)
            ]
            
            # Beats in track2's intro section (adjusted for start position)
            track2_absolute_beats = track2_beats_samples + track2_start_sample
            track2_transition_beats = track2_absolute_beats[
                (track2_absolute_beats >= outro_start) & 
                (track2_absolute_beats <= outro_start + aligned_length)
            ]
            
            if len(track1_transition_beats) > 0 and len(track2_transition_beats) > 0:
                # Calculate beat alignment accuracy
                total_offset = 0
                matches = 0
                
                for t1_beat in track1_transition_beats:
                    # Find closest t2 beat
                    distances = np.abs(track2_transition_beats - t1_beat)
                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        total_offset += min_distance / track1.sr * 1000  # Convert to ms
                        matches += 1
                
                if matches > 0:
                    avg_offset = total_offset / matches
                    print(f"    Beat alignment quality: {avg_offset:.1f}ms average offset")
                    print(f"    Aligned beats: {matches} pairs")
                    
                    if avg_offset < 5:
                        print("    ✓ Excellent beat alignment achieved")
                    elif avg_offset < 15:
                        print("    ✓ Good beat alignment achieved")
                    else:
                        print("    ⚠ Beat alignment could be improved")
                else:
                    print("    ⚠ No beat pairs found for verification")
            else:
                print("    ⚠ Insufficient beats for alignment verification")
                
        except Exception as e:
            print(f"    Warning: Alignment verification failed: {e}")
    
    def _verify_downbeat_alignment(self, track1: Track, track2: Track, 
                                 track1_end_sample: int, track2_start_sample: int, 
                                 transition_length: int):
        """Verify and report simultaneous downbeat alignment quality for debugging"""
        try:
            outro_start = max(0, track1_end_sample - transition_length)
            
            # Find downbeats in both tracks during the transition
            track1_downbeats_samples = librosa.frames_to_samples(track1.downbeats, hop_length=512)
            track2_downbeats_samples = librosa.frames_to_samples(track2.downbeats, hop_length=512)
            
            # Track1 downbeats during transition
            track1_transition_downbeats = track1_downbeats_samples[
                (track1_downbeats_samples >= outro_start) & 
                (track1_downbeats_samples <= track1_end_sample)
            ]
            
            # Track2 downbeats during transition (adjusted for start position)
            track2_absolute_downbeats = track2_downbeats_samples + track2_start_sample
            track2_transition_downbeats = track2_absolute_downbeats[
                (track2_absolute_downbeats >= outro_start) & 
                (track2_absolute_downbeats <= track1_end_sample)
            ]
            
            if len(track1_transition_downbeats) > 0 and len(track2_transition_downbeats) > 0:
                # Find the closest downbeat pairs and check alignment
                min_offset = float('inf')
                best_alignment = None
                
                for t1_beat in track1_transition_downbeats:
                    for t2_beat in track2_transition_downbeats:
                        offset = abs(t1_beat - t2_beat) / track1.sr
                        if offset < min_offset:
                            min_offset = offset
                            best_alignment = (t1_beat, t2_beat)
                
                if best_alignment:
                    t1_time = best_alignment[0] / track1.sr
                    t2_time = best_alignment[1] / track1.sr
                    
                    if min_offset < 0.01:  # Within 10ms
                        print(f"  ✓ Perfect simultaneous alignment: {min_offset*1000:.1f}ms offset")
                        print(f"    Track1 downbeat: {t1_time:.3f}s, Track2 downbeat: {t2_time:.3f}s")
                    elif min_offset < 0.05:  # Within 50ms  
                        print(f"  ✓ Good simultaneous alignment: {min_offset*1000:.1f}ms offset")
                        print(f"    Track1 downbeat: {t1_time:.3f}s, Track2 downbeat: {t2_time:.3f}s")
                    else:
                        print(f"  ⚠ Alignment warning: {min_offset*1000:.1f}ms offset")
                        print(f"    Track1 downbeat: {t1_time:.3f}s, Track2 downbeat: {t2_time:.3f}s")
                else:
                    print("  ⚠ Could not find matching downbeat pairs")
            else:
                print("  ⚠ Could not verify downbeat alignment (no downbeats in transition zone)")
                print(f"    Track1 downbeats in transition: {len(track1_transition_downbeats)}")
                print(f"    Track2 downbeats in transition: {len(track2_transition_downbeats)}")
                
        except Exception as e:
            print(f"  Warning: Downbeat alignment verification failed: {e}")
    
    def calculate_beat_phase_alignment(self, track1: Track, track2: Track, 
                                     track1_end_frame: int, track2_start_frame: int) -> float:
        """
        Calculate the phase alignment needed to sync beats perfectly
        Returns time offset in seconds to align track2 with track1's beat grid
        """
        # Get beat positions around the transition points
        track1_beats = track1.beats[track1.beats <= track1_end_frame]
        track2_beats = track2.beats[track2.beats >= track2_start_frame]
        
        if len(track1_beats) == 0 or len(track2_beats) == 0:
            return 0.0
        
        # Find the expected beat interval from track1
        if len(track1_beats) > 1:
            track1_beat_interval = np.mean(np.diff(track1_beats))
        else:
            # Fallback: calculate from BPM
            track1_beat_interval = (60.0 / track1.bpm) * track1.sr / 512  # frames per beat
        
        # Find how far track2's first beat is from the ideal position
        track2_first_beat = track2_beats[0] - track2_start_frame
        
        # Calculate phase offset (how much to adjust track2 timing)
        # Positive offset means track2 should start slightly later
        phase_offset_frames = track2_first_beat % track1_beat_interval
        
        # Convert to seconds
        phase_offset_seconds = librosa.frames_to_time(phase_offset_frames, sr=track1.sr, hop_length=512)
        
        return phase_offset_seconds