#!/usr/bin/env python3
"""
Beat alignment utilities for precise DJ mixing
"""

import numpy as np
import librosa
from typing import Tuple
from models import Track


class BeatAligner:
    """Handles precise beat and downbeat alignment for seamless transitions"""
    
    def __init__(self):
        pass
    
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
            track1, track2, track1_outro, track2_intro, outro_start, track2_start_sample
        )
        
        # Verify alignment quality
        self._verify_perfect_alignment(track1, track2, track1_end_sample, track2_start_sample, len(track1_outro_aligned))
        
        return track1_outro_aligned, track2_intro_aligned
    
    def _apply_beat_by_beat_alignment(self, track1: Track, track2: Track, 
                                    track1_outro: np.ndarray, track2_intro: np.ndarray,
                                    outro_start: int, track2_start_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perfect beat-by-beat alignment throughout the transition period.
        Uses micro-stretching and phase alignment to ensure every beat matches.
        """
        try:
            print("  Applying perfect beat-by-beat alignment...")
            
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
            
            # Find downbeats within the transition segments
            track1_segment_downbeats = track1_downbeats_samples[
                (track1_downbeats_samples >= outro_start) & 
                (track1_downbeats_samples <= outro_start + len(track1_outro))
            ] - outro_start
            
            track2_segment_downbeats = track2_downbeats_samples[
                (track2_downbeats_samples >= track2_start_sample) & 
                (track2_downbeats_samples <= track2_start_sample + len(track2_intro))
            ] - track2_start_sample
            
            if len(track1_segment_beats) == 0 or len(track2_segment_beats) == 0:
                print("  Warning: No beats found in transition segments, using original audio")
                return track1_outro, track2_intro
            
            # Create ideal beat grid based on track1's tempo (the reference)
            beats_per_second = track1.bpm / 60.0
            samples_per_beat = track1.sr / beats_per_second
            
            # Generate ideal beat positions for the transition duration
            segment_duration = len(track1_outro) / track1.sr
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
                print("  Warning: No ideal beats generated, using original audio")
                return track1_outro, track2_intro
            
            print(f"    Track1 beats in segment: {len(track1_segment_beats)}")
            print(f"    Track2 beats in segment: {len(track2_segment_beats)}")
            print(f"    Ideal beat grid: {len(ideal_beat_positions)} beats")
            
            # Apply micro-adjustments to track2 to match the ideal beat grid
            track2_intro_aligned = self._micro_stretch_to_beat_grid(
                track2_intro, track2_segment_beats, ideal_beat_positions, track2.sr
            )
            
            # Ensure both tracks are exactly the same length
            min_length = min(len(track1_outro), len(track2_intro_aligned))
            track1_outro_final = track1_outro[:min_length]
            track2_intro_final = track2_intro_aligned[:min_length]
            
            print(f"    Final alignment: {min_length} samples ({min_length/track1.sr:.3f}s)")
            
            return track1_outro_final, track2_intro_final
            
        except Exception as e:
            print(f"  Warning: Beat-by-beat alignment failed: {e}")
            print("  Using original audio segments")
            return track1_outro, track2_intro
    
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