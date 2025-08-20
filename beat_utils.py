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
        Returns (track1_outro, track2_intro) with precise downbeat alignment
        """
        transition_samples = int(transition_duration * track1.sr)
        
        # Extract outro from track1 (leading up to the downbeat)
        outro_start = max(0, track1_end_sample - transition_samples)
        track1_outro = track1.audio[outro_start:track1_end_sample]
        
        # Extract intro from track2 (starting from the aligned position)
        # Note: track2_start_sample has already been aligned for perfect downbeat sync
        intro_end = min(len(track2.audio), track2_start_sample + transition_samples)
        track2_intro = track2.audio[track2_start_sample:intro_end]
        
        # Ensure both segments are the same length for crossfade
        min_length = min(len(track1_outro), len(track2_intro))
        if min_length < transition_samples * 0.5:  # If segments are too short
            print(f"  Warning: Short transition segments ({min_length/track1.sr:.1f}s)")
            print(f"  Track1 outro: {len(track1_outro)/track1.sr:.1f}s, Track2 intro: {len(track2_intro)/track2.sr:.1f}s")
        
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Verify downbeat alignment (for debugging)
        self._verify_downbeat_alignment(track1, track2, track1_end_sample, track2_start_sample, min_length)
        
        return track1_outro, track2_intro
    
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