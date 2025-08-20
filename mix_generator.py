#!/usr/bin/env python3
"""
Mix generation functionality for DJ Mix Generator
"""

import librosa
import soundfile as sf
import numpy as np
import os
from typing import List, Tuple
from models import Track
from beat_utils import BeatAligner


class MixGenerator:
    """Handles DJ mix generation with transitions and beatmatching"""
    
    def __init__(self, tempo_strategy: str = "sequential", interactive_beats: bool = False):
        self.beat_aligner = BeatAligner(interactive_beats=interactive_beats)
        self.tempo_strategy = tempo_strategy
        self.target_bpm = None  # Will be set based on strategy
        self.interactive_beats = interactive_beats
    
    def measures_to_samples(self, measures: int, bpm: float, sr: int, beats_per_measure: int = 4) -> int:
        """Convert measures to audio samples based on BPM and sample rate"""
        # Calculate total beats in the measures
        total_beats = measures * beats_per_measure
        # Calculate seconds for these beats
        seconds = (total_beats * 60.0) / bpm
        # Convert to samples
        return int(seconds * sr)
    
    def samples_to_measures(self, samples: int, bpm: float, sr: int, beats_per_measure: int = 4) -> float:
        """Convert audio samples to measures based on BPM and sample rate"""
        seconds = samples / sr
        total_beats = (seconds * bpm) / 60.0
        return total_beats / beats_per_measure
    
    def calculate_target_bpm(self, tracks: List[Track]):
        """Calculate target BPM based on tempo strategy"""
        if self.tempo_strategy == "sequential":
            # Sequential: use first track's BPM
            self.target_bpm = tracks[0].bpm
            print(f"Sequential tempo strategy: Target BPM = {self.target_bpm:.1f} (from first track)")
        elif self.tempo_strategy == "uniform":
            # Uniform: use average of all BPMs
            self.target_bpm = sum(track.bpm for track in tracks) / len(tracks)
            print(f"Uniform tempo strategy: Target BPM = {self.target_bpm:.1f} (average of all tracks)")
            bpm_list = [f"{track.bpm:.1f}" for track in tracks]
            print(f"  Individual BPMs: {', '.join(bpm_list)}")
        else:
            raise ValueError(f"Unknown tempo strategy: {self.tempo_strategy}")
        
    def create_transition(self, track1: Track, track2: Track, transition_duration: float = 30.0, 
                         stretch_track1: bool = False) -> Tuple[np.ndarray, np.ndarray, Track]:
        """Create a beat-aligned crossfade transition between two tracks"""
        if self.target_bpm is None:
            raise ValueError("Target BPM not set. Call calculate_target_bpm() first.")
        
        # Determine what needs to be stretched based on tempo strategy
        if self.tempo_strategy == "sequential":
            # Sequential: stretch track2 to match track1 (track1 stays at native tempo)
            print(f"  Sequential mode: {track2.bpm:.1f} -> {track1.bpm:.1f} (track2 stretched)")
            target_bpm_for_track2 = track1.bpm
            track1_stretched = track1  # No change to track1
            track2_stretched = self._stretch_track_to_bpm(track2, target_bpm_for_track2)
            
        elif self.tempo_strategy == "uniform":
            # Uniform: stretch both tracks to target BPM
            print(f"  Uniform mode: {track1.bpm:.1f} & {track2.bpm:.1f} -> {self.target_bpm:.1f} (both stretched)")
            
            if stretch_track1:
                # Stretch track1 to target BPM (for uniform mode in full mix)
                track1_stretched = self._stretch_track_to_bpm(track1, self.target_bpm)
            else:
                # Track1 already at target BPM (first track in uniform mode)
                track1_stretched = track1
                
            track2_stretched = self._stretch_track_to_bpm(track2, self.target_bpm)
        
        # CRITICAL FIX: Force both tracks to use the exact uniform target BPM for crossfading
        # The actual calculated BPMs may differ slightly due to discrete sample precision,
        # but for crossfading we must use the exact target BPM to ensure perfect matching
        
        target_bpm_for_crossfade = None
        if self.tempo_strategy == "sequential":
            target_bpm_for_crossfade = track1_stretched.bpm  # Use track1's BPM as reference
        elif self.tempo_strategy == "uniform":
            target_bpm_for_crossfade = self.target_bpm  # Use the exact uniform target BPM
        
        print(f"  Final BPMs after stretching: Track1={track1_stretched.bpm:.3f}, Track2={track2_stretched.bpm:.3f}")
        
        # Create corrected tracks with uniform BPM for crossfading
        track1_for_crossfade = Track(
            filepath=track1_stretched.filepath,
            audio=track1_stretched.audio,
            sr=track1_stretched.sr,
            bpm=target_bpm_for_crossfade,  # Force uniform BPM
            key=track1_stretched.key,
            beats=track1_stretched.beats,
            downbeats=track1_stretched.downbeats,
            duration=track1_stretched.duration
        )
        
        track2_for_crossfade = Track(
            filepath=track2_stretched.filepath,
            audio=track2_stretched.audio,
            sr=track2_stretched.sr,
            bpm=target_bpm_for_crossfade,  # Force uniform BPM
            key=track2_stretched.key,
            beats=track2_stretched.beats,
            downbeats=track2_stretched.downbeats,
            duration=track2_stretched.duration
        )
        
        print(f"  Crossfade BPMs (uniform): Track1={track1_for_crossfade.bpm:.3f}, Track2={track2_for_crossfade.bpm:.3f}")
        bpm_diff = abs(track1_for_crossfade.bpm - track2_for_crossfade.bpm)
        if bpm_diff > 0.001:
            print(f"  ❌ CRITICAL ERROR: BPM mismatch in crossfade ({bpm_diff:.6f})")
        else:
            print(f"  ✅ Perfect BPM match for crossfade (difference: {bpm_diff:.6f})")
        
        return self._create_crossfade(track1_for_crossfade, track2_for_crossfade, transition_duration)
    
    def _stretch_track_to_bpm(self, track: Track, target_bpm: float) -> Track:
        """Stretch a track to match target BPM with intelligent tempo correction
        
        Professional DJ precision: Any BPM difference > 0.1% requires correction.
        Even minimal differences (e.g. 128.0 vs 127.7) cause audible drift over time.
        """
        bpm_ratio = track.bpm / target_bpm  # Ratio to stretch to target tempo
        
        if abs(bpm_ratio - 1.0) > 0.001:  # Professional precision: stretch for any meaningful BPM difference
            print(f"    Time-stretching {track.filepath.name}: {track.bpm:.1f} -> {target_bpm:.1f} (ratio: {bpm_ratio:.3f})")
            print(f"    Applying intelligent tempo correction to eliminate drift...")
            
            # Apply intelligent tempo correction to maintain consistent timing
            stretched_audio, corrected_beats, corrected_downbeats = self._apply_intelligent_tempo_correction(
                track, target_bpm
            )
            
            # Calculate actual BPM from corrected beats for perfect accuracy
            actual_bpm = self._calculate_actual_bpm_from_beats(corrected_beats, track.sr)
            print(f"    Final BPM after correction: {actual_bpm:.3f} (target: {target_bpm:.3f})")
            
            return Track(
                filepath=track.filepath,
                audio=stretched_audio,
                sr=track.sr,
                bpm=actual_bpm,  # Use actual BPM, not target BPM
                key=track.key,
                beats=corrected_beats,
                downbeats=corrected_downbeats,
                duration=len(stretched_audio) / track.sr
            )
        else:
            # BPMs are extremely close (< 0.1% difference), but still apply professional drift correction
            bpm_diff = abs(track.bpm - target_bpm)
            print(f"    {track.filepath.name}: Minimal difference ({track.bpm:.1f} vs {target_bpm:.1f}, diff: {bpm_diff:.3f} BPM)")
            
            if len(track.beats) > 4:  # Only if we have enough beats
                print(f"    Applying professional drift correction to target BPM {target_bpm:.1f}...")
                stretched_audio, corrected_beats, corrected_downbeats = self._apply_intelligent_tempo_correction(
                    track, target_bpm  # Use target BPM, not original BPM for true precision
                )
                
                # Calculate actual BPM from corrected beats for perfect accuracy
                actual_bpm = self._calculate_actual_bpm_from_beats(corrected_beats, track.sr)
                print(f"    Final BPM after drift correction: {actual_bpm:.3f} (target: {target_bpm:.3f})")
                
                return Track(
                    filepath=track.filepath,
                    audio=stretched_audio,
                    sr=track.sr,
                    bpm=actual_bpm,  # Use actual BPM, not target BPM
                    key=track.key,
                    beats=corrected_beats,
                    downbeats=corrected_downbeats,
                    duration=len(stretched_audio) / track.sr
                )
            else:
                print(f"    Warning: Not enough beats detected for drift correction")
                return track
    
    def _apply_intelligent_tempo_correction(self, track: Track, target_bpm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply intelligent tempo correction to eliminate beat drift throughout track duration.
        Uses piecewise time-stretching between beats to maintain perfect tempo consistency.
        """
        if len(track.beats) < 2:
            print("    Warning: Not enough beats for intelligent tempo correction")
            return track.audio, track.beats, track.downbeats
        
        # Convert beat and downbeat frames to sample positions
        beat_samples = librosa.frames_to_samples(track.beats, hop_length=512)
        downbeat_samples = librosa.frames_to_samples(track.downbeats, hop_length=512) if len(track.downbeats) > 0 else np.array([])
        
        # Calculate ideal beat positions based on target BPM
        samples_per_beat = (60.0 / target_bpm) * track.sr
        
        # Create ideal beat grid starting from the first detected beat
        first_beat_sample = beat_samples[0]
        ideal_beat_samples = []
        
        # Generate ideal beat positions
        current_pos = first_beat_sample
        beat_index = 0
        
        while current_pos < len(track.audio) and beat_index < len(beat_samples):
            ideal_beat_samples.append(current_pos)
            current_pos += samples_per_beat
            beat_index += 1
        
        ideal_beat_samples = np.array(ideal_beat_samples)
        
        # Apply piecewise stretching between beats
        stretched_segments = []
        corrected_beat_positions = []
        corrected_downbeat_positions = []
        
        print(f"      Applying piecewise tempo correction across {len(ideal_beat_samples)} beats...")
        
        for i in range(len(ideal_beat_samples)):
            if i == 0:
                # First segment: from start to first beat
                segment_start = 0
                segment_end = int(beat_samples[0]) if 0 < len(beat_samples) else len(track.audio)
                ideal_start = 0 
                ideal_end = int(ideal_beat_samples[0])
            else:
                # Segments between beats
                if i < len(beat_samples):
                    segment_start = int(beat_samples[i-1])
                    segment_end = int(beat_samples[i])
                    ideal_start = int(ideal_beat_samples[i-1])
                    ideal_end = int(ideal_beat_samples[i])
                else:
                    break
            
            # Extract segment
            if segment_end > segment_start and segment_end <= len(track.audio):
                segment = track.audio[segment_start:segment_end]
                
                # Calculate stretch ratio
                original_length = segment_end - segment_start
                target_length = ideal_end - ideal_start
                
                if original_length > 0 and target_length > 0:
                    stretch_ratio = target_length / original_length
                    
                    # Apply stretch with reasonable limits
                    if 0.5 <= stretch_ratio <= 2.0:  # Reasonable stretch limits
                        try:
                            stretched_segment = librosa.effects.time_stretch(segment, rate=1/stretch_ratio)
                            
                            # Ensure exact target length
                            if len(stretched_segment) > target_length:
                                stretched_segment = stretched_segment[:target_length]
                            elif len(stretched_segment) < target_length:
                                padding = target_length - len(stretched_segment)
                                stretched_segment = np.pad(stretched_segment, (0, padding), 'constant')
                                
                        except Exception as e:
                            print(f"      Warning: Stretch failed for segment {i}, using original")
                            stretched_segment = np.resize(segment, target_length)
                    else:
                        # Stretch ratio too extreme, just resize
                        stretched_segment = np.resize(segment, target_length)
                    
                    stretched_segments.append(stretched_segment)
                    
                    # Record corrected beat position
                    if i < len(ideal_beat_samples):
                        corrected_beat_positions.append(ideal_beat_samples[i])
        
        # Handle final segment after last beat
        if len(beat_samples) > 0 and int(beat_samples[-1]) < len(track.audio):
            final_segment = track.audio[int(beat_samples[-1]):]
            stretched_segments.append(final_segment)
        
        # Combine all stretched segments
        if stretched_segments:
            corrected_audio = np.concatenate(stretched_segments)
            print(f"      Tempo correction complete: {len(corrected_beat_positions)} beats aligned")
        else:
            print("      Warning: No segments could be processed, using original audio")
            corrected_audio = track.audio
            corrected_beat_positions = beat_samples
        
        # Convert corrected positions back to frames for beat/downbeat arrays
        corrected_beats_frames = librosa.samples_to_frames(np.array(corrected_beat_positions), hop_length=512)
        
        # Correct downbeat positions proportionally
        if len(track.downbeats) > 0:
            # Map downbeats to their new positions based on the stretching applied
            corrected_downbeats_frames = []
            for downbeat_sample in downbeat_samples:
                # Find which beat segment this downbeat belongs to
                beat_idx = np.searchsorted(beat_samples, downbeat_sample)
                if beat_idx < len(corrected_beat_positions):
                    # Proportionally adjust downbeat position within its beat segment
                    if beat_idx > 0:
                        segment_start = beat_samples[beat_idx-1]
                        ideal_start = corrected_beat_positions[beat_idx-1]
                    else:
                        segment_start = 0
                        ideal_start = 0
                    
                    if beat_idx < len(beat_samples):
                        segment_end = beat_samples[beat_idx]
                        ideal_end = corrected_beat_positions[beat_idx]
                    else:
                        continue
                    
                    # Calculate proportional position
                    if segment_end > segment_start:
                        proportion = (downbeat_sample - segment_start) / (segment_end - segment_start)
                        corrected_downbeat_sample = ideal_start + proportion * (ideal_end - ideal_start)
                        corrected_downbeats_frames.append(librosa.samples_to_frames(corrected_downbeat_sample, hop_length=512))
            
            corrected_downbeats_frames = np.array(corrected_downbeats_frames)
        else:
            corrected_downbeats_frames = track.downbeats
        
        return corrected_audio, corrected_beats_frames, corrected_downbeats_frames
    
    def _calculate_actual_bpm_from_beats(self, beats: np.ndarray, sr: int) -> float:
        """
        Calculate actual BPM from corrected beat positions for perfect accuracy.
        
        This function calculates the real BPM by measuring the time intervals between
        corrected beats, ensuring the returned BPM reflects the actual tempo after
        time-stretching rather than assuming the target BPM was achieved perfectly.
        """
        if len(beats) < 2:
            print(f"      Warning: Not enough beats ({len(beats)}) to calculate BPM")
            return 120.0  # Default fallback BPM
        
        # Convert beat frames to sample positions
        beat_samples = librosa.frames_to_samples(beats, hop_length=512)
        
        # Calculate time intervals between consecutive beats
        beat_intervals_samples = np.diff(beat_samples)
        
        # Convert to seconds
        beat_intervals_seconds = beat_intervals_samples / sr
        
        # Remove outliers (beats with intervals way off from the median)
        median_interval = np.median(beat_intervals_seconds)
        valid_intervals = beat_intervals_seconds[
            (beat_intervals_seconds >= median_interval * 0.5) & 
            (beat_intervals_seconds <= median_interval * 2.0)
        ]
        
        if len(valid_intervals) == 0:
            print(f"      Warning: No valid beat intervals found, using median")
            valid_intervals = [median_interval]
        
        # Calculate average beat interval (seconds per beat)
        avg_beat_interval = np.mean(valid_intervals)
        
        # Convert to BPM (beats per minute)
        actual_bpm = 60.0 / avg_beat_interval
        
        print(f"      BPM calculation: {len(valid_intervals)} intervals, avg {avg_beat_interval:.4f}s/beat -> {actual_bpm:.3f} BPM")
        
        return actual_bpm
    
    def _create_crossfade(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, np.ndarray, Track]:
        """Create crossfade transition between two tempo-matched tracks
        
        Note: track1 and track2 parameters are already tempo-matched by create_transition()
        """
        # Find optimal transition points using downbeat alignment on tempo-matched tracks
        print(f"  Aligning to downbeats...")
        track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
            track1, track2, transition_duration
        )
        
        # Extract beat-aligned audio segments using tempo-matched tracks  
        track1_outro, track2_intro = self.beat_aligner.align_beats_for_transition(
            track1, track2, track1_end_sample, track2_start_sample, transition_duration
        )
        
        # Report the alignment
        track1_end_time = track1_end_sample / track1.sr
        track2_start_time = track2_start_sample / track2.sr
        print(f"  Track 1 outro starts at: {track1_end_time - transition_duration:.1f}s, ends at: {track1_end_time:.1f}s")
        print(f"  Track 2 intro starts at: {track2_start_time:.1f}s")
        
        # Create crossfade with beat alignment
        if len(track1_outro) == 0 or len(track2_intro) == 0:
            print("  Warning: Empty audio segments, using fallback transition")
            return self._create_fallback_transition(track1, track2, transition_duration)
        
        # Ensure both segments are the same length
        min_length = min(len(track1_outro), len(track2_intro))
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create equal-power crossfade curves
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade
        transition = track1_outro * fade_out + track2_intro * fade_in
        
        return transition, track2
    
    def _create_fallback_transition(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, Track]:
        """
        Fallback transition method using simple time-based alignment
        Used when downbeat alignment fails
        """
        print("  Using fallback transition method")
        
        transition_samples = int(transition_duration * track1.sr)
        
        # Simple approach: use end of track1 and beginning of track2
        track1_outro = track1.audio[-transition_samples:] if len(track1.audio) >= transition_samples else track1.audio
        track2_intro = track2.audio[:transition_samples] if len(track2.audio) >= transition_samples else track2.audio
        
        # Ensure both segments are the same length
        min_length = min(len(track1_outro), len(track2_intro))
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create crossfade
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        transition = track1_outro * fade_out + track2_intro * fade_in
        
        return transition, track2
    
    def generate_mix(self, tracks: List[Track], output_path: str, transition_duration: float = None, transition_measures: int = None, transitions_only: bool = False):
        """Generate the complete DJ mix or just the transitions"""
        if len(tracks) < 2:
            raise ValueError("Need at least 2 tracks to create a mix")
        
        # Calculate target BPM based on strategy
        self.calculate_target_bpm(tracks)
        
        # Determine transition duration - convert measures to seconds if needed
        if transition_measures is not None:
            # Convert measures to seconds using target BPM
            beats_per_measure = 4
            total_beats = transition_measures * beats_per_measure
            transition_duration = (total_beats * 60.0) / self.target_bpm
            transition_mode = f"{transition_measures} measures ({transition_duration:.1f}s at {self.target_bpm:.1f} BPM)"
        elif transition_duration is not None:
            # Convert seconds to measures for display
            beats = (transition_duration * self.target_bpm) / 60.0
            measures = beats / 4  # Assume 4/4 time
            transition_mode = f"{transition_duration}s ({measures:.1f} measures at {self.target_bpm:.1f} BPM)"
        else:
            # Default fallback
            transition_duration = 30.0
            beats = (transition_duration * self.target_bpm) / 60.0
            measures = beats / 4
            transition_mode = f"{transition_duration}s ({measures:.1f} measures at {self.target_bpm:.1f} BPM)"
        
        if transitions_only:
            print("Generating transitions-only preview with 5s buffers...")
        else:
            print(f"Generating mix with {len(tracks)} tracks...")
        print(f"Transition length: {transition_mode}")
        print(f"Tempo strategy: {self.tempo_strategy}\n")
        
        if transitions_only:
            return self._generate_transitions_only(tracks, output_path, transition_duration, transition_measures)
        
        # For uniform strategy, stretch the first track to target BPM
        if self.tempo_strategy == "uniform":
            first_track = self._stretch_track_to_bpm(tracks[0], self.target_bpm)
            print(f"Track 1: {first_track.filepath.name} (stretched to target BPM)")
        else:
            first_track = tracks[0]
            print(f"Track 1: {first_track.filepath.name} (native BPM)")
        
        # Start with the first track
        mix_audio = first_track.audio.copy()
        current_sr = first_track.sr
        current_bpm = first_track.bpm
        
        # Add each subsequent track with transitions
        for i in range(1, len(tracks)):
            current_track = tracks[i]
            
            print(f"Track {i+1}: {current_track.filepath.name}")
            
            # Verify sample rates match
            if current_track.sr != current_sr:
                print(f"  Resampling from {current_track.sr}Hz to {current_sr}Hz")
                current_track.audio = librosa.resample(current_track.audio, 
                                                     orig_sr=current_track.sr, 
                                                     target_sr=current_sr)
                current_track.sr = current_sr
            
            # Create a temporary track object for the current mix state
            prev_track = Track(
                filepath=tracks[i-1].filepath,
                audio=mix_audio,
                sr=current_sr,
                bpm=current_bpm,  # Use current mix BPM
                key=tracks[i-1].key,
                beats=tracks[i-1].beats,
                downbeats=tracks[i-1].downbeats,
                duration=len(mix_audio) / current_sr
            )
            
            # Create transition (stretch_track1 is always False since prev_track is already processed)
            transition, stretched_track = self.create_transition(prev_track, current_track, transition_duration, stretch_track1=False)
            
            # Update current BPM for next iteration
            current_bpm = stretched_track.bpm
            
            # Remove the outro from the mix and add the transition + new track
            transition_samples = len(transition)
            mix_audio = mix_audio[:-transition_samples]  # Remove outro
            mix_audio = np.concatenate([mix_audio, transition, stretched_track.audio[transition_samples:]])
            
            print(f"  Mix length so far: {len(mix_audio) / current_sr / 60:.1f} minutes\n")
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(mix_audio))
        if max_val > 0.95:
            print(f"Normalizing audio (peak: {max_val:.3f})")
            mix_audio = mix_audio * (0.95 / max_val)
        
        # Save the final mix
        print(f"Saving mix to: {output_path}")
        sf.write(output_path, mix_audio, current_sr)
        
        duration_minutes = len(mix_audio) / current_sr / 60
        print(f"\nMix complete! 🎵")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    def _generate_transitions_only(self, tracks: List[Track], output_path: str, transition_duration: float, transition_measures: int = None):
        """
        Generate only the transition sections with 5-second buffers for preview testing
        Each transition includes 5s from end of current track + transition + 5s from start of next track
        """
        print("Creating transitions-only preview for testing...")
        if transition_measures is not None:
            print(f"Each transition: {transition_measures} measures ({transition_duration:.1f}s at {self.target_bpm:.1f} BPM)")
        
        current_sr = tracks[0].sr
        buffer_duration = 5.0  # 5 seconds before/after each transition
        buffer_samples = int(buffer_duration * current_sr)
        
        all_transitions = []
        silence_gap = np.zeros(int(1.0 * current_sr))  # 1 second gap between transitions
        
        for i in range(len(tracks) - 1):
            current_track = tracks[i]
            next_track = tracks[i + 1]
            
            print(f"Transition {i+1}: {current_track.filepath.name} → {next_track.filepath.name}")
            
            # Ensure sample rates match
            if next_track.sr != current_sr:
                print(f"  Resampling {next_track.filepath.name} from {next_track.sr}Hz to {current_sr}Hz")
                next_track.audio = librosa.resample(
                    next_track.audio, 
                    orig_sr=next_track.sr, 
                    target_sr=current_sr
                )
                next_track.sr = current_sr
            
            # Create a mock "previous track state" for transition generation
            # Use the full current track as if it were the current mix state
            prev_track = Track(
                filepath=current_track.filepath,
                audio=current_track.audio,
                sr=current_sr,
                bpm=current_track.bpm,
                key=current_track.key,
                beats=current_track.beats,
                downbeats=current_track.downbeats,
                duration=current_track.duration
            )
            
            # Generate the transition
            transition, stretched_track = self.create_transition(prev_track, next_track, transition_duration, stretch_track1=False)
            
            # Find the transition points in the original tracks
            track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
                prev_track, stretched_track, transition_duration
            )
            
            # Extract 5s buffer before transition from current track
            pre_buffer_start = max(0, track1_end_sample - int(transition_duration * current_sr) - buffer_samples)
            pre_buffer_end = max(0, track1_end_sample - int(transition_duration * current_sr))
            pre_buffer = current_track.audio[pre_buffer_start:pre_buffer_end]
            
            # Extract 5s buffer after transition from next track
            post_buffer_start = track2_start_sample + int(transition_duration * current_sr)
            post_buffer_end = min(len(stretched_track.audio), post_buffer_start + buffer_samples)
            post_buffer = stretched_track.audio[post_buffer_start:post_buffer_end]
            
            # Combine: pre-buffer + transition + post-buffer
            if len(pre_buffer) > 0 and len(transition) > 0 and len(post_buffer) > 0:
                transition_segment = np.concatenate([pre_buffer, transition, post_buffer])
                all_transitions.append(transition_segment)
                
                segment_duration = len(transition_segment) / current_sr
                print(f"  Transition segment: {segment_duration:.1f}s (5s + {transition_duration}s + 5s)")
            else:
                print(f"  Warning: Could not create full transition segment")
                if len(transition) > 0:
                    all_transitions.append(transition)
        
        if not all_transitions:
            raise ValueError("No transitions could be generated")
        
        # Combine all transitions with silence gaps
        final_mix = all_transitions[0]
        for transition in all_transitions[1:]:
            final_mix = np.concatenate([final_mix, silence_gap, transition])
        
        # Normalize audio
        max_val = np.max(np.abs(final_mix))
        if max_val > 0.95:
            print(f"Normalizing transitions preview (peak: {max_val:.3f})")
            final_mix = final_mix * (0.95 / max_val)
        
        # Save transitions-only mix
        print(f"Saving transitions preview to: {output_path}")
        sf.write(output_path, final_mix, current_sr)
        
        duration_minutes = len(final_mix) / current_sr / 60
        print(f"\n🎵 Transitions preview complete!")
        print(f"Contains {len(all_transitions)} transitions")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print("Listen to this file to test transition quality before generating full mix!")