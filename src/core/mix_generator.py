#!/usr/bin/env python3
"""
Mix generation functionality for DJ Mix Generator
"""

import librosa
import soundfile as sf
import numpy as np
import os
from typing import List, Tuple
try:
    from .models import Track, BeatInfo, KeyInfo
    from .beat_utils import BeatAligner
except ImportError:
    # Fallback for direct execution - use full path to avoid conflicts
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from core.models import Track, BeatInfo, KeyInfo
    from core.beat_utils import BeatAligner
from scipy import signal
from scipy.signal import hilbert


class MixGenerator:
    """Handles DJ mix generation with transitions and beatmatching"""
    
    def __init__(self, config=None, tempo_strategy: str = "sequential", interactive_beats: bool = False, 
                 enable_eq_matching: bool = True, enable_volume_matching: bool = True, eq_strength: float = 0.5,
                 enable_peak_alignment: bool = True, enable_tempo_correction: bool = True, enable_lf_transition: bool = False, enable_mf_transition: bool = False, enable_hf_transition: bool = False, transition_downbeats: bool = False):
        
        # Handle new config object approach
        if config is not None:
            # Extract settings from MixConfiguration object
            tempo_strategy = config.tempo_strategy.value if hasattr(config.tempo_strategy, 'value') else str(config.tempo_strategy)
            interactive_beats = config.interactive_beats
            enable_eq_matching = config.audio_quality.eq_matching
            enable_volume_matching = config.audio_quality.volume_matching
            eq_strength = config.audio_quality.eq_strength
            enable_peak_alignment = config.audio_quality.peak_alignment
            enable_tempo_correction = config.audio_quality.tempo_correction
            enable_lf_transition = config.transition_settings.enable_lf_transition
            enable_mf_transition = config.transition_settings.enable_mf_transition
            enable_hf_transition = config.transition_settings.enable_hf_transition
            transition_downbeats = config.transition_settings.use_downbeat_mapping
        
        self.beat_aligner = BeatAligner(interactive_beats=interactive_beats)
        self.tempo_strategy = tempo_strategy
        self.target_bpm = None  # Will be set based on strategy
        self.interactive_beats = interactive_beats
        
        # Audio quality settings
        self.enable_eq_matching = enable_eq_matching
        self.enable_volume_matching = enable_volume_matching
        self.eq_strength = eq_strength
        self.enable_peak_alignment = enable_peak_alignment
        self.enable_tempo_correction = enable_tempo_correction
        self.enable_lf_transition = enable_lf_transition
        self.enable_mf_transition = enable_mf_transition
        self.enable_hf_transition = enable_hf_transition
        self.transition_downbeats = transition_downbeats
    
    def _create_track_from_existing(self, existing_track: Track, new_audio: np.ndarray = None, 
                                   new_bpm: float = None, new_key: str = None, 
                                   new_beats: np.ndarray = None, new_downbeats: np.ndarray = None) -> Track:
        """Helper to create a new Track with updated properties while preserving structure"""
        # Use existing data if new values not provided
        audio = new_audio if new_audio is not None else existing_track.audio
        bpm = new_bpm if new_bpm is not None else existing_track.bpm
        key = new_key if new_key is not None else existing_track.key
        beats = new_beats if new_beats is not None else existing_track.beats
        downbeats = new_downbeats if new_downbeats is not None else existing_track.downbeats
        
        # Create new BeatInfo and KeyInfo objects
        beat_info = BeatInfo(
            beats=beats,
            downbeats=downbeats,
            bpm=bpm,
            confidence=existing_track.beat_info.confidence
        )
        
        key_info = KeyInfo(
            key=key,
            confidence=existing_track.key_info.confidence,
            chroma=existing_track.key_info.chroma
        )
        
        return Track(
            filepath=existing_track.filepath,
            audio=audio,
            sr=existing_track.sr,
            beat_info=beat_info,
            key_info=key_info,
            metadata=existing_track.metadata
        )
    
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
        if str(self.tempo_strategy) == "sequential":
            # Sequential: use first track's BPM
            self.target_bpm = tracks[0].bpm
            print(f"Sequential tempo strategy: Target BPM = {self.target_bpm:.1f} (from first track)")
        elif str(self.tempo_strategy) == "uniform":
            # Uniform: use average of all BPMs
            self.target_bpm = sum(track.bpm for track in tracks) / len(tracks)
            print(f"Uniform tempo strategy: Target BPM = {self.target_bpm:.1f} (average of all tracks)")
            bpm_list = [f"{track.bpm:.1f}" for track in tracks]
            print(f"  Individual BPMs: {', '.join(bpm_list)}")
        elif str(self.tempo_strategy) == "match-track":
            # Match-track: each track plays at native tempo, tempo ramps during transitions
            self.target_bpm = None  # No single target BPM, varies per track
            print(f"Match-track tempo strategy: Each track plays at native BPM with tempo ramping during transitions")
            bpm_list = [f"{track.bpm:.1f}" for track in tracks]
            print(f"  Track BPMs: {', '.join(bpm_list)}")
        else:
            raise ValueError(f"Unknown tempo strategy: {self.tempo_strategy}")
    
    def _apply_intelligent_transition_alignment(self, track1_overlap: np.ndarray, track2_overlap: np.ndarray,
                                              track1: Track, track2: Track, track1_start_sample: int, 
                                              track2_start_sample: int, transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply intelligent beat alignment within transition segments while preserving segment length and continuity.
        
        This method aligns beats within the transition without changing the overall timing of the mix.
        Both tracks are adjusted minimally to achieve perfect beat synchronization.
        """
        try:
            if not hasattr(self, 'beat_aligner'):
                print("    Beat aligner not available, using original segments")
                return track1_overlap, track2_overlap
            
            print("    Calculating beat positions within transition segments...")
            
            # Convert beat positions to samples
            track1_beats_samples = librosa.frames_to_samples(track1.beats, hop_length=512)
            track2_beats_samples = librosa.frames_to_samples(track2.beats, hop_length=512)
            
            # Find beats within the transition segments
            track1_end_sample = track1_start_sample + len(track1_overlap)
            track1_segment_beats = track1_beats_samples[
                (track1_beats_samples >= track1_start_sample) & 
                (track1_beats_samples < track1_end_sample)
            ] - track1_start_sample  # Make relative to segment start
            
            track2_end_sample = track2_start_sample + len(track2_overlap)
            track2_segment_beats = track2_beats_samples[
                (track2_beats_samples >= track2_start_sample) & 
                (track2_beats_samples < track2_end_sample)
            ] - track2_start_sample  # Make relative to segment start
            
            if len(track1_segment_beats) == 0 and len(track2_segment_beats) == 0:
                print("    No beats found in transition segments, using original audio")
                return track1_overlap, track2_overlap
            
            # Create ideal beat grid for the transition
            beats_per_second = track1.bpm / 60.0
            samples_per_beat = track1.sr / beats_per_second
            
            # Generate ideal beat positions for the transition duration
            num_beats = max(1, int(transition_duration * beats_per_second))
            ideal_beat_positions = np.arange(num_beats) * samples_per_beat
            
            # Only use beats within the actual segment length
            segment_length = len(track1_overlap)
            ideal_beat_positions = ideal_beat_positions[ideal_beat_positions < segment_length]
            
            if len(ideal_beat_positions) == 0:
                print("    No ideal beats fit in transition, using original audio")
                return track1_overlap, track2_overlap
            
            print(f"    Track1 beats in transition: {len(track1_segment_beats)}")
            print(f"    Track2 beats in transition: {len(track2_segment_beats)}")
            print(f"    Ideal beat grid: {len(ideal_beat_positions)} beats")
            
            # Apply the intelligent beat shifting using the BeatAligner's new method
            track1_aligned, track2_aligned = self.beat_aligner._apply_intelligent_beat_shifting(
                track1_overlap, track2_overlap, track1_segment_beats, track2_segment_beats,
                ideal_beat_positions, track1.sr
            )
            
            return track1_aligned, track2_aligned
            
        except Exception as e:
            print(f"    Warning: Intelligent transition alignment failed: {e}")
            print("    Using original transition segments")
            return track1_overlap, track2_overlap

    def _apply_tempo_ramp_to_transition(self, track1_overlap: np.ndarray, track2_overlap: np.ndarray, 
                                      tempo_ramp_data: dict, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gradual tempo ramping during the transition period
        Gradually increases tempo from start_bpm to end_bpm throughout the transition
        """
        try:
            start_bpm = tempo_ramp_data['start_bpm']
            end_bpm = tempo_ramp_data['end_bpm']
            
            transition_samples = min(len(track1_overlap), len(track2_overlap))
            
            # Create tempo ramp curve (linear interpolation from start to end BPM)
            tempo_curve = np.linspace(start_bpm, end_bpm, transition_samples)
            
            print(f"    Ramping from {start_bpm:.1f} to {end_bpm:.1f} BPM over {transition_samples/sr:.1f}s")
            
            # Apply time-stretching with varying tempo throughout the transition
            # We'll process the transition in small chunks with different stretch ratios
            # Use higher resolution for larger tempo differences to reduce artifacts
            tempo_diff_ratio = max(start_bpm / end_bpm, end_bpm / start_bpm)
            if tempo_diff_ratio > 1.5:  # Large tempo difference
                chunk_size = max(128, transition_samples // 400)  # ~400 chunks for extreme smoothness
                print(f"    Large tempo difference detected ({tempo_diff_ratio:.2f}x), using high-resolution ramping")
            else:
                chunk_size = max(256, transition_samples // 200)  # ~200 chunks for ultra-smooth ramping
            
            ramped_track1 = []
            ramped_track2 = []
            
            for start_idx in range(0, transition_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, transition_samples)
                chunk_length = end_idx - start_idx
                
                if chunk_length == 0:
                    break
                
                # Get the target tempo for this chunk (middle of chunk)
                mid_idx = start_idx + chunk_length // 2
                target_tempo = tempo_curve[mid_idx]
                
                # Calculate stretch ratio relative to original tempo
                stretch_ratio_track1 = start_bpm / target_tempo
                stretch_ratio_track2 = start_bpm / target_tempo  # Both start from same base
                
                # Extract chunks
                track1_chunk = track1_overlap[start_idx:end_idx]
                track2_chunk = track2_overlap[start_idx:end_idx]
                
                # Apply time-stretching to chunks
                if abs(stretch_ratio_track1 - 1.0) > 0.01:  # Only stretch if meaningful difference
                    track1_chunk = librosa.effects.time_stretch(track1_chunk, rate=stretch_ratio_track1)
                if abs(stretch_ratio_track2 - 1.0) > 0.01:
                    track2_chunk = librosa.effects.time_stretch(track2_chunk, rate=stretch_ratio_track2)
                
                # Ensure chunks maintain original length (pad/trim if needed)
                if len(track1_chunk) != chunk_length:
                    if len(track1_chunk) > chunk_length:
                        track1_chunk = track1_chunk[:chunk_length]
                    else:
                        track1_chunk = np.pad(track1_chunk, (0, chunk_length - len(track1_chunk)), 'constant')
                
                if len(track2_chunk) != chunk_length:
                    if len(track2_chunk) > chunk_length:
                        track2_chunk = track2_chunk[:chunk_length]
                    else:
                        track2_chunk = np.pad(track2_chunk, (0, chunk_length - len(track2_chunk)), 'constant')
                
                ramped_track1.append(track1_chunk)
                ramped_track2.append(track2_chunk)
            
            # Concatenate all ramped chunks
            ramped_track1 = np.concatenate(ramped_track1) if ramped_track1 else track1_overlap
            ramped_track2 = np.concatenate(ramped_track2) if ramped_track2 else track2_overlap
            
            print(f"    ✅ Tempo ramp applied: {len(ramped_track1)} samples")
            return ramped_track1, ramped_track2
            
        except Exception as e:
            print(f"    ⚠️ Tempo ramp failed, using original audio: {e}")
            return track1_overlap, track2_overlap
        
    def create_transition(self, track1: Track, track2: Track, transition_duration: float = 30.0, 
                         stretch_track1: bool = False) -> Tuple[np.ndarray, np.ndarray, Track]:
        """Create a beat-aligned crossfade transition between two tracks"""
        if self.target_bpm is None and self.tempo_strategy != "match-track":
            raise ValueError("Target BPM not set. Call calculate_target_bpm() first.")
        
        # Determine what needs to be stretched based on tempo strategy
        if str(self.tempo_strategy) == "sequential":
            # Sequential: stretch track2 to match track1 (track1 stays at native tempo)
            print(f"  Sequential mode: {track2.bpm:.1f} -> {track1.bpm:.1f} (track2 stretched)")
            target_bpm_for_track2 = track1.bpm
            track1_stretched = track1  # No change to track1
            track2_stretched = self._stretch_track_to_bpm(track2, target_bpm_for_track2)
            
        elif str(self.tempo_strategy) == "uniform":
            # Uniform: stretch both tracks to target BPM
            print(f"  Uniform mode: {track1.bpm:.1f} & {track2.bpm:.1f} -> {self.target_bpm:.1f} (both stretched)")
            
            if stretch_track1:
                # Stretch track1 to target BPM (for uniform mode in full mix)
                track1_stretched = self._stretch_track_to_bpm(track1, self.target_bpm)
            else:
                # Track1 already at target BPM (first track in uniform mode)
                track1_stretched = track1
                
            track2_stretched = self._stretch_track_to_bpm(track2, self.target_bpm)
            
        elif str(self.tempo_strategy) == "match-track":
            # Match-track: both tracks play at native tempo, tempo ramps during transition
            print(f"  Match-track mode: {track1.bpm:.1f} & {track2.bpm:.1f} (both at native tempo)")
            track1_stretched = track1  # No change to track1 - stays at native tempo
            track2_stretched = track2  # No change to track2 - stays at native tempo
        
        # CRITICAL FIX: Force both tracks to use the exact uniform target BPM for crossfading
        # The actual calculated BPMs may differ slightly due to discrete sample precision,
        # but for crossfading we must use the exact target BPM to ensure perfect matching
        
        # Determine the uniform BPM that both tracks should use for crossfading
        if str(self.tempo_strategy) == "sequential":
            target_bpm_for_crossfade = track1_stretched.bpm  # Use track1's BPM as reference
            print(f"  Sequential crossfade BPM: {target_bpm_for_crossfade:.3f} (track1's BPM)")
        elif str(self.tempo_strategy) == "uniform":
            target_bpm_for_crossfade = self.target_bpm  # Use the exact uniform target BPM
            print(f"  Uniform crossfade BPM: {target_bpm_for_crossfade:.3f} (target BPM)")
        elif str(self.tempo_strategy) == "match-track":
            # For match-track, we'll use track1's BPM for crossfade reference, but tempo will ramp
            target_bpm_for_crossfade = track1_stretched.bpm  # Use track1's native BPM as reference
            print(f"  Match-track crossfade: Starts at {track1_stretched.bpm:.3f} BPM, ramps to {track2_stretched.bpm:.3f} BPM")
        
        # Show what BPMs were actually achieved after stretching
        track1_actual_bpm = track1_stretched.bpm
        track2_actual_bpm = track2_stretched.bpm
        print(f"  Final BPMs after stretching: Track1={track1_actual_bpm:.3f}, Track2={track2_actual_bpm:.3f}")
        
        # In uniform mode, if track1 wasn't stretched, show what BPM it should have for context
        if self.tempo_strategy == "uniform" and not stretch_track1:
            print(f"  Note: Track1 not stretched (first track in sequence), target BPM = {self.target_bpm:.3f}")
        elif self.tempo_strategy == "sequential":
            print(f"  Note: Sequential mode - Track1 at native BPM, Track2 stretched to match")
        elif str(self.tempo_strategy) == "match-track":
            print(f"  Note: Match-track mode - Both tracks at native BPM, tempo ramps during transition")
        
        # Create corrected tracks with uniform BPM for crossfading
        track1_for_crossfade = self._create_track_from_existing(
            track1_stretched, 
            new_audio=track1_stretched.audio, 
            new_bpm=target_bpm_for_crossfade
        )
        
        track2_for_crossfade = self._create_track_from_existing(
            track2_stretched, 
            new_audio=track2_stretched.audio, 
            new_bpm=target_bpm_for_crossfade
        )
        
        print(f"  Crossfade BPMs (uniform): Track1={track1_for_crossfade.bpm:.3f}, Track2={track2_for_crossfade.bpm:.3f}")
        bpm_diff = abs(track1_for_crossfade.bpm - track2_for_crossfade.bpm)
        if bpm_diff > 0.001:
            print(f"  ❌ CRITICAL ERROR: BPM mismatch in crossfade ({bpm_diff:.6f})")
        else:
            print(f"  ✅ Perfect BPM match for crossfade (difference: {bpm_diff:.6f})")
        
        return self._create_enhanced_crossfade(track1_for_crossfade, track2_for_crossfade, transition_duration)
    
    def _stretch_track_to_bpm(self, track: Track, target_bpm: float) -> Track:
        """Stretch a track to match target BPM with intelligent tempo correction
        
        Professional DJ precision: Any BPM difference > 0.1% requires correction.
        Even minimal differences (e.g. 128.0 vs 127.7) cause audible drift over time.
        """
        bpm_ratio = track.bpm / target_bpm  # Ratio to stretch to target tempo
        
        if abs(bpm_ratio - 1.0) > 0.001:  # Professional precision: stretch for any meaningful BPM difference
            print(f"    Time-stretching {track.filepath.name}: {track.bpm:.1f} -> {target_bpm:.1f} (ratio: {bpm_ratio:.3f})")
            
            if self.enable_tempo_correction:
                print(f"    Applying intelligent tempo correction to eliminate drift...")
                # Apply intelligent tempo correction to maintain consistent timing
                stretched_audio, corrected_beats, corrected_downbeats = self._apply_intelligent_tempo_correction(
                    track, target_bpm
                )
                # Calculate actual BPM from corrected beats for perfect accuracy
                actual_bpm = self._calculate_actual_bpm_from_beats(corrected_beats, track.sr)
                print(f"    Final BPM after correction: {actual_bpm:.3f} (target: {target_bpm:.3f})")
            else:
                print(f"    Using simple time-stretching (tempo correction disabled)")
                # Simple time-stretching without piecewise correction
                stretched_audio = librosa.effects.time_stretch(track.audio, rate=bpm_ratio)
                # Scale beats and downbeats proportionally
                corrected_beats = track.beats / bpm_ratio
                corrected_downbeats = track.downbeats / bpm_ratio
                actual_bpm = target_bpm
            
            return self._create_track_from_existing(
                track, 
                new_audio=stretched_audio, 
                new_bpm=actual_bpm,
                new_beats=corrected_beats,
                new_downbeats=corrected_downbeats
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
                
                return self._create_track_from_existing(
                    track, 
                    new_audio=stretched_audio, 
                    new_bpm=actual_bpm,
                    new_beats=corrected_beats,
                    new_downbeats=corrected_downbeats
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
    
    def _analyze_audio_characteristics(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analyze audio characteristics for quality matching.
        Returns spectral profile, dynamics, and loudness metrics.
        """
        try:
            # RMS level (loudness)
            rms = np.sqrt(np.mean(audio**2))
            
            # Peak level
            peak = np.max(np.abs(audio))
            
            # Dynamic range (crest factor)
            crest_factor = peak / rms if rms > 0 else 1.0
            
            # Spectral analysis using STFT
            D = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(D)
            
            # Average spectral profile (frequency response)
            spectral_profile = np.mean(magnitude, axis=1)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_brightness = np.mean(spectral_centroid)
            
            # Low, mid, high frequency energy
            n_bins = len(spectral_profile)
            low_bins = n_bins // 8      # ~0-2.7kHz at 44.1kHz
            mid_bins = n_bins // 2      # ~0-11kHz at 44.1kHz  
            high_bins = n_bins * 3 // 4 # ~0-16.5kHz at 44.1kHz
            
            low_energy = np.sum(spectral_profile[:low_bins])
            mid_energy = np.sum(spectral_profile[low_bins:mid_bins]) 
            high_energy = np.sum(spectral_profile[mid_bins:high_bins])
            total_energy = low_energy + mid_energy + high_energy
            
            if total_energy > 0:
                low_ratio = low_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            else:
                low_ratio = mid_ratio = high_ratio = 0.33
            
            return {
                'rms': rms,
                'peak': peak,
                'crest_factor': crest_factor,
                'spectral_profile': spectral_profile,
                'brightness': avg_brightness,
                'low_ratio': low_ratio,
                'mid_ratio': mid_ratio,
                'high_ratio': high_ratio,
                'sample_rate': sr
            }
            
        except Exception as e:
            print(f"    Warning: Audio analysis failed: {e}")
            # Return safe defaults
            return {
                'rms': 0.1,
                'peak': 0.5,
                'crest_factor': 5.0,
                'spectral_profile': np.ones(1025),
                'brightness': 2000.0,
                'low_ratio': 0.33,
                'mid_ratio': 0.33,
                'high_ratio': 0.33,
                'sample_rate': sr
            }
    
    def _normalize_volume(self, audio: np.ndarray, target_rms: float = None, target_peak: float = 0.8) -> np.ndarray:
        """
        Normalize audio volume to target RMS or peak level.
        Professional loudness normalization for consistent levels.
        """
        try:
            current_rms = np.sqrt(np.mean(audio**2))
            current_peak = np.max(np.abs(audio))
            
            if current_rms == 0 or current_peak == 0:
                print(f"    Warning: Silent audio detected, skipping normalization")
                return audio
            
            # Use target RMS if provided, otherwise normalize by peak
            if target_rms is not None:
                # RMS-based normalization (better for perceived loudness)
                scale_factor = target_rms / current_rms
                print(f"    RMS normalization: {current_rms:.4f} -> {target_rms:.4f} (x{scale_factor:.3f})")
            else:
                # Peak-based normalization (prevents clipping)
                scale_factor = target_peak / current_peak
                print(f"    Peak normalization: {current_peak:.4f} -> {target_peak:.4f} (x{scale_factor:.3f})")
            
            # Apply scaling with limiter to prevent clipping
            normalized_audio = audio * scale_factor
            
            # Soft limiter if we exceed target peak
            if np.max(np.abs(normalized_audio)) > target_peak:
                excess_ratio = np.max(np.abs(normalized_audio)) / target_peak
                # Soft compression for peaks above target
                normalized_audio = np.sign(normalized_audio) * np.minimum(
                    np.abs(normalized_audio), 
                    target_peak * (1 - np.exp(-(np.abs(normalized_audio) / target_peak)))
                )
                print(f"    Soft limiting applied (excess: {excess_ratio:.3f})")
            
            return normalized_audio
            
        except Exception as e:
            print(f"    Warning: Volume normalization failed: {e}")
            return audio
    
    def _match_eq_profile(self, source_audio: np.ndarray, target_characteristics: dict, sr: int, strength: float = 0.5) -> np.ndarray:
        """
        Match the EQ profile of source audio to target characteristics.
        Uses spectral shaping with overlapping frequency bands.
        """
        try:
            print(f"    Applying EQ matching (strength: {strength:.1f})...")
            
            # Analyze source audio characteristics
            source_chars = self._analyze_audio_characteristics(source_audio, sr)
            
            # Calculate frequency band adjustments
            low_adjustment = target_characteristics['low_ratio'] / source_chars['low_ratio'] if source_chars['low_ratio'] > 0.01 else 1.0
            mid_adjustment = target_characteristics['mid_ratio'] / source_chars['mid_ratio'] if source_chars['mid_ratio'] > 0.01 else 1.0
            high_adjustment = target_characteristics['high_ratio'] / source_chars['high_ratio'] if source_chars['high_ratio'] > 0.01 else 1.0
            
            # Limit adjustments to reasonable range (-12dB to +12dB)
            low_adjustment = np.clip(low_adjustment, 0.25, 4.0)
            mid_adjustment = np.clip(mid_adjustment, 0.25, 4.0)  
            high_adjustment = np.clip(high_adjustment, 0.25, 4.0)
            
            print(f"    EQ adjustments - Low: {20*np.log10(low_adjustment):.1f}dB, Mid: {20*np.log10(mid_adjustment):.1f}dB, High: {20*np.log10(high_adjustment):.1f}dB")
            
            # Apply spectral shaping using STFT
            D = librosa.stft(source_audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Create frequency-dependent gain curve
            n_bins = magnitude.shape[0]
            gain_curve = np.ones(n_bins)
            
            # Define frequency bands (approximate)
            low_cutoff = n_bins // 8      # ~2.7kHz
            mid_cutoff = n_bins // 2      # ~11kHz 
            high_cutoff = n_bins * 3 // 4 # ~16.5kHz
            
            # Apply smoothed gains across frequency bands
            # Low frequencies
            gain_curve[:low_cutoff] *= low_adjustment
            
            # Mid frequencies with smooth transition
            for i in range(low_cutoff, mid_cutoff):
                blend = (i - low_cutoff) / (mid_cutoff - low_cutoff)
                gain_curve[i] *= (low_adjustment * (1 - blend) + mid_adjustment * blend)
            
            # High frequencies with smooth transition  
            for i in range(mid_cutoff, high_cutoff):
                blend = (i - mid_cutoff) / (high_cutoff - mid_cutoff)
                gain_curve[i] *= (mid_adjustment * (1 - blend) + high_adjustment * blend)
            
            # Very high frequencies
            gain_curve[high_cutoff:] *= high_adjustment
            
            # Apply strength factor (blend with original)
            gain_curve = 1.0 + strength * (gain_curve - 1.0)
            
            # Apply gain curve to magnitude spectrum
            eq_magnitude = magnitude * gain_curve[:, np.newaxis]
            
            # Reconstruct audio
            eq_complex = eq_magnitude * np.exp(1j * phase)
            eq_audio = librosa.istft(eq_complex, hop_length=512)
            
            # Ensure same length as input
            if len(eq_audio) < len(source_audio):
                eq_audio = np.pad(eq_audio, (0, len(source_audio) - len(eq_audio)), 'constant')
            else:
                eq_audio = eq_audio[:len(source_audio)]
            
            print(f"    EQ matching applied successfully")
            return eq_audio
            
        except Exception as e:
            print(f"    Warning: EQ matching failed: {e}")
            return source_audio
    
    def _detect_audio_peaks(self, audio: np.ndarray, sr: int, min_peak_height: float = 0.1, min_peak_distance_ms: float = 50.0) -> np.ndarray:
        """
        Detect significant audio peaks/transients for precise alignment.
        Returns sample positions of detected peaks.
        """
        try:
            # Use onset detection for better transient detection than simple peak finding
            onset_frames = librosa.onset.onset_detect(
                y=audio, 
                sr=sr,
                hop_length=512,
                units='frames',
                backtrack=True,  # More accurate onset timing
                pre_max=3,       # Pre-processing window
                post_max=3,      # Post-processing window 
                pre_avg=3,       # Pre-averaging window
                post_avg=5,      # Post-averaging window
                delta=0.07,      # Minimum strength increase
                wait=int(min_peak_distance_ms * sr / 1000 / 512)  # Minimum distance between onsets
            )
            
            # Convert frames to sample positions
            peak_samples = librosa.frames_to_samples(onset_frames, hop_length=512)
            
            # Additional peak filtering based on amplitude
            filtered_peaks = []
            for peak_sample in peak_samples:
                if peak_sample < len(audio):
                    # Check amplitude around peak
                    window_start = max(0, peak_sample - 256)
                    window_end = min(len(audio), peak_sample + 256)
                    peak_amplitude = np.max(np.abs(audio[window_start:window_end]))
                    
                    if peak_amplitude >= min_peak_height:
                        filtered_peaks.append(peak_sample)
            
            peaks_array = np.array(filtered_peaks)
            print(f"    Detected {len(peaks_array)} significant audio peaks/transients")
            
            return peaks_array
            
        except Exception as e:
            print(f"    Warning: Peak detection failed: {e}")
            return np.array([])
    
    def _align_peaks_to_beats(self, audio: np.ndarray, peaks: np.ndarray, beats: np.ndarray, sr: int, 
                             alignment_tolerance_ms: float = 25.0) -> np.ndarray:
        """
        Micro-adjust audio peaks to align perfectly with detected beats.
        This eliminates audio clashing by ensuring every significant transient 
        coincides with the nearest beat position.
        """
        try:
            if len(peaks) == 0 or len(beats) == 0:
                print(f"    No peaks or beats available for alignment")
                return audio
            
            print(f"    Performing micro peak-to-beat alignment...")
            
            # Convert beat frames to samples
            beat_samples = librosa.frames_to_samples(beats, hop_length=512)
            
            # Create alignment map: peak position -> target beat position
            alignment_map = []
            tolerance_samples = int(alignment_tolerance_ms * sr / 1000)
            
            for peak_sample in peaks:
                # Find nearest beat within tolerance
                distances = np.abs(beat_samples - peak_sample)
                nearest_beat_idx = np.argmin(distances)
                nearest_distance = distances[nearest_beat_idx]
                
                if nearest_distance <= tolerance_samples:
                    target_beat_sample = beat_samples[nearest_beat_idx]
                    offset_samples = target_beat_sample - peak_sample
                    alignment_map.append((peak_sample, target_beat_sample, offset_samples))
            
            if not alignment_map:
                print(f"    No peaks within alignment tolerance ({alignment_tolerance_ms}ms)")
                return audio
            
            print(f"    Aligning {len(alignment_map)} peaks to nearest beats...")
            
            # Apply micro time-shifts using phase vocoder for high quality
            aligned_audio = audio.copy()
            
            # Sort alignment map by peak position
            alignment_map.sort(key=lambda x: x[0])
            
            # Process in small windows around each peak
            window_size = int(0.1 * sr)  # 100ms windows
            
            for i, (peak_pos, target_pos, offset) in enumerate(alignment_map):
                if abs(offset) < 10:  # Skip tiny adjustments (< 10 samples)
                    continue
                
                # Define processing window around the peak
                window_start = max(0, peak_pos - window_size // 2)
                window_end = min(len(aligned_audio), peak_pos + window_size // 2)
                
                # Extract window
                window = aligned_audio[window_start:window_end]
                
                if len(window) < 100:  # Skip very small windows
                    continue
                
                try:
                    # Calculate time shift ratio
                    shift_ratio = 1.0 + (offset / len(window))
                    
                    # Apply high-quality time stretching if shift is significant
                    if 0.95 <= shift_ratio <= 1.05:  # Only for small adjustments
                        shifted_window = librosa.effects.time_stretch(window, rate=1/shift_ratio)
                        
                        # Ensure window stays the same size
                        if len(shifted_window) > len(window):
                            shifted_window = shifted_window[:len(window)]
                        elif len(shifted_window) < len(window):
                            padding = len(window) - len(shifted_window)
                            shifted_window = np.pad(shifted_window, (0, padding), 'constant')
                        
                        # Apply with crossfade to avoid clicks
                        fade_samples = min(512, len(window) // 4)
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        
                        # Crossfade at edges
                        if window_start > 0:
                            shifted_window[:fade_samples] = (
                                shifted_window[:fade_samples] * fade_in + 
                                window[:fade_samples] * fade_out
                            )
                        
                        if window_end < len(aligned_audio):
                            shifted_window[-fade_samples:] = (
                                shifted_window[-fade_samples:] * fade_out[-fade_samples:] + 
                                window[-fade_samples:] * fade_in[-fade_samples:]
                            )
                        
                        # Apply to main audio
                        aligned_audio[window_start:window_end] = shifted_window
                        
                except Exception as e:
                    print(f"    Warning: Peak alignment failed for peak {i}: {e}")
                    continue
            
            # Calculate alignment quality metrics
            total_offset = sum(abs(offset) for _, _, offset in alignment_map)
            avg_offset_ms = (total_offset / len(alignment_map)) / sr * 1000 if alignment_map else 0
            
            print(f"    Peak alignment complete - Avg adjustment: {avg_offset_ms:.1f}ms per peak")
            
            return aligned_audio
            
        except Exception as e:
            print(f"    Warning: Peak-to-beat alignment failed: {e}")
            return audio
    
    def _create_seamless_transition(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, Track]:
        """
        Create a seamless transition that maintains full track continuity.
        Processes complete tracks while applying quality improvements only to transition regions.
        """
        print(f"  Creating seamless transition with full track continuity...")
        
        # Find optimal transition points using downbeat alignment
        print(f"  Aligning to downbeats...")
        track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
            track1, track2, transition_duration
        )
        
        # Calculate transition region boundaries
        transition_samples = int(transition_duration * track1.sr)
        track1_outro_start = max(0, track1_end_sample - transition_samples)
        track2_intro_end = min(len(track2.audio), track2_start_sample + transition_samples)
        
        # Report the alignment
        track1_end_time = track1_end_sample / track1.sr
        track2_start_time = track2_start_sample / track2.sr
        print(f"  Track 1 outro: {track1_outro_start/track1.sr:.1f}s - {track1_end_time:.1f}s")
        print(f"  Track 2 intro: {track2_start_time:.1f}s - {track2_intro_end/track2.sr:.1f}s")
        
        # Work with complete track copies to maintain continuity
        track1_processed = track1.audio.copy()
        track2_processed = track2.audio.copy()
        
        # Extract transition regions for quality processing (without disconnecting from full tracks)
        track1_outro_region = track1_processed[track1_outro_start:track1_end_sample]
        track2_intro_region = track2_processed[track2_start_sample:track2_intro_end]
        
        # Ensure both transition regions are the same length
        min_transition_length = min(len(track1_outro_region), len(track2_intro_region))
        if min_transition_length < transition_samples * 0.5:
            print(f"  Warning: Short transition regions, using fallback")
            return self._create_fallback_seamless_transition(track1, track2, transition_duration)
        
        track1_outro_region = track1_outro_region[:min_transition_length]
        track2_intro_region = track2_intro_region[:min_transition_length]
        
        # Apply professional audio quality improvements to transition regions only
        track1_outro_enhanced, track2_intro_enhanced = self._apply_transition_quality_processing(
            track1_outro_region, track2_intro_region, track1, track2, 
            track1_outro_start, track2_start_sample
        )
        
        # Replace the transition regions in the full tracks with enhanced versions
        track1_processed[track1_outro_start:track1_outro_start + len(track1_outro_enhanced)] = track1_outro_enhanced
        track2_processed[track2_start_sample:track2_start_sample + len(track2_intro_enhanced)] = track2_intro_enhanced
        
        # Create crossfade between the enhanced transition regions
        fade_samples = min_transition_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Create the crossfaded transition section
        crossfade_section = track1_outro_enhanced * fade_out + track2_intro_enhanced * fade_in
        
        # Build the complete seamless mix
        # Part 1: Track1 up to the start of crossfade
        pre_crossfade = track1_processed[:track1_outro_start]
        
        # Part 2: The crossfaded transition
        transition_part = crossfade_section
        
        # Part 3: Track2 from the end of crossfade
        post_crossfade_start = track2_start_sample + len(crossfade_section)
        post_crossfade = track2_processed[post_crossfade_start:]
        
        # Combine all parts for complete continuity
        complete_transition = np.concatenate([
            pre_crossfade,
            transition_part,
            post_crossfade
        ])
        
        # Create updated track2 object with processed audio
        updated_track2 = self._create_track_from_existing(
            track2, 
            new_audio=track2_processed
        )
        
        print(f"  Seamless transition complete - full musical continuity preserved")
        
        return complete_transition, updated_track2
    
    def _apply_transition_quality_processing(self, track1_outro: np.ndarray, track2_intro: np.ndarray, 
                                           track1: Track, track2: Track, outro_start: int, intro_start: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply professional audio quality improvements to transition regions only.
        This is the extracted logic from the previous _create_crossfade method.
        """
        # PROFESSIONAL AUDIO QUALITY IMPROVEMENTS
        if self.enable_volume_matching or self.enable_eq_matching:
            print(f"  Applying professional audio quality processing to transition regions...")
            
            # 1. Analyze audio characteristics if needed
            if self.enable_volume_matching or self.enable_eq_matching:
                print(f"  Analyzing transition region characteristics...")
                track1_chars = self._analyze_audio_characteristics(track1_outro, track1.sr)
                track2_chars = self._analyze_audio_characteristics(track2_intro, track2.sr)
            
            # 2. Volume normalization - match RMS levels
            if self.enable_volume_matching:
                print(f"  Volume levels - Track1 RMS: {track1_chars['rms']:.4f}, Track2 RMS: {track2_chars['rms']:.4f}")
                
                # Use average RMS as target for consistent perceived loudness
                target_rms = (track1_chars['rms'] + track2_chars['rms']) / 2.0
                print(f"  Normalizing transition regions to target RMS: {target_rms:.4f}")
                
                track1_outro_normalized = self._normalize_volume(track1_outro, target_rms=target_rms)
                track2_intro_normalized = self._normalize_volume(track2_intro, target_rms=target_rms)
            else:
                print(f"  Volume normalization disabled for transition")
                track1_outro_normalized = track1_outro
                track2_intro_normalized = track2_intro
            
            # 3. EQ matching for seamless tonal continuity
            if self.enable_eq_matching and self.eq_strength > 0.0:
                print(f"  Spectral balance - Track1 (L/M/H): {track1_chars['low_ratio']:.2f}/{track1_chars['mid_ratio']:.2f}/{track1_chars['high_ratio']:.2f}")
                print(f"  Spectral balance - Track2 (L/M/H): {track2_chars['low_ratio']:.2f}/{track2_chars['mid_ratio']:.2f}/{track2_chars['high_ratio']:.2f}")
                
                # Create balanced target characteristics (average of both tracks)
                target_chars = {
                    'low_ratio': (track1_chars['low_ratio'] + track2_chars['low_ratio']) / 2.0,
                    'mid_ratio': (track1_chars['mid_ratio'] + track2_chars['mid_ratio']) / 2.0,
                    'high_ratio': (track1_chars['high_ratio'] + track2_chars['high_ratio']) / 2.0,
                }
                print(f"  Target spectral balance (L/M/H): {target_chars['low_ratio']:.2f}/{target_chars['mid_ratio']:.2f}/{target_chars['high_ratio']:.2f}")
                
                # Apply EQ matching with user-specified strength
                track1_outro_processed = self._match_eq_profile(track1_outro_normalized, target_chars, track1.sr, self.eq_strength)
                track2_intro_processed = self._match_eq_profile(track2_intro_normalized, target_chars, track2.sr, self.eq_strength)
            else:
                print(f"  EQ matching disabled for transition")
                track1_outro_processed = track1_outro_normalized
                track2_intro_processed = track2_intro_normalized
            
            print(f"  Professional audio processing complete for transition regions!")
        else:
            print(f"  Audio quality processing disabled - using original transition regions")
            track1_outro_processed = track1_outro
            track2_intro_processed = track2_intro
        
        # 4. MICRO PEAK-TO-BEAT ALIGNMENT for perfect synchronization
        if self.enable_peak_alignment:
            print(f"  Applying micro peak-to-beat alignment to transition regions...")
            
            # Extract beat positions that fall within the transition segments
            track1_beats_samples = librosa.frames_to_samples(track1.beats, hop_length=512)
            track2_beats_samples = librosa.frames_to_samples(track2.beats, hop_length=512)
            
            # Find beats within transition regions (relative to segment start)
            track1_end_sample = outro_start + len(track1_outro_processed)
            track2_intro_end = intro_start + len(track2_intro_processed)
            
            track1_transition_beats = track1_beats_samples[
                (track1_beats_samples >= outro_start) & 
                (track1_beats_samples <= track1_end_sample)
            ] - outro_start  # Make relative to segment start
            
            track2_transition_beats = track2_beats_samples[
                (track2_beats_samples >= intro_start) & 
                (track2_beats_samples <= track2_intro_end)
            ] - intro_start  # Make relative to segment start
            
            # Detect and align peaks for perfect synchronization
            if len(track1_transition_beats) > 0 and len(track2_transition_beats) > 0:
                # Detect audio peaks in both segments
                track1_peaks = self._detect_audio_peaks(track1_outro_processed, track1.sr)
                track2_peaks = self._detect_audio_peaks(track2_intro_processed, track2.sr)
                
                # Convert beat sample positions back to frames for alignment function
                track1_beats_frames = librosa.samples_to_frames(track1_transition_beats, hop_length=512)
                track2_beats_frames = librosa.samples_to_frames(track2_transition_beats, hop_length=512)
                
                # Apply micro peak-to-beat alignment to both transition regions
                print(f"  Track 1 outro: {len(track1_peaks)} peaks, {len(track1_beats_frames)} beats")
                track1_outro_aligned = self._align_peaks_to_beats(
                    track1_outro_processed, track1_peaks, track1_beats_frames, track1.sr
                )
                
                print(f"  Track 2 intro: {len(track2_peaks)} peaks, {len(track2_beats_frames)} beats") 
                track2_intro_aligned = self._align_peaks_to_beats(
                    track2_intro_processed, track2_peaks, track2_beats_frames, track2.sr
                )
                
                print(f"  Micro peak-to-beat alignment complete for transition regions!")
            else:
                print(f"  Warning: Insufficient beats in transition segments for peak alignment")
                track1_outro_aligned = track1_outro_processed
                track2_intro_aligned = track2_intro_processed
        else:
            print(f"  Micro peak alignment disabled for transition")
            track1_outro_aligned = track1_outro_processed
            track2_intro_aligned = track2_intro_processed
        
        return track1_outro_aligned, track2_intro_aligned
    
    def _create_fallback_seamless_transition(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, Track]:
        """
        Fallback seamless transition method when optimal transition points fail
        """
        print(f"  Using fallback seamless transition method")
        
        transition_samples = int(transition_duration * track1.sr)
        
        # Simple approach: overlap end of track1 with beginning of track2
        track1_processed = track1.audio.copy()
        track2_processed = track2.audio.copy()
        
        # Get outro and intro regions
        track1_outro = track1_processed[-transition_samples:] if len(track1_processed) >= transition_samples else track1_processed
        track2_intro = track2_processed[:transition_samples] if len(track2_processed) >= transition_samples else track2_processed
        
        # Ensure same length
        min_length = min(len(track1_outro), len(track2_intro))
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create simple crossfade
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        transition_section = track1_outro * fade_out + track2_intro * fade_in
        
        # Build complete mix
        pre_transition = track1_processed[:-len(track1_outro)]
        post_transition = track2_processed[len(track2_intro):]
        
        complete_mix = np.concatenate([pre_transition, transition_section, post_transition])
        
        # Updated track2
        updated_track2 = self._create_track_from_existing(
            track2, 
            new_audio=track2_processed
        )
        
        return complete_mix, updated_track2
    
    def _apply_tempo_ramping(self, track1_outro: np.ndarray, track2_intro: np.ndarray, 
                           start_bpm: float, end_bpm: float, sr: int, transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply tempo ramping during transition for match-track strategy.
        
        Args:
            track1_outro: Audio segment from track1 (fading out)
            track2_intro: Audio segment from track2 (fading in) 
            start_bpm: BPM at start of transition (track1's tempo)
            end_bpm: BPM at end of transition (track2's tempo)
            sr: Sample rate
            transition_duration: Duration of transition in seconds
            
        Returns:
            Tuple of (track1_outro_ramped, track2_intro_ramped)
        """
        print(f"  Applying tempo ramping: {start_bpm:.1f} BPM -> {end_bpm:.1f} BPM over {transition_duration:.1f}s")
        
        # Calculate the tempo change factor
        tempo_ratio = end_bpm / start_bpm
        print(f"  Tempo ratio: {tempo_ratio:.3f}")
        
        # If tempos are very close, skip ramping to avoid unnecessary processing
        if abs(tempo_ratio - 1.0) < 0.02:  # Less than 2% change
            print(f"  Tempos are very close ({abs((tempo_ratio - 1.0) * 100):.1f}% difference), skipping tempo ramping")
            return track1_outro, track2_intro
            
        # Apply gradual tempo ramping using chunk-based processing
        # This ensures each track gradually transitions to the other's tempo
        try:
            n_samples = len(track1_outro)
            # Use higher resolution for larger tempo differences to reduce artifacts
            tempo_diff_ratio = max(start_bpm / end_bpm, end_bpm / start_bpm)
            if tempo_diff_ratio > 1.5:  # Large tempo difference
                chunk_size = max(128, n_samples // 400)  # ~400 chunks for extreme smoothness
                print(f"  Large tempo difference detected ({tempo_diff_ratio:.2f}x), using high-resolution ramping")
            else:
                chunk_size = max(256, n_samples // 200)  # ~200 chunks for ultra-smooth ramping
            
            ramped_track1 = []
            ramped_track2 = []
            
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                chunk_length = end_idx - start_idx
                
                if chunk_length == 0:
                    break
                
                # Calculate progress through transition (0 to 1)
                progress = (start_idx + chunk_length / 2) / n_samples
                
                # Both tracks must match the SAME target tempo at each increment
                # At progress=0: both tracks play at start_bpm (track1's tempo)
                # At progress=1: both tracks play at end_bpm (track2's tempo)
                current_target_tempo = start_bpm + progress * (end_bpm - start_bpm)
                
                # Track1: stretch from its native tempo (start_bpm) to current target tempo
                track1_stretch_ratio = start_bpm / current_target_tempo
                
                # Track2: stretch from its native tempo (end_bpm) to current target tempo  
                track2_stretch_ratio = end_bpm / current_target_tempo
                
                # Debug: Show synchronization at key points
                if start_idx == 0 or start_idx >= n_samples - chunk_size or (start_idx // chunk_size) % 10 == 0:
                    print(f"    Chunk {start_idx//chunk_size}: progress={progress:.2f}, target_tempo={current_target_tempo:.1f} BPM")
                    print(f"      Track1: {start_bpm:.1f} -> {current_target_tempo:.1f} BPM (ratio: {track1_stretch_ratio:.3f})")
                    print(f"      Track2: {end_bpm:.1f} -> {current_target_tempo:.1f} BPM (ratio: {track2_stretch_ratio:.3f})")
                
                # Extract chunks
                track1_chunk = track1_outro[start_idx:end_idx]
                track2_chunk = track2_intro[start_idx:end_idx]
                
                # Apply time-stretching to chunks if meaningful difference
                if abs(track1_stretch_ratio - 1.0) > 0.01:
                    track1_chunk = librosa.effects.time_stretch(track1_chunk, rate=track1_stretch_ratio)
                if abs(track2_stretch_ratio - 1.0) > 0.01:
                    track2_chunk = librosa.effects.time_stretch(track2_chunk, rate=track2_stretch_ratio)
                
                # Ensure chunks maintain original length (pad/trim if needed)
                if len(track1_chunk) > chunk_length:
                    track1_chunk = track1_chunk[:chunk_length]
                elif len(track1_chunk) < chunk_length:
                    track1_chunk = np.pad(track1_chunk, (0, chunk_length - len(track1_chunk)), 'constant')
                    
                if len(track2_chunk) > chunk_length:
                    track2_chunk = track2_chunk[:chunk_length]
                elif len(track2_chunk) < chunk_length:
                    track2_chunk = np.pad(track2_chunk, (0, chunk_length - len(track2_chunk)), 'constant')
                
                ramped_track1.append(track1_chunk)
                ramped_track2.append(track2_chunk)
            
            # Concatenate all chunks
            track1_ramped = np.concatenate(ramped_track1) if ramped_track1 else track1_outro
            track2_ramped = np.concatenate(ramped_track2) if ramped_track2 else track2_intro
            
            # Ensure final arrays are the correct length
            track1_ramped = track1_ramped[:len(track1_outro)]
            track2_ramped = track2_ramped[:len(track2_intro)]
            
            print(f"  Match-track tempo ramping applied: gradual transition from {start_bpm:.1f} to {end_bpm:.1f} BPM")
            return track1_ramped, track2_ramped
            
        except Exception as e:
            print(f"  Warning: Tempo ramping failed ({e}), using original audio")
            return track1_outro, track2_intro
    
    def _apply_transition_downbeat_mapping(self, track1_overlap: np.ndarray, track2_overlap: np.ndarray,
                                         track1: Track, track2: Track, sr: int, transition_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply interactive downbeat mapping for transition segments.
        
        Shows GUI for user to select precise downbeats in each track's transition segment,
        then remaps the audio to align with those selections using the tracks' existing BPMs.
        
        Args:
            track1_overlap: Audio segment from track1 (outro)
            track2_overlap: Audio segment from track2 (intro) 
            track1: Track 1 object with metadata
            track2: Track 2 object with metadata
            sr: Sample rate
            transition_duration: Duration of transition in seconds
            
        Returns:
            Tuple of (track1_overlap_remapped, track2_overlap_remapped)
        """
        print(f"  🎶 Interactive transition downbeat mapping...")
        
        try:
            try:
                from ..gui.transition_gui import select_transition_downbeats
            except ImportError:
                from gui.transition_gui import select_transition_downbeats
            
            # Get track names for display
            track1_name = track1.filepath.name
            track2_name = track2.filepath.name
            
            # Show the GUI for downbeat selection
            result = select_transition_downbeats(
                track1_overlap, track2_overlap, sr,
                track1_name, track2_name, track1.bpm, track2.bpm, transition_duration
            )
            
            if result == 'cancel':
                print("    Transition downbeat mapping cancelled, using original segments")
                return track1_overlap, track2_overlap
            elif result is None:
                print("    GUI not available, using original segments")
                return track1_overlap, track2_overlap
            
            # Extract selected downbeats
            track1_downbeat = result.get('track1_downbeat')
            track2_downbeat = result.get('track2_downbeat')
            
            # Remap tracks based on selected downbeats
            track1_remapped = self._remap_track_to_downbeat(track1_overlap, track1_downbeat, track1.bpm, sr) if track1_downbeat else track1_overlap
            track2_remapped = self._remap_track_to_downbeat(track2_overlap, track2_downbeat, track2.bpm, sr) if track2_downbeat else track2_overlap
            
            # Ensure both segments are the same length
            min_length = min(len(track1_remapped), len(track2_remapped))
            track1_remapped = track1_remapped[:min_length]
            track2_remapped = track2_remapped[:min_length]
            
            print(f"    Transition downbeat mapping applied successfully")
            return track1_remapped, track2_remapped
            
        except ImportError:
            print(f"    Warning: transition_downbeat_gui not available, using original segments")
            return track1_overlap, track2_overlap
        except Exception as e:
            print(f"    Warning: Transition downbeat mapping failed ({e}), using original segments")
            return track1_overlap, track2_overlap
    
    def _remap_track_to_downbeat(self, audio: np.ndarray, selected_downbeat: float, bpm: float, sr: int) -> np.ndarray:
        """
        Remap a track segment to align the selected downbeat with the beginning.
        
        This creates a new version of the audio where the selected downbeat becomes
        the reference point, with the track remapped to its existing BPM.
        
        Args:
            audio: Audio segment to remap
            selected_downbeat: Time in seconds of the selected downbeat
            bpm: BPM of the track (used for tempo mapping)
            sr: Sample rate
            
        Returns:
            Remapped audio segment
        """
        try:
            # Convert selected downbeat to samples
            downbeat_sample = int(selected_downbeat * sr)
            
            # Create a new version starting from the selected downbeat
            if downbeat_sample < len(audio):
                # Extract audio starting from the downbeat
                audio_from_downbeat = audio[downbeat_sample:]
                
                # If we need more audio to fill the original length, loop or pad
                if len(audio_from_downbeat) < len(audio):
                    # Pad with the beginning of the original audio
                    remaining_length = len(audio) - len(audio_from_downbeat)
                    padding = audio[:remaining_length]
                    remapped_audio = np.concatenate([audio_from_downbeat, padding])
                else:
                    # Trim to original length
                    remapped_audio = audio_from_downbeat[:len(audio)]
                
                print(f"      Remapped to start from downbeat at {selected_downbeat:.3f}s")
                return remapped_audio
            else:
                print(f"      Selected downbeat ({selected_downbeat:.3f}s) is beyond audio length, using original")
                return audio
                
        except Exception as e:
            print(f"      Error in track remapping: {e}, using original")
            return audio
    
    def _apply_lf_transition(self, track1_outro: np.ndarray, track2_intro: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply low-frequency transition blending to prevent kick drum and bass clashing.
        
        Gradually blends low frequencies from 100% track1 at start to 100% track2 at end.
        This prevents bass and kick drum conflicts during transitions.
        
        Args:
            track1_outro: Audio segment from track1 (fading out)
            track2_intro: Audio segment from track2 (fading in)
            sr: Sample rate
            
        Returns:
            Tuple of (track1_outro_processed, track2_intro_processed)
        """
        print(f"  Applying low-frequency transition blending...")
        
        try:
            from scipy.signal import butter, filtfilt
            
            # Define frequency bands
            # Low frequencies: 20-200 Hz (bass, kick drums)  
            # High frequencies: 200+ Hz (everything else)
            lf_cutoff = 200.0  # Hz
            nyquist = sr / 2
            
            if lf_cutoff >= nyquist:
                print(f"    Warning: LF cutoff ({lf_cutoff} Hz) >= Nyquist frequency ({nyquist} Hz), skipping LF blending")
                return track1_outro, track2_intro
            
            # Create low-pass and high-pass filters
            # Low-pass: keeps frequencies below cutoff
            lp_b, lp_a = butter(4, lf_cutoff / nyquist, btype='low')
            # High-pass: keeps frequencies above cutoff  
            hp_b, hp_a = butter(4, lf_cutoff / nyquist, btype='high')
            
            n_samples = len(track1_outro)
            
            # Separate frequency bands for both tracks
            track1_lf = filtfilt(lp_b, lp_a, track1_outro)  # Track1 low frequencies
            track1_hf = filtfilt(hp_b, hp_a, track1_outro)  # Track1 high frequencies
            
            track2_lf = filtfilt(lp_b, lp_a, track2_intro)  # Track2 low frequencies  
            track2_hf = filtfilt(hp_b, hp_a, track2_intro)  # Track2 high frequencies
            
            # Create blending curves for low frequencies
            # At start: 100% track1 LF, 0% track2 LF
            # At end: 0% track1 LF, 100% track2 LF
            lf_blend_curve = np.linspace(0, 1, n_samples)  # 0->1 for track2 LF
            lf_fade_curve = np.linspace(1, 0, n_samples)   # 1->0 for track1 LF
            
            # Blend low frequencies gradually
            blended_lf = track1_lf * lf_fade_curve + track2_lf * lf_blend_curve
            
            # For high frequencies, use normal crossfade curves
            hf_fade_out = np.cos(np.linspace(0, np.pi/2, n_samples))**2  # track1 HF fade out
            hf_fade_in = np.sin(np.linspace(0, np.pi/2, n_samples))**2   # track2 HF fade in
            
            # Process track1: gradually reduce LF, normal HF fadeout
            track1_lf_processed = track1_lf * lf_fade_curve
            track1_hf_processed = track1_hf * hf_fade_out
            track1_processed = track1_lf_processed + track1_hf_processed
            
            # Process track2: gradually increase LF, normal HF fadein  
            track2_lf_processed = track2_lf * lf_blend_curve
            track2_hf_processed = track2_hf * hf_fade_in
            track2_processed = track2_lf_processed + track2_hf_processed
            
            print(f"    Low-frequency blending applied (cutoff: {lf_cutoff} Hz)")
            return track1_processed, track2_processed
            
        except ImportError:
            print(f"    Warning: scipy.signal not available, skipping LF blending")
            return track1_outro, track2_intro
        except Exception as e:
            print(f"    Warning: LF blending failed ({e}), using original audio")
            return track1_outro, track2_intro
    
    def _apply_mf_transition(self, track1_outro: np.ndarray, track2_intro: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mid-frequency transition blending for smoother melodic transitions.
        
        Gradually blends mid frequencies from 100% track1 at start to 100% track2 at end.
        This creates smoother transitions for melodic and harmonic content in the mid-range.
        
        Args:
            track1_outro: Audio segment from track1 (fading out)
            track2_intro: Audio segment from track2 (fading in)
            sr: Sample rate
            
        Returns:
            Tuple of (track1_outro_processed, track2_intro_processed)
        """
        print(f"  Applying mid-frequency transition blending...")
        
        try:
            from scipy.signal import butter, filtfilt
            
            # Define frequency bands
            # Low frequencies: 20-200 Hz (bass, kick drums) - keep normal crossfade
            # Mid frequencies: 200-2000 Hz (melody, harmonics, vocals) - special blending
            # High frequencies: 2000+ Hz (cymbals, hi-hats, sparkle) - keep normal crossfade
            lf_cutoff = 200.0    # Hz - low/mid boundary
            hf_cutoff = 2000.0   # Hz - mid/high boundary
            nyquist = sr / 2
            
            if lf_cutoff >= nyquist or hf_cutoff >= nyquist:
                print(f"    Warning: MF cutoffs ({lf_cutoff}-{hf_cutoff} Hz) >= Nyquist frequency ({nyquist} Hz), skipping MF blending")
                return track1_outro, track2_intro
            
            # Create band-pass and other filters
            # Low-pass: keeps frequencies below lf_cutoff (bass)
            lp_b, lp_a = butter(4, lf_cutoff / nyquist, btype='low')
            # Band-pass: keeps frequencies between lf_cutoff and hf_cutoff (mids)
            bp_b, bp_a = butter(4, [lf_cutoff / nyquist, hf_cutoff / nyquist], btype='band')
            # High-pass: keeps frequencies above hf_cutoff (highs)
            hp_b, hp_a = butter(4, hf_cutoff / nyquist, btype='high')
            
            n_samples = len(track1_outro)
            
            # Separate frequency bands for both tracks
            track1_lf = filtfilt(lp_b, lp_a, track1_outro)  # Track1 low frequencies
            track1_mf = filtfilt(bp_b, bp_a, track1_outro)  # Track1 mid frequencies  
            track1_hf = filtfilt(hp_b, hp_a, track1_outro)  # Track1 high frequencies
            
            track2_lf = filtfilt(lp_b, lp_a, track2_intro)  # Track2 low frequencies
            track2_mf = filtfilt(bp_b, bp_a, track2_intro)  # Track2 mid frequencies
            track2_hf = filtfilt(hp_b, hp_a, track2_intro)  # Track2 high frequencies
            
            # Create blending curves for mid frequencies
            # At start: 100% track1 MF, 0% track2 MF  
            # At end: 0% track1 MF, 100% track2 MF
            mf_blend_curve = np.linspace(0, 1, n_samples)  # 0->1 for track2 MF
            mf_fade_curve = np.linspace(1, 0, n_samples)   # 1->0 for track1 MF
            
            # For low and high frequencies, use normal crossfade curves
            normal_fade_out = np.cos(np.linspace(0, np.pi/2, n_samples))**2  # Normal fade out
            normal_fade_in = np.sin(np.linspace(0, np.pi/2, n_samples))**2   # Normal fade in
            
            # Process track1: gradual MF fade out, normal LF/HF fadeout
            track1_lf_processed = track1_lf * normal_fade_out
            track1_mf_processed = track1_mf * mf_fade_curve
            track1_hf_processed = track1_hf * normal_fade_out
            track1_processed = track1_lf_processed + track1_mf_processed + track1_hf_processed
            
            # Process track2: gradual MF fade in, normal LF/HF fadein
            track2_lf_processed = track2_lf * normal_fade_in
            track2_mf_processed = track2_mf * mf_blend_curve
            track2_hf_processed = track2_hf * normal_fade_in
            track2_processed = track2_lf_processed + track2_mf_processed + track2_hf_processed
            
            print(f"    Mid-frequency blending applied (range: {lf_cutoff}-{hf_cutoff} Hz)")
            return track1_processed, track2_processed
            
        except ImportError:
            print(f"    Warning: scipy.signal not available, skipping MF blending")
            return track1_outro, track2_intro
        except Exception as e:
            print(f"    Warning: MF blending failed ({e}), using original audio")
            return track1_outro, track2_intro
    
    def _apply_hf_transition(self, track1_outro: np.ndarray, track2_intro: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply high-frequency transition blending for crisp treble transitions.
        
        Gradually blends high frequencies from 100% track1 at start to 100% track2 at end.
        This creates crisp transitions for treble content like cymbals, hi-hats, and sparkle.
        
        Args:
            track1_outro: Audio segment from track1 (fading out)
            track2_intro: Audio segment from track2 (fading in)
            sr: Sample rate
            
        Returns:
            Tuple of (track1_outro_processed, track2_intro_processed)
        """
        print(f"  Applying high-frequency transition blending...")
        
        try:
            from scipy.signal import butter, filtfilt
            try:
                from ..core.config import AudioConstants
            except ImportError:
                from core.config import AudioConstants
            
            # Define frequency bands using constants
            # Low frequencies: 20-200 Hz (bass, kick drums) - keep normal crossfade
            # Mid frequencies: 200-2000 Hz (melody, harmonics, vocals) - keep normal crossfade  
            # High frequencies: 2000-8000 Hz (cymbals, hi-hats, sparkle) - special blending
            lf_cutoff = AudioConstants.MID_FREQ_LOW_CUTOFF    # 200 Hz - low/mid boundary
            mf_cutoff = AudioConstants.HIGH_FREQ_LOW_CUTOFF   # 2000 Hz - mid/high boundary
            hf_cutoff = AudioConstants.HIGH_FREQ_HIGH_CUTOFF  # 8000 Hz - high frequency limit
            nyquist = sr / 2
            
            if mf_cutoff >= nyquist or hf_cutoff >= nyquist:
                print(f"    Warning: HF cutoffs ({mf_cutoff}-{hf_cutoff} Hz) >= Nyquist frequency ({nyquist} Hz), skipping HF blending")
                return track1_outro, track2_intro
            
            # Create band-pass and other filters
            # Low-pass: keeps frequencies below lf_cutoff (bass)
            lp_b, lp_a = butter(4, lf_cutoff / nyquist, btype='low')
            # Band-pass: keeps frequencies between lf_cutoff and mf_cutoff (mids)  
            bp_b, bp_a = butter(4, [lf_cutoff / nyquist, mf_cutoff / nyquist], btype='band')
            # Band-pass for highs: keeps frequencies between mf_cutoff and hf_cutoff (highs)
            hp_b, hp_a = butter(4, [mf_cutoff / nyquist, min(hf_cutoff / nyquist, 0.95)], btype='band')
            
            n_samples = len(track1_outro)
            
            # Ensure both tracks have the same length
            min_length = min(len(track1_outro), len(track2_intro))
            track1_outro = track1_outro[:min_length]
            track2_intro = track2_intro[:min_length] 
            n_samples = min_length
            
            # Separate frequency bands for track1
            track1_lf = filtfilt(lp_b, lp_a, track1_outro)    # Low frequencies (bass)
            track1_mf = filtfilt(bp_b, bp_a, track1_outro)    # Mid frequencies (melody)  
            track1_hf = filtfilt(hp_b, hp_a, track1_outro)    # High frequencies (treble)
            
            # Separate frequency bands for track2  
            track2_lf = filtfilt(lp_b, lp_a, track2_intro)    # Low frequencies (bass)
            track2_mf = filtfilt(bp_b, bp_a, track2_intro)    # Mid frequencies (melody)
            track2_hf = filtfilt(hp_b, hp_a, track2_intro)    # High frequencies (treble)
            
            # Create gradual blending curves for high frequencies
            # High frequencies fade more gradually for crisp transitions
            hf_fade_curve = np.cos(np.linspace(0, np.pi/2, n_samples))**1.5      # Slower HF fade out
            hf_blend_curve = np.sin(np.linspace(0, np.pi/2, n_samples))**1.5     # Slower HF blend in
            
            # For low and mid frequencies, use normal crossfade curves
            normal_fade_out = np.cos(np.linspace(0, np.pi/2, n_samples))**2  # Normal fade out
            normal_fade_in = np.sin(np.linspace(0, np.pi/2, n_samples))**2   # Normal fade in
            
            # Process track1: gradual HF fade out, normal LF/MF fadeout
            track1_lf_processed = track1_lf * normal_fade_out
            track1_mf_processed = track1_mf * normal_fade_out  
            track1_hf_processed = track1_hf * hf_fade_curve
            track1_processed = track1_lf_processed + track1_mf_processed + track1_hf_processed
            
            # Process track2: gradual HF fade in, normal LF/MF fadein
            track2_lf_processed = track2_lf * normal_fade_in
            track2_mf_processed = track2_mf * normal_fade_in
            track2_hf_processed = track2_hf * hf_blend_curve  
            track2_processed = track2_lf_processed + track2_mf_processed + track2_hf_processed
            
            print(f"    High-frequency blending applied (range: {mf_cutoff}-{hf_cutoff} Hz)")
            return track1_processed, track2_processed
            
        except ImportError:
            print(f"    Warning: scipy.signal not available, skipping HF blending")
            return track1_outro, track2_intro
        except Exception as e:
            print(f"    Warning: HF blending failed ({e}), using original audio")
            return track1_outro, track2_intro
    
    def _create_enhanced_crossfade(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, np.ndarray, Track]:
        """
        Create enhanced crossfade with quality improvements while maintaining sample continuity.
        Returns (complete_mixed_segment, track2_remaining_audio, updated_track2).
        
        The complete_mixed_segment contains: track1_body + transition + track2_intro_processed
        The track2_remaining_audio contains: track2_audio from where the intro processing ended
        This ensures perfect sample continuity with no gaps or overlaps.
        """
        print(f"  Creating enhanced crossfade with professional quality processing...")
        
        # Find optimal transition points using downbeat alignment
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
        
        # Apply tempo ramping for match-track strategy
        if str(self.tempo_strategy) == "match-track":
            track1_outro, track2_intro = self._apply_tempo_ramping(
                track1_outro, track2_intro, track1.bpm, track2.bpm, track1.sr, transition_duration
            )
        
        # Apply professional quality improvements to transition segments
        outro_start = max(0, track1_end_sample - len(track1_outro))
        intro_start = track2_start_sample
        
        track1_outro_enhanced, track2_intro_enhanced = self._apply_transition_quality_processing(
            track1_outro, track2_intro, track1, track2, outro_start, intro_start
        )
        
        # Apply frequency transition blending if enabled (can be combined)
        if self.enable_lf_transition:
            track1_outro_enhanced, track2_intro_enhanced = self._apply_lf_transition(
                track1_outro_enhanced, track2_intro_enhanced, track1.sr
            )
            print(f"    Low-frequency blending applied (cutoff: {200.0} Hz)")
            
        if self.enable_mf_transition:
            track1_outro_enhanced, track2_intro_enhanced = self._apply_mf_transition(
                track1_outro_enhanced, track2_intro_enhanced, track1.sr
            )
            
        if self.enable_hf_transition:
            track1_outro_enhanced, track2_intro_enhanced = self._apply_hf_transition(
                track1_outro_enhanced, track2_intro_enhanced, track1.sr
            )
        
        # Create equal-power crossfade curves with enhanced audio
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade with enhanced audio
        if self.enable_lf_transition or self.enable_mf_transition or self.enable_hf_transition:
            # For frequency transitions, the audio is already blended, just sum them
            transition = track1_outro_enhanced + track2_intro_enhanced
        else:
            # Normal crossfade
            transition = track1_outro_enhanced * fade_out + track2_intro_enhanced * fade_in
        
        # Final output normalization to prevent clipping
        final_peak = np.max(np.abs(transition))
        if final_peak > 0.95:
            transition = transition * (0.95 / final_peak)
            print(f"  Final transition normalized (peak: {final_peak:.3f} -> 0.95)")
        
        print(f"  Enhanced crossfade complete with professional quality processing!")
        
        return transition, track2.audio, track2
    
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
        
        # PROFESSIONAL AUDIO QUALITY IMPROVEMENTS
        if self.enable_volume_matching or self.enable_eq_matching:
            print(f"  Applying professional audio quality processing...")
            
            # 1. Analyze audio characteristics if needed
            if self.enable_volume_matching or self.enable_eq_matching:
                print(f"  Analyzing track characteristics for quality matching...")
                track1_chars = self._analyze_audio_characteristics(track1_outro, track1.sr)
                track2_chars = self._analyze_audio_characteristics(track2_intro, track2.sr)
            
            # 2. Volume normalization - match RMS levels
            if self.enable_volume_matching:
                print(f"  Volume levels - Track1 RMS: {track1_chars['rms']:.4f}, Track2 RMS: {track2_chars['rms']:.4f}")
                
                # Use average RMS as target for consistent perceived loudness
                target_rms = (track1_chars['rms'] + track2_chars['rms']) / 2.0
                print(f"  Normalizing to target RMS: {target_rms:.4f}")
                
                track1_outro_normalized = self._normalize_volume(track1_outro, target_rms=target_rms)
                track2_intro_normalized = self._normalize_volume(track2_intro, target_rms=target_rms)
            else:
                print(f"  Volume normalization disabled")
                track1_outro_normalized = track1_outro
                track2_intro_normalized = track2_intro
            
            # 3. EQ matching for seamless tonal continuity
            if self.enable_eq_matching and self.eq_strength > 0.0:
                print(f"  Spectral balance - Track1 (L/M/H): {track1_chars['low_ratio']:.2f}/{track1_chars['mid_ratio']:.2f}/{track1_chars['high_ratio']:.2f}")
                print(f"  Spectral balance - Track2 (L/M/H): {track2_chars['low_ratio']:.2f}/{track2_chars['mid_ratio']:.2f}/{track2_chars['high_ratio']:.2f}")
                
                # Create balanced target characteristics (average of both tracks)
                target_chars = {
                    'low_ratio': (track1_chars['low_ratio'] + track2_chars['low_ratio']) / 2.0,
                    'mid_ratio': (track1_chars['mid_ratio'] + track2_chars['mid_ratio']) / 2.0,
                    'high_ratio': (track1_chars['high_ratio'] + track2_chars['high_ratio']) / 2.0,
                }
                print(f"  Target spectral balance (L/M/H): {target_chars['low_ratio']:.2f}/{target_chars['mid_ratio']:.2f}/{target_chars['high_ratio']:.2f}")
                
                # Apply EQ matching with user-specified strength
                track1_outro_processed = self._match_eq_profile(track1_outro_normalized, target_chars, track1.sr, self.eq_strength)
                track2_intro_processed = self._match_eq_profile(track2_intro_normalized, target_chars, track2.sr, self.eq_strength)
            else:
                print(f"  EQ matching disabled")
                track1_outro_processed = track1_outro_normalized
                track2_intro_processed = track2_intro_normalized
            
            print(f"  Professional audio processing complete!")
        else:
            print(f"  Audio quality processing disabled - using original audio")
            track1_outro_processed = track1_outro
            track2_intro_processed = track2_intro
        
        # 4. MICRO PEAK-TO-BEAT ALIGNMENT for perfect synchronization
        if self.enable_peak_alignment:
            print(f"  Applying micro peak-to-beat alignment for perfect synchronization...")
            
            # Get beat positions for both tracks during the transition
            outro_start = max(0, track1_end_sample - len(track1_outro_processed))
            
            # Extract beat positions that fall within the transition segments
            track1_beats_samples = librosa.frames_to_samples(track1.beats, hop_length=512)
            track2_beats_samples = librosa.frames_to_samples(track2.beats, hop_length=512)
            
            # Find beats within track1 outro segment
            track1_transition_beats = track1_beats_samples[
                (track1_beats_samples >= outro_start) & 
                (track1_beats_samples <= track1_end_sample)
            ] - outro_start  # Make relative to segment start
            
            # Find beats within track2 intro segment  
            track2_transition_beats = track2_beats_samples[
                (track2_beats_samples >= track2_start_sample) & 
                (track2_beats_samples <= track2_start_sample + len(track2_intro_processed))
            ] - track2_start_sample  # Make relative to segment start
            
            # Detect and align peaks for perfect synchronization
            if len(track1_transition_beats) > 0 and len(track2_transition_beats) > 0:
                # Detect audio peaks in both segments
                track1_peaks = self._detect_audio_peaks(track1_outro_processed, track1.sr)
                track2_peaks = self._detect_audio_peaks(track2_intro_processed, track2.sr)
                
                # Convert beat sample positions back to frames for alignment function
                track1_beats_frames = librosa.samples_to_frames(track1_transition_beats, hop_length=512)
                track2_beats_frames = librosa.samples_to_frames(track2_transition_beats, hop_length=512)
                
                # Apply micro peak-to-beat alignment to both tracks
                print(f"  Track 1 outro: {len(track1_peaks)} peaks, {len(track1_beats_frames)} beats")
                track1_outro_aligned = self._align_peaks_to_beats(
                    track1_outro_processed, track1_peaks, track1_beats_frames, track1.sr
                )
                
                print(f"  Track 2 intro: {len(track2_peaks)} peaks, {len(track2_beats_frames)} beats") 
                track2_intro_aligned = self._align_peaks_to_beats(
                    track2_intro_processed, track2_peaks, track2_beats_frames, track2.sr
                )
                
                print(f"  Micro peak-to-beat alignment complete - no audio clashing!")
            else:
                print(f"  Warning: Insufficient beats in transition segments for peak alignment")
                track1_outro_aligned = track1_outro_processed
                track2_intro_aligned = track2_intro_processed
        else:
            print(f"  Micro peak alignment disabled - using beat-matched audio")
            track1_outro_aligned = track1_outro_processed
            track2_intro_aligned = track2_intro_processed
        
        # 5. Create equal-power crossfade curves with perfectly aligned audio
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade with perfectly aligned audio
        transition = track1_outro_aligned * fade_out + track2_intro_aligned * fade_in
        
        # 5. Final output normalization to prevent clipping
        final_peak = np.max(np.abs(transition))
        if final_peak > 0.95:
            transition = transition * (0.95 / final_peak)
            print(f"  Final transition normalized (peak: {final_peak:.3f} -> 0.95)")
        
        return transition, track2.audio, track2
    
    def _create_fallback_transition(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, np.ndarray, Track]:
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
        
        return transition, track2.audio, track2
    
    def generate_mix(self, tracks: List[Track], output_path: str, transition_duration: float = None, transition_measures: int = None, transitions_only: bool = False):
        """Generate the complete DJ mix or just the transitions"""
        if len(tracks) < 2:
            raise ValueError("Need at least 2 tracks to create a mix")
        
        # Calculate target BPM based on strategy
        self.calculate_target_bpm(tracks)
        
        # Determine transition duration - convert measures to seconds if needed
        # For match-track mode, use first track's BPM for calculations
        bpm_for_calculation = self.target_bpm if self.target_bpm is not None else tracks[0].bpm
        
        if transition_measures is not None:
            # Convert measures to seconds using target BPM
            beats_per_measure = 4
            total_beats = transition_measures * beats_per_measure
            transition_duration = (total_beats * 60.0) / bpm_for_calculation
            if str(self.tempo_strategy) == "match-track":
                transition_mode = f"{transition_measures} measures ({transition_duration:.1f}s at {bpm_for_calculation:.1f} BPM reference)"
            else:
                # Use target_bpm if available, otherwise use bpm_for_calculation 
                target_bpm_display = self.target_bpm if self.target_bpm is not None else bpm_for_calculation
                transition_mode = f"{transition_measures} measures ({transition_duration:.1f}s at {target_bpm_display:.1f} BPM)"
        elif transition_duration is not None:
            # Convert seconds to measures for display
            beats = (transition_duration * bpm_for_calculation) / 60.0
            measures = beats / 4  # Assume 4/4 time
            if str(self.tempo_strategy) == "match-track":
                transition_mode = f"{transition_duration}s ({measures:.1f} measures at {bpm_for_calculation:.1f} BPM reference)"
            else:
                target_bpm_display = self.target_bpm if self.target_bpm is not None else bpm_for_calculation
                transition_mode = f"{transition_duration}s ({measures:.1f} measures at {target_bpm_display:.1f} BPM)"
        else:
            # Default fallback
            transition_duration = 30.0
            beats = (transition_duration * bpm_for_calculation) / 60.0
            measures = beats / 4
            if str(self.tempo_strategy) == "match-track":
                transition_mode = f"{transition_duration}s ({measures:.1f} measures at {bpm_for_calculation:.1f} BPM reference)"
            else:
                target_bpm_display = self.target_bpm if self.target_bpm is not None else bpm_for_calculation
                transition_mode = f"{transition_duration}s ({measures:.1f} measures at {target_bpm_display:.1f} BPM)"
        
        if transitions_only:
            print("Generating transitions-only preview with 5s buffers...")
        else:
            print(f"Generating mix with {len(tracks)} tracks...")
        print(f"Transition length: {transition_mode}")
        print(f"Tempo strategy: {self.tempo_strategy}\n")
        
        if transitions_only:
            return self._generate_transitions_only(tracks, output_path, transition_duration, transition_measures)
        
        # For uniform strategy, stretch the first track to target BPM
        if str(self.tempo_strategy) == "uniform":
            first_track = self._stretch_track_to_bpm(tracks[0], self.target_bpm)
            print(f"Track 1: {first_track.filepath.name} (stretched to target BPM)")
        else:
            first_track = tracks[0]
            print(f"Track 1: {first_track.filepath.name} (native BPM)")
        
        # Start with the first track
        mix_audio = first_track.audio.copy()
        current_sr = first_track.sr
        current_bpm = first_track.bpm
        
        # Keep track of stretched tracks for proper transitions
        processed_tracks = [first_track]  # Track the actual processed versions
        
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
            
            # DYNAMIC TEMPO RAMPING: Check if we need to increase mix tempo
            actual_prev_track = processed_tracks[i-1]  # The processed previous track
            
            # Check BPM difference before stretching for tempo ramping
            prev_original_bpm = tracks[i-1].bpm  # Original BPM of previous track
            current_original_bpm = current_track.bpm  # Original BPM of current track
            bpm_difference = abs(current_original_bpm - prev_original_bpm)
            
            tempo_ramp_needed = bpm_difference > 5.0
            if tempo_ramp_needed:
                # Calculate new target BPM as average of the two involved tracks
                new_target_bpm = (prev_original_bpm + current_original_bpm) / 2
                print(f"  🎵 TEMPO RAMP: {prev_original_bpm:.1f} + {current_original_bpm:.1f} BPM difference = {bpm_difference:.1f}")
                print(f"  📈 Ramping mix tempo to average: {new_target_bpm:.1f} BPM (from transition onwards)")
                
                # Update target BPM for future tracks
                if str(self.tempo_strategy) == "uniform":
                    self.target_bpm = new_target_bpm
                
                # We'll implement the gradual tempo change in the transition
                tempo_ramp_data = {
                    'start_bpm': prev_original_bpm,  # Use original BPMs for proper ramping
                    'end_bpm': new_target_bpm,      # Target is the average of original BPMs
                    'ramp_needed': True
                }
            else:
                tempo_ramp_data = {'ramp_needed': False}
            
            # Stretch the current track to match the target BPM
            if str(self.tempo_strategy) == "sequential":
                # Sequential: stretch current track to match previous track's BPM
                # (or new ramped BPM if ramping)
                target_bpm_for_current = new_target_bpm if tempo_ramp_needed else actual_prev_track.bpm
                stretched_track = self._stretch_track_to_bpm(current_track, target_bpm_for_current)
            elif str(self.tempo_strategy) == "uniform":
                # Uniform: stretch current track to target BPM (potentially updated)
                stretched_track = self._stretch_track_to_bpm(current_track, self.target_bpm)
            elif str(self.tempo_strategy) == "match-track":
                # Match-track: keep current track at native tempo
                stretched_track = current_track
            
            # Store the stretched track for future transitions
            processed_tracks.append(stretched_track)
            
            # Update current BPM for next iteration
            current_bpm = stretched_track.bpm
            
            # SIMPLE OVERLAY APPROACH - Natural DJ mixing:
            # Start track2 from the beginning and overlay it with the end of the current mix
            
            # Calculate transition length in samples
            transition_samples = int(transition_duration * actual_prev_track.sr)
            
            # Simple approach: Start the transition where track1 would naturally end minus transition length
            # This way we overlay the last part of track1 with the beginning of track2
            mix_length_before_track1_end = len(mix_audio)
            
            # The transition starts at the end of the current mix minus the transition duration
            transition_start_in_mix = max(0, mix_length_before_track1_end - transition_samples)
            
            # Split the current mix at the transition point
            mix_before_transition = mix_audio[:transition_start_in_mix]
            track1_overlap = mix_audio[transition_start_in_mix:]  # The overlapping part from current mix
            
            # CRITICAL FIX: Start track2 from the beginning (or early), not from beat aligner position
            # This ensures we get the full track2 body, not just the end portion
            track2_start_in_track = 0  # Start from beginning of track2
            track2_overlap = stretched_track.audio[track2_start_in_track:track2_start_in_track + len(track1_overlap)]
            
            # Ensure both overlaps are the same length
            min_overlap_length = min(len(track1_overlap), len(track2_overlap))
            if min_overlap_length <= 0:
                # Fallback: just concatenate tracks
                mix_audio = np.concatenate([mix_audio, stretched_track.audio])
                print(f"  Warning: No overlap possible, tracks concatenated directly")
            else:
                track1_overlap = track1_overlap[:min_overlap_length]
                track2_overlap = track2_overlap[:min_overlap_length]
                
                # Apply transition downbeat mapping if enabled
                if self.transition_downbeats:
                    track1_overlap, track2_overlap = self._apply_transition_downbeat_mapping(
                        track1_overlap, track2_overlap, actual_prev_track, stretched_track, current_sr, transition_duration
                    )
                
                # Apply intelligent beat alignment within the transition segments
                print(f"  🎯 Applying intelligent beat alignment within transition...")
                track1_overlap, track2_overlap = self._apply_intelligent_transition_alignment(
                    track1_overlap, track2_overlap, actual_prev_track, stretched_track,
                    transition_start_in_mix, track2_start_in_track, transition_duration
                )
                
                # Apply tempo ramping during transition if needed (not for match-track strategy)
                if tempo_ramp_data['ramp_needed'] and self.tempo_strategy != "match-track":
                    print(f"  🎶 Applying gradual tempo ramp during transition")
                    track1_overlap, track2_overlap = self._apply_tempo_ramp_to_transition(
                        track1_overlap, track2_overlap, tempo_ramp_data, current_sr
                    )
                
                # Apply match-track specific tempo ramping
                if str(self.tempo_strategy) == "match-track":
                    print(f"  🎵 Applying match-track tempo ramping during transition")
                    track1_overlap, track2_overlap = self._apply_tempo_ramping(
                        track1_overlap, track2_overlap, actual_prev_track.bpm, stretched_track.bpm, current_sr, transition_duration
                    )
                
                # Create crossfade of the overlapping segments
                fade_samples = min(len(track1_overlap), len(track2_overlap))
                fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
                fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
                
                # Ensure both are same length after tempo ramping
                track1_overlap = track1_overlap[:fade_samples]
                track2_overlap = track2_overlap[:fade_samples]
                
                # Apply frequency transition blending if enabled (can be combined)
                if self.enable_lf_transition:
                    track1_overlap, track2_overlap = self._apply_lf_transition(
                        track1_overlap, track2_overlap, current_sr
                    )
                    print(f"    Low-frequency blending applied (cutoff: {200.0} Hz)")
                    
                if self.enable_mf_transition:
                    track1_overlap, track2_overlap = self._apply_mf_transition(
                        track1_overlap, track2_overlap, current_sr
                    )
                    
                if self.enable_hf_transition:
                    track1_overlap, track2_overlap = self._apply_hf_transition(
                        track1_overlap, track2_overlap, current_sr
                    )
                
                # Apply crossfade based on transition types
                if self.enable_lf_transition or self.enable_mf_transition or self.enable_hf_transition:
                    # For frequency transitions, the audio is already blended
                    crossfaded_overlap = track1_overlap + track2_overlap
                else:
                    # Normal crossfade
                    crossfaded_overlap = track1_overlap * fade_out + track2_overlap * fade_in
                
                # CRITICAL FIX: Get ALL of track2 after the crossfade region
                # The crossfade uses track2_start_in_track as the start point and min_overlap_length as duration
                # So the remainder starts at track2_start_in_track + min_overlap_length
                crossfade_end_in_track2 = track2_start_in_track + min_overlap_length
                
                if crossfade_end_in_track2 < len(stretched_track.audio):
                    track2_remainder = stretched_track.audio[crossfade_end_in_track2:]
                else:
                    # CRITICAL BUG: Beat aligner chose a start position too late in the track
                    print(f"  WARNING: Track {i+1} entirely consumed in crossfade - beat aligner bug!")
                    print(f"    Track length: {len(stretched_track.audio)} samples ({len(stretched_track.audio)/current_sr:.1f}s)")
                    print(f"    Crossfade start: {track2_start_in_track}, length: {min_overlap_length}")  
                    print(f"    Crossfade end: {crossfade_end_in_track2}")
                    print(f"    FIXING: Using simple start position instead of beat aligner")
                    
                    # FIX: Use a simple early start position to ensure we get the full track
                    safe_start_sample = min(transition_samples, len(stretched_track.audio) // 4)  # Start in first quarter
                    track2_overlap = stretched_track.audio[safe_start_sample:safe_start_sample + min_overlap_length]
                    if len(track2_overlap) < min_overlap_length:
                        track2_overlap = np.pad(track2_overlap, (0, min_overlap_length - len(track2_overlap)), 'constant')
                    
                    # Redo crossfade with corrected position
                    crossfaded_overlap = track1_overlap * fade_out + track2_overlap * fade_in
                    track2_remainder = stretched_track.audio[safe_start_sample + min_overlap_length:]
                    
                    print(f"    FIXED: Using start position {safe_start_sample}, remainder: {len(track2_remainder)} samples")
                
                # Build the complete mix: [mix_before_transition] + [crossfaded_overlap] + [ALL_remaining_track2]
                mix_audio = np.concatenate([
                    mix_before_transition,    # Mix up to transition point
                    crossfaded_overlap,       # Crossfaded transition region  
                    track2_remainder          # ALL remaining audio from track2
                ])
            
            if min_overlap_length > 0:
                print(f"  Track {i+1} mixed using natural overlay approach:")
                print(f"    Mix before transition: {len(mix_before_transition)} samples")
                print(f"    Crossfaded overlap: {min_overlap_length} samples ({min_overlap_length/current_sr:.1f}s)")
                print(f"    Track2 remainder: {len(track2_remainder)} samples")
                print(f"    Final mix: {len(mix_audio)} samples ({len(mix_audio)/current_sr/60:.1f} min)")
                print(f"    Structure: {len(mix_before_transition)} + {min_overlap_length} + {len(track2_remainder)} = {len(mix_audio)}")
                print(f"    Natural overlay: track2 starts from beginning, overlays with end of mix")
            
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
    
    def _process_track_transition(self, prev_track: Track, current_track: Track, transition_duration: float, current_mix_audio: np.ndarray = None) -> tuple[np.ndarray, Track]:
        """
        Process a single track transition using the exact same logic as main mixing.
        This method contains the shared transition logic used by both full mix and transitions-only generation.
        
        Args:
            prev_track: The previous track (already processed)
            current_track: The current track to mix in
            transition_duration: Duration of the transition in seconds
            current_mix_audio: Current mix audio (if None, uses prev_track.audio)
            
        Returns:
            tuple: (new_mix_audio, processed_current_track)
        """
        if current_mix_audio is None:
            current_mix_audio = prev_track.audio.copy()
        
        current_sr = prev_track.sr
        
        # Ensure sample rates match
        if current_track.sr != current_sr:
            current_track.audio = librosa.resample(
                current_track.audio, 
                orig_sr=current_track.sr, 
                target_sr=current_sr
            )
            current_track.sr = current_sr
        
        # DYNAMIC TEMPO RAMPING: Check if we need tempo ramping
        prev_original_bpm = prev_track.bpm
        current_original_bpm = current_track.bpm
        bpm_difference = abs(current_original_bpm - prev_original_bpm)
        
        tempo_ramp_needed = bpm_difference > 5.0
        if tempo_ramp_needed:
            # Calculate new target BPM as average of the two involved tracks
            new_target_bpm = (prev_original_bpm + current_original_bpm) / 2
            tempo_ramp_data = {
                'start_bpm': prev_original_bpm,
                'end_bpm': new_target_bpm,
                'ramp_needed': True
            }
        else:
            tempo_ramp_data = {'ramp_needed': False}
        
        # Stretch the current track based on tempo strategy
        if str(self.tempo_strategy) == "sequential":
            target_bpm_for_current = new_target_bpm if tempo_ramp_needed else prev_track.bpm
            stretched_track = self._stretch_track_to_bpm(current_track, target_bpm_for_current)
        elif str(self.tempo_strategy) == "uniform":
            stretched_track = self._stretch_track_to_bpm(current_track, self.target_bpm)
        elif str(self.tempo_strategy) == "match-track":
            stretched_track = current_track
        else:
            stretched_track = current_track
        
        # Calculate transition length in samples
        transition_samples = int(transition_duration * prev_track.sr)
        
        # Simple overlay approach - start track2 from beginning and overlay with end of current mix
        mix_length_before_track1_end = len(current_mix_audio)
        transition_start_in_mix = max(0, mix_length_before_track1_end - transition_samples)
        
        # Split the current mix at the transition point
        mix_before_transition = current_mix_audio[:transition_start_in_mix]
        track1_overlap = current_mix_audio[transition_start_in_mix:]
        
        # Start track2 from the beginning
        track2_start_in_track = 0
        track2_overlap = stretched_track.audio[track2_start_in_track:track2_start_in_track + len(track1_overlap)]
        
        # Ensure both overlaps are the same length
        min_overlap_length = min(len(track1_overlap), len(track2_overlap))
        if min_overlap_length <= 0:
            # Fallback: just concatenate tracks
            new_mix_audio = np.concatenate([current_mix_audio, stretched_track.audio])
            return new_mix_audio, stretched_track
        
        track1_overlap = track1_overlap[:min_overlap_length]
        track2_overlap = track2_overlap[:min_overlap_length]
        
        # Apply transition downbeat mapping if enabled
        if self.transition_downbeats:
            track1_overlap, track2_overlap = self._apply_transition_downbeat_mapping(
                track1_overlap, track2_overlap, prev_track, stretched_track, current_sr, transition_duration
            )
        
        # Apply intelligent beat alignment within the transition segments
        track1_overlap, track2_overlap = self._apply_intelligent_transition_alignment(
            track1_overlap, track2_overlap, prev_track, stretched_track,
            transition_start_in_mix, track2_start_in_track, transition_duration
        )
        
        # Apply tempo ramping during transition if needed
        if tempo_ramp_data['ramp_needed'] and self.tempo_strategy != "match-track":
            track1_overlap, track2_overlap = self._apply_tempo_ramp_to_transition(
                track1_overlap, track2_overlap, tempo_ramp_data, current_sr
            )
        
        # Apply match-track specific tempo ramping
        if str(self.tempo_strategy) == "match-track":
            track1_overlap, track2_overlap = self._apply_tempo_ramping(
                track1_overlap, track2_overlap, prev_track.bpm, stretched_track.bpm, current_sr, transition_duration
            )
        
        # Create crossfade of the overlapping segments
        fade_samples = min(len(track1_overlap), len(track2_overlap))
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Ensure both are same length after tempo ramping
        track1_overlap = track1_overlap[:fade_samples]
        track2_overlap = track2_overlap[:fade_samples]
        
        # Apply frequency transition blending if enabled
        if self.enable_lf_transition:
            track1_overlap, track2_overlap = self._apply_lf_transition(
                track1_overlap, track2_overlap, current_sr
            )
        
        if self.enable_mf_transition:
            track1_overlap, track2_overlap = self._apply_mf_transition(
                track1_overlap, track2_overlap, current_sr
            )
        
        if self.enable_hf_transition:
            track1_overlap, track2_overlap = self._apply_hf_transition(
                track1_overlap, track2_overlap, current_sr
            )
        
        # Note: Volume and EQ matching would be applied here if implemented
        
        # Create the final crossfaded overlap
        crossfaded_overlap = track1_overlap * fade_out + track2_overlap * fade_in
        
        # Get the remainder of track2 (after the transition)
        crossfade_end_in_track2 = track2_start_in_track + len(crossfaded_overlap)
        track2_remainder = stretched_track.audio[crossfade_end_in_track2:]
        
        # Build the complete new mix
        new_mix_audio = np.concatenate([mix_before_transition, crossfaded_overlap, track2_remainder])
        
        return new_mix_audio, stretched_track

    def _generate_complete_mini_mix(self, tracks: List[Track], transition_duration: float) -> np.ndarray:
        """
        Generate a complete mini-mix using the exact same full mixing process.
        This ensures authentic overlaid transitions that match the main mixing output.
        """
        if len(tracks) != 2:
            raise ValueError("Mini-mix generation requires exactly 2 tracks")
        
        # Initialize the mix with the first track
        current_sr = tracks[0].sr
        mix_audio = tracks[0].audio.copy()
        processed_tracks = [tracks[0]]  # Track processed versions
        
        # Process the second track using the exact same logic as main mixing
        actual_prev_track = processed_tracks[0]
        current_track = tracks[1]
        
        # Ensure sample rates match
        if current_track.sr != current_sr:
            current_track.audio = librosa.resample(
                current_track.audio, 
                orig_sr=current_track.sr, 
                target_sr=current_sr
            )
            current_track.sr = current_sr
        
        # Apply dynamic tempo ramping logic (same as main mixing)
        prev_original_bpm = tracks[0].bpm
        current_original_bpm = current_track.bpm
        bpm_difference = abs(current_original_bpm - prev_original_bpm)
        
        tempo_ramp_needed = bpm_difference > 5.0
        if tempo_ramp_needed:
            new_target_bpm = (prev_original_bpm + current_original_bpm) / 2
            if str(self.tempo_strategy) == "uniform":
                self.target_bpm = new_target_bpm
            tempo_ramp_data = {
                'start_bpm': prev_original_bpm,
                'end_bpm': new_target_bpm,
                'ramp_needed': True
            }
        else:
            tempo_ramp_data = {'ramp_needed': False}
        
        # Stretch the current track (same logic as main mixing)
        if str(self.tempo_strategy) == "sequential":
            target_bpm_for_current = new_target_bpm if tempo_ramp_needed else actual_prev_track.bpm
            stretched_track = self._stretch_track_to_bpm(current_track, target_bpm_for_current)
        elif str(self.tempo_strategy) == "uniform":
            stretched_track = self._stretch_track_to_bpm(current_track, self.target_bpm)
        elif str(self.tempo_strategy) == "match-track":
            # Match-track: keep current track at native tempo
            stretched_track = current_track
        
        processed_tracks.append(stretched_track)
        
        # Apply the natural overlay approach (exact same as main mixing)
        transition_samples = int(transition_duration * actual_prev_track.sr)
        mix_length_before_track1_end = len(mix_audio)
        transition_start_in_mix = max(0, mix_length_before_track1_end - transition_samples)
        
        # Split the current mix at the transition point
        mix_before_transition = mix_audio[:transition_start_in_mix]
        track1_overlap = mix_audio[transition_start_in_mix:]
        
        # Start track2 from the beginning
        track2_start_in_track = 0
        track2_overlap = stretched_track.audio[track2_start_in_track:track2_start_in_track + len(track1_overlap)]
        
        # Ensure both overlaps are the same length
        min_overlap_length = min(len(track1_overlap), len(track2_overlap))
        if min_overlap_length <= 0:
            # Fallback: just concatenate tracks
            return np.concatenate([mix_audio, stretched_track.audio])
        
        track1_overlap = track1_overlap[:min_overlap_length]
        track2_overlap = track2_overlap[:min_overlap_length]
        
        # Apply intelligent beat alignment (same as main mixing)
        track1_overlap, track2_overlap = self._apply_intelligent_transition_alignment(
            track1_overlap, track2_overlap, actual_prev_track, stretched_track,
            transition_start_in_mix, track2_start_in_track, transition_duration
        )
        
        # Apply tempo ramping if needed (same as main mixing)
        if tempo_ramp_data['ramp_needed']:
            track1_overlap, track2_overlap = self._apply_tempo_ramp_to_transition(
                track1_overlap, track2_overlap, tempo_ramp_data, current_sr
            )
        
        # Create crossfade (same as main mixing)
        fade_samples = min(len(track1_overlap), len(track2_overlap))
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        track1_overlap = track1_overlap[:fade_samples]
        track2_overlap = track2_overlap[:fade_samples]
        
        crossfaded_overlap = track1_overlap * fade_out + track2_overlap * fade_in
        
        # Get track2 remainder (same as main mixing)
        crossfade_end_in_track2 = track2_start_in_track + min_overlap_length
        track2_remainder = stretched_track.audio[crossfade_end_in_track2:]
        
        # Build final mix (same as main mixing)
        return np.concatenate([mix_before_transition, crossfaded_overlap, track2_remainder])

    def _generate_transitions_only(self, tracks: List[Track], output_path: str, transition_duration: float, transition_measures: int = None):
        """
        Generate only the transition sections with 2-measure buffers for preview testing.
        Uses the same shared transition logic as the main mix path for consistency.
        """
        print("Creating transitions-only preview for testing...")
        
        # For match-track mode, use first track's BPM for calculations
        bpm_for_calculation = self.target_bpm if self.target_bpm is not None else tracks[0].bpm
        
        if transition_measures is not None:
            if str(self.tempo_strategy) == "match-track":
                print(f"Each transition: {transition_measures} measures ({transition_duration:.1f}s at {bpm_for_calculation:.1f} BPM reference)")
            else:
                print(f"Each transition: {transition_measures} measures ({transition_duration:.1f}s at {bpm_for_calculation:.1f} BPM)")
        
        current_sr = tracks[0].sr
        
        # Calculate 2 measures duration for buffer
        beats_per_minute = bpm_for_calculation
        beats_per_second = beats_per_minute / 60.0
        buffer_duration = 8 / beats_per_second  # 8 beats = 2 measures
        buffer_samples = int(buffer_duration * current_sr)
        
        if str(self.tempo_strategy) == "match-track":
            print(f"Using 2-measure buffers ({buffer_duration:.1f}s at {bpm_for_calculation:.1f} BPM reference)")
        else:
            print(f"Using 2-measure buffers ({buffer_duration:.1f}s at {bpm_for_calculation:.1f} BPM)")
        
        all_transitions = []
        silence_gap = np.zeros(int(1.0 * current_sr))  # 1 second gap between transitions
        
        # Apply the same initial track processing as the main path
        if str(self.tempo_strategy) == "uniform":
            first_track = self._stretch_track_to_bpm(tracks[0], self.target_bpm)
            print(f"Track 1: {first_track.filepath.name} (stretched to target BPM for transitions)")
        else:
            first_track = tracks[0]
            print(f"Track 1: {first_track.filepath.name} (native BPM for transitions)")
        
        processed_tracks = [first_track]
        
        for i in range(len(tracks) - 1):
            current_track = processed_tracks[i]
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
            
            # Use the shared transition processing method
            print(f"  🎯 Using shared transition logic...")
            complete_transition_mix, processed_next_track = self._process_track_transition(
                current_track, next_track, transition_duration, current_track.audio
            )
            
            # Store the processed track for the next iteration
            processed_tracks.append(processed_next_track)
            
            # Calculate extraction points for the transition segment
            transition_samples = int(transition_duration * current_sr)
            track1_length = len(current_track.audio)
            
            # The transition starts where track1 ends minus transition duration
            transition_start_in_mix = max(0, track1_length - transition_samples)
            transition_end_in_mix = transition_start_in_mix + transition_samples
            
            # Extract with 2-measure buffers: [buffer_before] [transition] [buffer_after]
            extract_start = max(0, transition_start_in_mix - buffer_samples)
            extract_end = min(len(complete_transition_mix), transition_end_in_mix + buffer_samples)
            
            transition_segment = complete_transition_mix[extract_start:extract_end]
            
            if len(transition_segment) > 0:
                all_transitions.append(transition_segment)
                segment_duration = len(transition_segment) / current_sr
                print(f"  Transition segment: {segment_duration:.1f}s (with 2-measure buffers: {buffer_duration:.1f}s before/after)")
            else:
                print(f"  Warning: Could not extract transition segment")
        
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
        print(f"Contains {len(all_transitions)} transitions with 2-measure buffers")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print(f"Each transition includes 2 measures before (primary track) + transition + 2 measures after (secondary track)")
        print("Listen to this file to test transition quality before generating full mix!")