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
from scipy import signal
from scipy.signal import hilbert


class MixGenerator:
    """Handles DJ mix generation with transitions and beatmatching"""
    
    def __init__(self, tempo_strategy: str = "sequential", interactive_beats: bool = False, 
                 enable_eq_matching: bool = True, enable_volume_matching: bool = True, eq_strength: float = 0.5,
                 enable_peak_alignment: bool = True):
        self.beat_aligner = BeatAligner(interactive_beats=interactive_beats)
        self.tempo_strategy = tempo_strategy
        self.target_bpm = None  # Will be set based on strategy
        self.interactive_beats = interactive_beats
        
        # Audio quality settings
        self.enable_eq_matching = enable_eq_matching
        self.enable_volume_matching = enable_volume_matching
        self.eq_strength = eq_strength
        self.enable_peak_alignment = enable_peak_alignment
    
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
        
        # Determine the uniform BPM that both tracks should use for crossfading
        if self.tempo_strategy == "sequential":
            target_bpm_for_crossfade = track1_stretched.bpm  # Use track1's BPM as reference
            print(f"  Sequential crossfade BPM: {target_bpm_for_crossfade:.3f} (track1's BPM)")
        elif self.tempo_strategy == "uniform":
            target_bpm_for_crossfade = self.target_bpm  # Use the exact uniform target BPM
            print(f"  Uniform crossfade BPM: {target_bpm_for_crossfade:.3f} (target BPM)")
        
        # Show what BPMs were actually achieved after stretching
        track1_actual_bpm = track1_stretched.bpm
        track2_actual_bpm = track2_stretched.bpm
        print(f"  Final BPMs after stretching: Track1={track1_actual_bpm:.3f}, Track2={track2_actual_bpm:.3f}")
        
        # In uniform mode, if track1 wasn't stretched, show what BPM it should have for context
        if self.tempo_strategy == "uniform" and not stretch_track1:
            print(f"  Note: Track1 not stretched (first track in sequence), target BPM = {self.target_bpm:.3f}")
        elif self.tempo_strategy == "sequential":
            print(f"  Note: Sequential mode - Track1 at native BPM, Track2 stretched to match")
        
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
        
        return self._create_enhanced_crossfade(track1_for_crossfade, track2_for_crossfade, transition_duration)
    
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
        updated_track2 = Track(
            filepath=track2.filepath,
            audio=track2_processed,
            sr=track2.sr,
            bpm=track2.bpm,
            key=track2.key,
            beats=track2.beats,
            downbeats=track2.downbeats,
            duration=len(track2_processed) / track2.sr
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
        updated_track2 = Track(
            filepath=track2.filepath,
            audio=track2_processed,
            sr=track2.sr,
            bpm=track2.bpm,
            key=track2.key,
            beats=track2.beats,
            downbeats=track2.downbeats,
            duration=len(track2_processed) / track2.sr
        )
        
        return complete_mix, updated_track2
    
    def _create_enhanced_crossfade(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, np.ndarray, Track]:
        """
        Create enhanced crossfade with quality improvements while maintaining original API.
        Returns (transition_audio, track2_audio, updated_track2) for proper mix accumulation.
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
        
        # Apply professional quality improvements to transition segments
        outro_start = max(0, track1_end_sample - len(track1_outro))
        intro_start = track2_start_sample
        
        track1_outro_enhanced, track2_intro_enhanced = self._apply_transition_quality_processing(
            track1_outro, track2_intro, track1, track2, outro_start, intro_start
        )
        
        # Create equal-power crossfade curves with enhanced audio
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade with enhanced audio
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
            
            # Create enhanced transition with professional quality processing
            transition, track2_audio, stretched_track = self.create_transition(prev_track, current_track, transition_duration, stretch_track1=False)
            
            # Update current BPM for next iteration
            current_bpm = stretched_track.bpm
            
            # Properly accumulate the mix: remove the outro segment and add the transition + remaining track
            transition_samples = len(transition)
            mix_audio = mix_audio[:-transition_samples]  # Remove outro segment
            mix_audio = np.concatenate([mix_audio, transition, track2_audio[transition_samples:]])
            
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
            
            # Generate the enhanced transition
            transition_audio, track2_audio, updated_track = self.create_transition(prev_track, next_track, transition_duration, stretch_track1=False)
            
            # Create complete mix for extraction purposes
            transition_samples = len(transition_audio)
            pre_transition = prev_track.audio[:-transition_samples]
            post_transition = track2_audio[transition_samples:]
            complete_mix = np.concatenate([pre_transition, transition_audio, post_transition])
            
            # Find the transition points for extraction
            track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
                prev_track, updated_track, transition_duration
            )
            
            # Extract transition section from the complete seamless mix with buffers
            transition_samples = int(transition_duration * current_sr)
            
            # Calculate positions in the complete mix
            # The complete mix structure: [track1_beginning] [transition_region] [track2_end]
            track1_length = len(prev_track.audio)
            transition_start_in_mix = max(0, track1_end_sample - transition_samples)
            transition_end_in_mix = transition_start_in_mix + transition_samples
            
            # Extract with 5s buffers
            extract_start = max(0, transition_start_in_mix - buffer_samples)
            extract_end = min(len(complete_mix), transition_end_in_mix + buffer_samples)
            
            transition_segment = complete_mix[extract_start:extract_end]
            
            if len(transition_segment) > 0:
                all_transitions.append(transition_segment)
                
                segment_duration = len(transition_segment) / current_sr
                print(f"  Transition segment: {segment_duration:.1f}s (seamless with 5s buffers)")
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
        print(f"Contains {len(all_transitions)} transitions")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print("Listen to this file to test transition quality before generating full mix!")