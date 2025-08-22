#!/usr/bin/env python3
"""
Common audio processing utilities
Extracted from various components to follow DRY principles
"""

import numpy as np
import librosa
from typing import Tuple, Optional, Union
from core.config import AudioConstants


class AudioProcessor:
    """Common audio processing operations"""
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, peak: float = AudioConstants.NORMALIZATION_PEAK) -> np.ndarray:
        """Normalize audio to specified peak level"""
        current_peak = np.max(np.abs(audio))
        if current_peak > 0:
            return audio * (peak / current_peak)
        return audio
    
    @staticmethod
    def create_fade_curves(length: int, curve_type: str = "equal_power") -> Tuple[np.ndarray, np.ndarray]:
        """Create crossfade curves"""
        if curve_type == "equal_power":
            fade_out = np.cos(np.linspace(0, np.pi/2, length))
            fade_in = np.sin(np.linspace(0, np.pi/2, length))
        elif curve_type == "linear":
            fade_out = np.linspace(1, 0, length)
            fade_in = np.linspace(0, 1, length)
        elif curve_type == "exponential":
            fade_out = np.exp(-5 * np.linspace(0, 1, length))
            fade_in = 1 - fade_out
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        return fade_out, fade_in
    
    @staticmethod
    def apply_crossfade(track1: np.ndarray, track2: np.ndarray, 
                       fade_out: Optional[np.ndarray] = None,
                       fade_in: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply crossfade between two audio segments"""
        min_length = min(len(track1), len(track2))
        track1 = track1[:min_length]
        track2 = track2[:min_length]
        
        if fade_out is None or fade_in is None:
            fade_out, fade_in = AudioProcessor.create_fade_curves(min_length)
        
        return track1 * fade_out + track2 * fade_in
    
    @staticmethod
    def time_stretch_audio(audio: np.ndarray, stretch_ratio: float, 
                          method: str = "librosa") -> np.ndarray:
        """Time-stretch audio by given ratio"""
        if abs(stretch_ratio - 1.0) < 0.001:  # No stretching needed
            return audio
            
        if method == "librosa":
            return librosa.effects.time_stretch(audio, rate=stretch_ratio)
        else:
            raise ValueError(f"Unknown time stretch method: {method}")


class FrequencyProcessor:
    """Frequency domain audio processing"""
    
    @staticmethod
    def create_filters(sr: int, low_cutoff: float, high_cutoff: Optional[float] = None):
        """Create Butterworth filters"""
        from scipy.signal import butter
        
        nyquist = sr / 2
        filters = {}
        
        # Low-pass filter
        if low_cutoff < nyquist:
            lp_b, lp_a = butter(4, low_cutoff / nyquist, btype='low')
            filters['lowpass'] = (lp_b, lp_a)
        
        # High-pass filter  
        if high_cutoff and high_cutoff < nyquist:
            hp_b, hp_a = butter(4, high_cutoff / nyquist, btype='high')
            filters['highpass'] = (hp_b, hp_a)
        
        # Band-pass filter
        if low_cutoff < nyquist and high_cutoff and high_cutoff < nyquist:
            bp_b, bp_a = butter(4, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
            filters['bandpass'] = (bp_b, bp_a)
        
        return filters
    
    @staticmethod
    def apply_filter(audio: np.ndarray, filter_coeffs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Apply a filter to audio"""
        from scipy.signal import filtfilt
        b, a = filter_coeffs
        return filtfilt(b, a, audio)
    
    @staticmethod
    def separate_frequency_bands(audio: np.ndarray, sr: int, 
                                low_cutoff: float = AudioConstants.LOW_FREQ_CUTOFF,
                                high_cutoff: float = AudioConstants.MID_FREQ_HIGH_CUTOFF) -> dict:
        """Separate audio into frequency bands"""
        filters = FrequencyProcessor.create_filters(sr, low_cutoff, high_cutoff)
        bands = {}
        
        if 'lowpass' in filters:
            bands['low'] = FrequencyProcessor.apply_filter(audio, filters['lowpass'])
        
        if 'bandpass' in filters:
            bands['mid'] = FrequencyProcessor.apply_filter(audio, filters['bandpass'])
        
        if 'highpass' in filters:
            bands['high'] = FrequencyProcessor.apply_filter(audio, filters['highpass'])
        
        return bands


class TransitionProcessor:
    """Specialized processors for transitions"""
    
    @staticmethod
    def apply_frequency_transition(track1_audio: np.ndarray, track2_audio: np.ndarray,
                                 sr: int, transition_type: str = "low") -> Tuple[np.ndarray, np.ndarray]:
        """Apply frequency-specific transition blending"""
        if transition_type == "low":
            return TransitionProcessor._apply_low_frequency_transition(track1_audio, track2_audio, sr)
        elif transition_type == "mid":
            return TransitionProcessor._apply_mid_frequency_transition(track1_audio, track2_audio, sr)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
    
    @staticmethod
    def _apply_low_frequency_transition(track1_audio: np.ndarray, track2_audio: np.ndarray,
                                      sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply low-frequency transition blending"""
        n_samples = len(track1_audio)
        
        # Separate frequency bands
        bands1 = FrequencyProcessor.separate_frequency_bands(track1_audio, sr)
        bands2 = FrequencyProcessor.separate_frequency_bands(track2_audio, sr)
        
        # Create blending curves
        lf_fade_curve = np.linspace(1, 0, n_samples)   # Track1 LF fade out
        lf_blend_curve = np.linspace(0, 1, n_samples)  # Track2 LF fade in
        
        # Normal crossfade curves for high frequencies
        hf_fade_out, hf_fade_in = AudioProcessor.create_fade_curves(n_samples)
        
        # Process track1: gradual LF reduction, normal HF fadeout
        track1_lf = bands1.get('low', np.zeros_like(track1_audio)) * lf_fade_curve
        track1_hf = bands1.get('high', track1_audio) * hf_fade_out
        track1_processed = track1_lf + track1_hf
        
        # Process track2: gradual LF increase, normal HF fadein
        track2_lf = bands2.get('low', np.zeros_like(track2_audio)) * lf_blend_curve
        track2_hf = bands2.get('high', track2_audio) * hf_fade_in
        track2_processed = track2_lf + track2_hf
        
        return track1_processed, track2_processed
    
    @staticmethod
    def _apply_mid_frequency_transition(track1_audio: np.ndarray, track2_audio: np.ndarray,
                                      sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mid-frequency transition blending"""
        n_samples = len(track1_audio)
        
        # Separate frequency bands
        bands1 = FrequencyProcessor.separate_frequency_bands(track1_audio, sr)
        bands2 = FrequencyProcessor.separate_frequency_bands(track2_audio, sr)
        
        # Create blending curves for mid frequencies
        mf_fade_curve = np.linspace(1, 0, n_samples)   # Track1 MF fade out
        mf_blend_curve = np.linspace(0, 1, n_samples)  # Track2 MF fade in
        
        # Normal crossfade curves for low and high frequencies
        normal_fade_out, normal_fade_in = AudioProcessor.create_fade_curves(n_samples)
        
        # Process track1: gradual MF reduction, normal LF/HF fadeout
        track1_lf = bands1.get('low', np.zeros_like(track1_audio)) * normal_fade_out
        track1_mf = bands1.get('mid', np.zeros_like(track1_audio)) * mf_fade_curve
        track1_hf = bands1.get('high', track1_audio) * normal_fade_out
        track1_processed = track1_lf + track1_mf + track1_hf
        
        # Process track2: gradual MF increase, normal LF/HF fadein
        track2_lf = bands2.get('low', np.zeros_like(track2_audio)) * normal_fade_in
        track2_mf = bands2.get('mid', np.zeros_like(track2_audio)) * mf_blend_curve
        track2_hf = bands2.get('high', track2_audio) * normal_fade_in
        track2_processed = track2_lf + track2_mf + track2_hf
        
        return track2_processed, track2_processed


class TempoProcessor:
    """Tempo and rhythm processing utilities"""
    
    @staticmethod
    def calculate_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
        """Calculate time-stretch ratio for BPM conversion"""
        return source_bpm / target_bpm
    
    @staticmethod
    def apply_chunk_based_tempo_ramping(track1_audio: np.ndarray, track2_audio: np.ndarray,
                                      start_bpm: float, end_bpm: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply gradual tempo ramping using chunk-based processing"""
        n_samples = len(track1_audio)
        # Use higher resolution for larger tempo differences to reduce artifacts
        tempo_diff_ratio = max(start_bpm / end_bpm, end_bpm / start_bpm)
        if tempo_diff_ratio > 1.5:  # Large tempo difference
            chunk_size = max(128, n_samples // 400)  # ~400 chunks for extreme smoothness
        else:
            chunk_size = max(256, n_samples // 200)  # ~200 chunks for ultra-smooth ramping
        
        ramped_track1 = []
        ramped_track2 = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_length = end_idx - start_idx
            
            if chunk_length == 0:
                break
            
            # Calculate progress and target tempo
            progress = (start_idx + chunk_length / 2) / n_samples
            current_target_tempo = start_bpm + progress * (end_bpm - start_bpm)
            
            # Calculate stretch ratios for both tracks to match target tempo
            track1_ratio = start_bpm / current_target_tempo
            track2_ratio = end_bpm / current_target_tempo
            
            # Extract and process chunks
            track1_chunk = track1_audio[start_idx:end_idx]
            track2_chunk = track2_audio[start_idx:end_idx]
            
            # Apply time-stretching if significant difference
            if abs(track1_ratio - 1.0) > AudioConstants.TEMPO_THRESHOLD:
                track1_chunk = AudioProcessor.time_stretch_audio(track1_chunk, track1_ratio)
            if abs(track2_ratio - 1.0) > AudioConstants.TEMPO_THRESHOLD:
                track2_chunk = AudioProcessor.time_stretch_audio(track2_chunk, track2_ratio)
            
            # Ensure chunks maintain original length
            track1_chunk = TempoProcessor._ensure_chunk_length(track1_chunk, chunk_length)
            track2_chunk = TempoProcessor._ensure_chunk_length(track2_chunk, chunk_length)
            
            ramped_track1.append(track1_chunk)
            ramped_track2.append(track2_chunk)
        
        # Concatenate all chunks
        track1_ramped = np.concatenate(ramped_track1) if ramped_track1 else track1_audio
        track2_ramped = np.concatenate(ramped_track2) if ramped_track2 else track2_audio
        
        # Ensure final arrays are correct length
        track1_ramped = track1_ramped[:len(track1_audio)]
        track2_ramped = track2_ramped[:len(track2_audio)]
        
        return track1_ramped, track2_ramped
    
    @staticmethod
    def _ensure_chunk_length(chunk: np.ndarray, target_length: int) -> np.ndarray:
        """Ensure chunk is exactly target length"""
        if len(chunk) > target_length:
            return chunk[:target_length]
        elif len(chunk) < target_length:
            return np.pad(chunk, (0, target_length - len(chunk)), 'constant')
        return chunk