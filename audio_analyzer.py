#!/usr/bin/env python3
"""
Audio analysis functionality for DJ Mix Generator
"""

import librosa
import numpy as np
import os
from pathlib import Path
from models import Track
from scipy.signal import find_peaks
from track_cache import TrackCache


class AudioAnalyzer:
    """Handles audio analysis for BPM, key detection, and beat tracking"""
    
    def __init__(self, use_cache: bool = True, manual_downbeats: bool = False):
        self.key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.use_cache = use_cache
        self.manual_downbeats = manual_downbeats
        self.cache = TrackCache() if use_cache else None
        
    def analyze_track(self, filepath: str) -> Track:
        """Analyze a single audio file for BPM, key, and beat positions"""
        print(f"Analyzing: {os.path.basename(filepath)}")
        
        # Check cache first
        if self.use_cache and self.cache:
            cached_track = self.cache.get_cached_analysis(filepath, self.manual_downbeats)
            if cached_track is not None:
                cache_type = "manual" if self.manual_downbeats else "automatic"
                print(f"  ✓ Loaded from cache ({cache_type} downbeats)")
                return cached_track
        
        try:
            print(f"  Performing fresh analysis...")
            
            # Load audio file
            audio, sr = librosa.load(filepath, sr=None)
            duration = len(audio) / sr
            
            # BPM detection
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            bpm = float(tempo)
            
            # Downbeat detection (stronger beats that mark musical measures)
            if self.manual_downbeats:
                downbeats, was_manual = self._get_manual_downbeats(audio, sr, beats, os.path.basename(filepath))
            else:
                downbeats = self._detect_downbeats(audio, sr, beats, bpm)
                was_manual = False
            
            # Key detection (simplified chromagram approach)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            key = self._estimate_key(chroma)
            
            track = Track(
                filepath=Path(filepath),
                audio=audio,
                sr=sr,
                bpm=bpm,
                key=key,
                beats=beats,
                downbeats=downbeats,
                duration=duration
            )
            
            # Cache the results - only cache as manual if it was actually manually selected
            if self.use_cache and self.cache:
                cache_as_manual = self.manual_downbeats and was_manual
                self.cache.cache_analysis(track, cache_as_manual)
            
            return track
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            raise
    
    def _estimate_key(self, chroma) -> str:
        """Simple key estimation using chromagram"""
        # This is a simplified approach - you might want to use more sophisticated methods
        chroma_mean = np.mean(chroma, axis=1)
        estimated_key = self.key_names[np.argmax(chroma_mean)]
        
        # Simple major/minor detection based on chord patterns
        # This is very basic - real key detection is much more complex
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Rotate profiles to match the detected root note
        root_idx = np.argmax(chroma_mean)
        major_rotated = np.roll(major_profile, root_idx)
        minor_rotated = np.roll(minor_profile, root_idx)
        
        # Calculate correlation with major and minor profiles
        major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
        
        # Add major/minor suffix
        if major_corr > minor_corr:
            return f"{estimated_key} major"
        else:
            return f"{estimated_key} minor"
    
    def _detect_downbeats(self, audio: np.ndarray, sr: int, beats: np.ndarray, bpm: float) -> np.ndarray:
        """
        Enhanced downbeat detection optimized for kick drums and percussive elements
        Uses multiple spectral analysis techniques for better accuracy
        """
        try:
            # Estimate typical beats per measure (assume 4/4 time signature)
            beats_per_measure = 4
            
            # Method 1: Percussive element analysis (kick drums, claps)
            # Separate percussive and harmonic content
            percussive, harmonic = librosa.effects.hpss(audio)
            
            # Focus on low-frequency content for kick drums (20-250 Hz)
            kick_freq_audio = librosa.effects.preemphasis(percussive)
            
            # Calculate enhanced onset strength focusing on percussive elements
            onset_envelope = librosa.onset.onset_strength(
                y=kick_freq_audio, 
                sr=sr,
                aggregate=np.median,  # Use median for more robust detection
                fmax=250,  # Focus on kick drum frequency range
                n_mels=64
            )
            
            # Method 2: Spectral flux analysis for percussive onsets
            # Calculate spectral flux (measure of spectral change)
            S = np.abs(librosa.stft(percussive))
            spectral_flux = np.sum(np.diff(S, axis=1) > 0, axis=0)
            
            # Method 3: Beat strength analysis with improved windowing
            beat_strengths = []
            percussive_strengths = []
            
            for i, beat in enumerate(beats):
                if beat < len(onset_envelope):
                    # Enhanced windowing around each beat
                    window_start = max(0, int(beat - 3))
                    window_end = min(len(onset_envelope), int(beat + 4))
                    
                    # Multiple strength metrics
                    onset_strength = np.max(onset_envelope[window_start:window_end])
                    beat_strengths.append(onset_strength)
                    
                    # Percussive strength using spectral flux
                    if beat < len(spectral_flux):
                        flux_window_start = max(0, int(beat - 2))
                        flux_window_end = min(len(spectral_flux), int(beat + 3))
                        perc_strength = np.max(spectral_flux[flux_window_start:flux_window_end])
                        percussive_strengths.append(perc_strength)
                    else:
                        percussive_strengths.append(0)
                else:
                    beat_strengths.append(0)
                    percussive_strengths.append(0)
            
            beat_strengths = np.array(beat_strengths)
            percussive_strengths = np.array(percussive_strengths)
            
            # Method 4: Low-frequency energy analysis for kick drums
            # Focus on 60-120 Hz range where kick drums are prominent
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr)
            kick_freq_mask = (freqs >= 60) & (freqs <= 120)
            
            kick_energy = []
            for beat in beats:
                beat_sample = librosa.frames_to_samples(beat, hop_length=512)
                stft_window_start = max(0, beat_sample // 512 - 2)
                stft_window_end = min(stft.shape[1], beat_sample // 512 + 3)
                
                if stft_window_start < stft.shape[1]:
                    kick_spectrum = np.abs(stft[kick_freq_mask, stft_window_start:stft_window_end])
                    kick_energy.append(np.max(kick_spectrum) if kick_spectrum.size > 0 else 0)
                else:
                    kick_energy.append(0)
            
            kick_energy = np.array(kick_energy)
            
            # Method 5: Combined scoring system
            # Normalize all metrics to 0-1 range
            if np.max(beat_strengths) > 0:
                beat_strengths = beat_strengths / np.max(beat_strengths)
            if np.max(percussive_strengths) > 0:
                percussive_strengths = percussive_strengths / np.max(percussive_strengths)
            if np.max(kick_energy) > 0:
                kick_energy = kick_energy / np.max(kick_energy)
            
            # Combined score with weights favoring percussive elements
            combined_score = (
                0.4 * beat_strengths +      # General onset strength
                0.35 * percussive_strengths +  # Percussive content
                0.25 * kick_energy          # Low-frequency kick content
            )
            
            # Find downbeat candidates using enhanced peak detection
            downbeat_candidates = set()
            
            # Method 6: Peak detection on combined score
            if len(combined_score) > 0:
                # Adaptive threshold based on score distribution
                threshold = np.mean(combined_score) + 0.5 * np.std(combined_score)
                min_distance = max(2, beats_per_measure - 1)  # Minimum 2 beats apart
                
                peaks, properties = find_peaks(
                    combined_score, 
                    height=threshold,
                    distance=min_distance,
                    prominence=0.1  # Require significant prominence
                )
                downbeat_candidates.update(peaks)
            
            # Method 7: Regular interval validation
            # Start with highest scoring beat in first measure and check regular intervals
            if len(beats) >= beats_per_measure:
                # Find the best candidate in the first measure
                first_measure_end = min(beats_per_measure, len(combined_score))
                if first_measure_end > 0:
                    best_first_beat = np.argmax(combined_score[:first_measure_end])
                    
                    # Add regular intervals from this starting point
                    for i in range(best_first_beat, len(beats), beats_per_measure):
                        if i < len(beats):
                            downbeat_candidates.add(i)
            
            # Method 8: Pattern validation using autocorrelation
            # Check if the detected pattern repeats consistently
            if len(combined_score) > beats_per_measure * 2:
                # Calculate autocorrelation of the combined score
                autocorr = np.correlate(combined_score, combined_score, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Look for peak at expected measure interval
                expected_interval = beats_per_measure
                if expected_interval < len(autocorr):
                    # If strong autocorrelation at measure interval, trust regular pattern more
                    autocorr_strength = autocorr[expected_interval] / autocorr[0] if autocorr[0] > 0 else 0
                    
                    if autocorr_strength > 0.6:  # Strong regular pattern detected
                        # Add regular intervals starting from best first beat
                        first_quarter = min(beats_per_measure, len(combined_score))
                        if first_quarter > 0:
                            best_start = np.argmax(combined_score[:first_quarter])
                            for i in range(best_start, len(beats), beats_per_measure):
                                downbeat_candidates.add(i)
            
            # Convert to sorted list and ensure reasonable spacing
            downbeat_indices = sorted(list(downbeat_candidates))
            
            # Filter out candidates that are too close together
            filtered_indices = []
            min_separation = max(2, beats_per_measure - 1)
            
            for idx in downbeat_indices:
                if not filtered_indices or idx - filtered_indices[-1] >= min_separation:
                    filtered_indices.append(idx)
            
            # Ensure we have at least some downbeats
            if not filtered_indices and len(beats) > 0:
                # Fallback: regular intervals starting from first beat
                filtered_indices = list(range(0, len(beats), beats_per_measure))
            
            # Convert to beat frame numbers
            downbeats = beats[filtered_indices] if filtered_indices else np.array([])
            
            print(f"  Detected {len(downbeats)} downbeats using enhanced percussion analysis")
            
            return downbeats
            
        except Exception as e:
            print(f"Warning: Enhanced downbeat detection failed: {e}")
            # Fallback: use regular intervals
            if len(beats) >= 4:
                return beats[::4]  # Every 4th beat as downbeat
            else:
                return np.array([beats[0]]) if len(beats) > 0 else np.array([])
    
    def _get_manual_downbeats(self, audio: np.ndarray, sr: int, beats: np.ndarray, track_name: str) -> tuple[np.ndarray, bool]:
        """
        Get downbeats through manual selection via GUI
        Returns (downbeats, was_manually_selected)
        """
        try:
            from downbeat_gui import select_first_downbeat
            
            # Show GUI for manual selection
            result = select_first_downbeat(audio, sr, track_name, beats)
            
            if result == 'cancel':
                print("  Manual selection cancelled, using automatic detection")
                return self._detect_downbeats(audio, sr, beats, 0), False  # Not manual
            elif result is None:
                print("  Using automatic detection as requested")
                return self._detect_downbeats(audio, sr, beats, 0), False  # Not manual
            else:
                print(f"  Manual downbeat selected at: {result:.3f}s")
                
                # Convert the selected time to downbeat pattern
                first_downbeat_frame = librosa.time_to_frames(result, sr=sr, hop_length=512)
                
                # Generate regular downbeats from this starting point
                beats_per_measure = 4
                downbeat_indices = []
                
                # Find the beat index closest to the selected time
                if len(beats) > 0:
                    beat_distances = np.abs(beats - first_downbeat_frame)
                    closest_beat_idx = np.argmin(beat_distances)
                    
                    # Generate downbeats at regular intervals from this point
                    for i in range(closest_beat_idx, len(beats), beats_per_measure):
                        downbeat_indices.append(i)
                    
                    # Also add downbeats before the selected point if possible
                    for i in range(closest_beat_idx - beats_per_measure, -1, -beats_per_measure):
                        downbeat_indices.insert(0, i)
                
                # Convert indices to frame numbers
                if downbeat_indices:
                    downbeats = beats[downbeat_indices]
                    print(f"  Generated {len(downbeats)} downbeats from manual selection")
                    return downbeats, True  # Was manually selected
                else:
                    print("  Could not generate downbeats from selection, using automatic detection")
                    return self._detect_downbeats(audio, sr, beats, 0), False  # Fallback to auto
            
        except ImportError:
            print("  Warning: GUI not available, falling back to automatic detection")
            return self._detect_downbeats(audio, sr, beats, 0), False  # Not manual
        except Exception as e:
            print(f"  Warning: Manual selection failed ({e}), using automatic detection")
            return self._detect_downbeats(audio, sr, beats, 0), False  # Not manual