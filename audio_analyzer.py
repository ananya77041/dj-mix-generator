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


class AudioAnalyzer:
    """Handles audio analysis for BPM, key detection, and beat tracking"""
    
    def __init__(self):
        self.key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def analyze_track(self, filepath: str) -> Track:
        """Analyze a single audio file for BPM, key, and beat positions"""
        print(f"Analyzing: {os.path.basename(filepath)}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(filepath, sr=None)
            duration = len(audio) / sr
            
            # BPM detection
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            bpm = float(tempo)
            
            # Downbeat detection (stronger beats that mark musical measures)
            downbeats = self._detect_downbeats(audio, sr, beats, bpm)
            
            # Key detection (simplified chromagram approach)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            key = self._estimate_key(chroma)
            
            return Track(
                filepath=Path(filepath),
                audio=audio,
                sr=sr,
                bpm=bpm,
                key=key,
                beats=beats,
                downbeats=downbeats,
                duration=duration
            )
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
        Detect downbeats (strong beats that mark the beginning of musical measures)
        Uses spectral analysis and beat strength to identify measure boundaries
        """
        try:
            # Calculate beat times in seconds
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Estimate typical beats per measure (assume 4/4 time signature)
            beats_per_measure = 4
            
            # Calculate spectral features at beat positions for strength analysis
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
            beat_strengths = []
            
            for beat in beats:
                if beat < len(onset_envelope):
                    # Get a window around each beat for strength calculation
                    window_start = max(0, int(beat - 2))
                    window_end = min(len(onset_envelope), int(beat + 3))
                    beat_strength = np.mean(onset_envelope[window_start:window_end])
                    beat_strengths.append(beat_strength)
                else:
                    beat_strengths.append(0)
            
            beat_strengths = np.array(beat_strengths)
            
            # Find potential downbeats using multiple approaches
            downbeat_candidates = set()
            
            # Method 1: Regular intervals (assume first beat is a downbeat)
            if len(beats) > 0:
                for i in range(0, len(beats), beats_per_measure):
                    if i < len(beats):
                        downbeat_candidates.add(i)
            
            # Method 2: Peak detection on beat strengths
            if len(beat_strengths) > 0:
                # Find peaks in beat strength with minimum distance
                min_distance = max(1, beats_per_measure - 1)
                strength_peaks, _ = find_peaks(beat_strengths, 
                                             distance=min_distance,
                                             height=np.mean(beat_strengths))
                downbeat_candidates.update(strength_peaks)
            
            # Method 3: Harmonic/tonal analysis for measure boundaries
            if len(beats) >= beats_per_measure:
                # Use chromagram to detect harmonic changes that often occur on downbeats
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                
                for i in range(0, len(beats) - beats_per_measure + 1, beats_per_measure):
                    # Check if there's a significant harmonic change
                    beat_frame = beats[i]
                    if beat_frame < chroma.shape[1]:
                        # Compare harmonic content before and after this beat
                        window_size = min(10, chroma.shape[1] - beat_frame)
                        if beat_frame >= window_size and beat_frame + window_size < chroma.shape[1]:
                            before = chroma[:, beat_frame - window_size:beat_frame]
                            after = chroma[:, beat_frame:beat_frame + window_size]
                            
                            # Calculate harmonic change magnitude
                            change = np.linalg.norm(np.mean(after, axis=1) - np.mean(before, axis=1))
                            if change > np.std([np.linalg.norm(np.mean(chroma[:, j:j+window_size], axis=1) - 
                                                              np.mean(chroma[:, j+window_size:j+2*window_size], axis=1)) 
                                              for j in range(0, chroma.shape[1] - 2*window_size, window_size)]):
                                downbeat_candidates.add(i)
            
            # Convert candidate indices to beat frame numbers
            downbeat_indices = sorted(list(downbeat_candidates))
            
            # Ensure we have at least some downbeats
            if not downbeat_indices and len(beats) > 0:
                # Fallback: regular intervals starting from first beat
                downbeat_indices = list(range(0, len(beats), beats_per_measure))
            
            # Convert to beat frame numbers
            downbeats = beats[downbeat_indices] if downbeat_indices else np.array([])
            
            return downbeats
            
        except Exception as e:
            print(f"Warning: Downbeat detection failed: {e}")
            # Fallback: use regular intervals
            if len(beats) >= 4:
                return beats[::4]  # Every 4th beat as downbeat
            else:
                return np.array([beats[0]]) if len(beats) > 0 else np.array([])