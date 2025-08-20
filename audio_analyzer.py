#!/usr/bin/env python3
"""
Audio analysis functionality for DJ Mix Generator
"""

import librosa
import numpy as np
import os
from pathlib import Path
from models import Track


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