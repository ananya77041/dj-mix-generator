#!/usr/bin/env python3
"""
Enhanced data models for DJ Mix Generator
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from core.config import AudioConstants


@dataclass
class AudioSegment:
    """Represents a segment of audio data"""
    data: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return len(self.data) / self.sample_rate
    
    @property
    def start_time(self) -> float:
        """Start time in seconds"""
        return self.start_sample / self.sample_rate
    
    @property
    def end_time(self) -> float:
        """End time in seconds"""
        return self.end_sample / self.sample_rate


@dataclass
class BeatInfo:
    """Beat and rhythm analysis information"""
    beats: np.ndarray
    downbeats: np.ndarray
    bpm: float
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate beat information"""
        if self.bpm < AudioConstants.MIN_BPM or self.bpm > AudioConstants.MAX_BPM:
            raise ValueError(f"BPM {self.bpm} outside valid range {AudioConstants.MIN_BPM}-{AudioConstants.MAX_BPM}")
    
    @property
    def beat_count(self) -> int:
        """Number of beats"""
        return len(self.beats)
    
    @property
    def downbeat_count(self) -> int:
        """Number of downbeats"""
        return len(self.downbeats)
    
    def get_beats_in_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Get beats within a sample range"""
        beat_samples = librosa.frames_to_samples(self.beats, hop_length=AudioConstants.DEFAULT_HOP_LENGTH)
        return beat_samples[(beat_samples >= start_sample) & (beat_samples < end_sample)]
    
    def get_downbeats_in_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Get downbeats within a sample range"""
        downbeat_samples = librosa.frames_to_samples(self.downbeats, hop_length=AudioConstants.DEFAULT_HOP_LENGTH)
        return downbeat_samples[(downbeat_samples >= start_sample) & (downbeat_samples < end_sample)]


@dataclass
class KeyInfo:
    """Key and harmonic information"""
    key: str
    confidence: float = 0.0
    chroma: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate key information"""
        if not self.key:
            self.key = "Unknown"


@dataclass 
class Track:
    """Enhanced track representation with better organization"""
    filepath: Path
    audio: np.ndarray
    sr: int
    beat_info: BeatInfo
    key_info: KeyInfo
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if len(self.audio) == 0:
            raise ValueError("Audio data cannot be empty")
        if self.sr <= 0:
            raise ValueError("Sample rate must be positive")
    
    # Legacy compatibility properties
    @property
    def bpm(self) -> float:
        return self.beat_info.bpm
    
    @property
    def key(self) -> str:
        return self.key_info.key
    
    @property
    def beats(self) -> np.ndarray:
        return self.beat_info.beats
    
    @property
    def downbeats(self) -> np.ndarray:
        return self.beat_info.downbeats
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return len(self.audio) / self.sr
    
    @property
    def name(self) -> str:
        """Human-readable track name"""
        return self.filepath.stem
    
    def get_segment(self, start_sample: int, length: int) -> AudioSegment:
        """Extract an audio segment"""
        end_sample = min(start_sample + length, len(self.audio))
        segment_data = self.audio[start_sample:end_sample]
        
        return AudioSegment(
            data=segment_data,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=self.sr
        )
    
    def get_outro_segment(self, duration_seconds: float) -> AudioSegment:
        """Get the outro segment of specified duration"""
        duration_samples = int(duration_seconds * self.sr)
        start_sample = max(0, len(self.audio) - duration_samples)
        return self.get_segment(start_sample, duration_samples)
    
    def get_intro_segment(self, duration_seconds: float) -> AudioSegment:
        """Get the intro segment of specified duration"""
        duration_samples = int(duration_seconds * self.sr)
        return self.get_segment(0, duration_samples)


@dataclass
class TransitionInfo:
    """Information about a transition between tracks"""
    track1: Track
    track2: Track
    duration_seconds: float
    overlap_samples: int
    track1_start_sample: int
    track2_start_sample: int
    
    @property
    def duration_samples(self) -> int:
        """Transition duration in samples"""
        return int(self.duration_seconds * self.track1.sr)


@dataclass
class MixResult:
    """Result of a mix generation operation"""
    audio: np.ndarray
    sample_rate: int
    track_count: int
    total_duration: float
    transitions: list[TransitionInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes"""
        return self.total_duration / 60.0
    
    @property
    def file_size_mb(self) -> float:
        """Estimated file size in MB (16-bit WAV)"""
        return (len(self.audio) * 2) / (1024 * 1024)


# Import librosa at the end to avoid circular imports
try:
    import librosa
except ImportError:
    librosa = None