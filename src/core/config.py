#!/usr/bin/env python3
"""
Configuration and constants for DJ Mix Generator
Centralized configuration to follow DRY principles
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class TempoStrategy(Enum):
    """Tempo alignment strategies"""
    SEQUENTIAL = "sequential"
    UNIFORM = "uniform"
    MATCH_TRACK = "match-track"


class FrequencyTransition(Enum):
    """Frequency transition types"""
    NONE = "none"
    LOW_FREQUENCY = "lf"
    MID_FREQUENCY = "mf"


@dataclass
class AudioQualitySettings:
    """Audio quality enhancement settings"""
    eq_matching: bool = True
    volume_matching: bool = True
    peak_alignment: bool = True
    tempo_correction: bool = True
    eq_strength: float = 0.5
    
    def validate(self):
        """Validate settings"""
        if not 0.0 <= self.eq_strength <= 1.0:
            raise ValueError("EQ strength must be between 0.0 and 1.0")
        if self.eq_strength == 0.0:
            self.eq_matching = False


@dataclass
class TransitionSettings:
    """Transition configuration settings"""
    measures: Optional[int] = 8
    seconds: Optional[float] = None
    enable_lf_transition: bool = True
    enable_mf_transition: bool = True
    enable_hf_transition: bool = False
    use_downbeat_mapping: bool = False
    
    def validate(self):
        """Validate transition settings"""
        if self.measures is not None and self.measures < 1:
            raise ValueError("Transition measures must be at least 1")
        if self.seconds is not None and self.seconds < 1.0:
            raise ValueError("Transition seconds must be at least 1.0")


@dataclass
class MixConfiguration:
    """Complete mix generation configuration"""
    tempo_strategy: TempoStrategy = TempoStrategy.UNIFORM
    transition_settings: TransitionSettings = None
    audio_quality: AudioQualitySettings = None
    
    # Advanced features
    reorder_by_key: bool = False
    bpm_sort: bool = False
    random_order: Optional[int] = None
    interactive_beats: bool = False
    manual_downbeats: bool = False
    allow_irregular_tempo: bool = False
    transitions_only: bool = False
    
    # System settings
    use_cache: bool = True
    
    # Track timing settings
    custom_play_time: Optional[float] = None
    
    def __post_init__(self):
        if self.transition_settings is None:
            self.transition_settings = TransitionSettings()
        if self.audio_quality is None:
            self.audio_quality = AudioQualitySettings()
    
    def validate(self):
        """Validate complete configuration"""
        self.transition_settings.validate()
        self.audio_quality.validate()
        
        if self.custom_play_time is not None and self.custom_play_time < 30.0:
            raise ValueError("Custom play time must be at least 30 seconds")


class AudioConstants:
    """Audio processing constants"""
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_HOP_LENGTH = 512
    
    # Frequency bands (Hz)
    LOW_FREQ_CUTOFF = 200.0
    MID_FREQ_LOW_CUTOFF = 200.0
    MID_FREQ_HIGH_CUTOFF = 2000.0
    HIGH_FREQ_LOW_CUTOFF = 2000.0
    HIGH_FREQ_HIGH_CUTOFF = 8000.0
    
    # Beat detection
    BEATS_PER_MEASURE = 4
    MIN_BPM = 60.0
    MAX_BPM = 200.0
    
    # Quality settings
    NORMALIZATION_PEAK = 0.95
    TEMPO_THRESHOLD = 0.02  # 2% threshold for tempo ramping


class FileConstants:
    """File handling constants"""
    SUPPORTED_FORMATS = ['.wav']
    DEFAULT_OUTPUT_NAME = 'dj_mix.wav'
    TRANSITIONS_OUTPUT_NAME = 'dj_transitions_preview.wav'
    CACHE_EXTENSION = '.cache'
    
    # Cache settings
    CACHE_DIR_NAME = '.dj_cache'
    MAX_CACHE_SIZE_MB = 1000


class GuiConstants:
    """GUI-related constants"""
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    PLOT_DPI = 100
    
    # Colors
    TRACK1_COLOR = 'steelblue'
    TRACK2_COLOR = 'darkorange'
    SELECTION_COLOR = 'red'
    GRID_ALPHA = 0.3