#!/usr/bin/env python3
"""
Data models for DJ Mix Generator
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Track:
    """Represents an audio track with its analysis data"""
    filepath: Path
    audio: np.ndarray
    sr: int
    bpm: float
    key: str
    beats: np.ndarray
    duration: float