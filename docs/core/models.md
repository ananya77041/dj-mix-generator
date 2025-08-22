# Data Models

The models module defines the core data structures used throughout the DJ Mix Generator. These models provide type-safe, validated containers for audio analysis results and processing metadata.

## Overview

The data models use Python dataclasses for clean, maintainable data structures with automatic validation, serialization support, and clear interfaces. They represent the fundamental building blocks of the audio processing pipeline.

## Core Models

### `Track`
The primary data structure representing a complete audio track with all analysis results.

```python
@dataclass
class Track:
    """Enhanced track representation with comprehensive analysis data."""
    filepath: Path              # Original file path
    audio: np.ndarray          # Raw audio data
    sr: int                    # Sample rate
    beat_info: BeatInfo        # Beat analysis results
    key_info: KeyInfo          # Key detection results
    metadata: Dict[str, Any]   # Additional metadata
```

#### Properties
```python
@property
def bpm(self) -> float:
    """BPM from beat analysis."""
    return self.beat_info.bpm

@property
def key(self) -> str:
    """Key signature from key analysis."""
    return self.key_info.key

@property
def duration(self) -> float:
    """Track duration in seconds."""
    return len(self.audio) / self.sr

@property
def beats(self) -> np.ndarray:
    """Beat timestamps in seconds."""
    return self.beat_info.beats

@property
def downbeats(self) -> np.ndarray:
    """Downbeat timestamps in seconds."""
    return self.beat_info.downbeats

@property
def name(self) -> str:
    """Track name from filename."""
    return self.filepath.stem
```

#### Validation
```python
def __post_init__(self):
    """Post-initialization validation and setup."""
    if len(self.audio) == 0:
        raise ValueError("Audio data cannot be empty")
    if self.sr <= 0:
        raise ValueError("Sample rate must be positive")
    # Additional validation for beat_info and key_info
```

### `BeatInfo`
Contains all rhythm and tempo analysis results.

```python
@dataclass
class BeatInfo:
    """Beat and rhythm analysis information."""
    beats: np.ndarray          # Beat timestamps (seconds)
    downbeats: np.ndarray      # Downbeat timestamps (seconds)
    bpm: float                 # Beats per minute
    confidence: float          # Analysis confidence (0.0-1.0)
```

#### Validation
```python
def __post_init__(self):
    """Validate beat information."""
    if self.bpm < AudioConstants.MIN_BPM or self.bpm > AudioConstants.MAX_BPM:
        raise ValueError(f"BPM {self.bpm} outside valid range")
    
    if not 0.0 <= self.confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0")
```

#### Properties
```python
@property
def beat_interval(self) -> float:
    """Average interval between beats in seconds."""
    return 60.0 / self.bpm

@property
def downbeat_count(self) -> int:
    """Number of detected downbeats."""
    return len(self.downbeats)

@property
def beat_regularity(self) -> float:
    """Measure of beat timing consistency (0.0-1.0)."""
    if len(self.beats) < 2:
        return 0.0
    intervals = np.diff(self.beats)
    expected_interval = self.beat_interval
    deviations = np.abs(intervals - expected_interval)
    return 1.0 - np.mean(deviations) / expected_interval
```

### `KeyInfo`
Contains musical key detection results and confidence metrics.

```python
@dataclass
class KeyInfo:
    """Musical key detection information."""
    key: str                   # Key signature (e.g., "C major", "A minor")
    confidence: float          # Detection confidence (0.0-1.0)
    chroma: np.ndarray        # Chromagram analysis data
```

#### Key Format
```python
# Standard key naming convention
"C major", "C minor"         # Natural keys
"C# major", "F# minor"       # Sharp keys  
"Db major", "Bb minor"       # Flat keys (enharmonic equivalents)
```

#### Properties
```python
@property
def is_major(self) -> bool:
    """True if key is major, False if minor."""
    return "major" in self.key.lower()

@property
def is_minor(self) -> bool:
    """True if key is minor, False if major."""
    return "minor" in self.key.lower()

@property
def root_note(self) -> str:
    """Root note without major/minor designation."""
    return self.key.split()[0]

@property
def key_number(self) -> int:
    """Numerical representation for key matching (0-23)."""
    # 0-11: Major keys (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    # 12-23: Minor keys (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
```

## Supporting Models

### `AudioSegment`
Represents a segment of audio data with timing information.

```python
@dataclass
class AudioSegment:
    """Represents a segment of audio data."""
    data: np.ndarray          # Audio samples
    start_sample: int         # Start position in samples
    end_sample: int           # End position in samples
    sample_rate: int          # Audio sample rate
```

#### Properties
```python
@property
def duration(self) -> float:
    """Duration in seconds."""
    return len(self.data) / self.sample_rate

@property
def start_time(self) -> float:
    """Start time in seconds."""
    return self.start_sample / self.sample_rate

@property
def end_time(self) -> float:
    """End time in seconds."""
    return self.end_sample / self.sample_rate
```

### `TransitionInfo`
Metadata about transitions between tracks.

```python
@dataclass
class TransitionInfo:
    """Information about a transition between tracks."""
    track1: Track             # Source track
    track2: Track             # Destination track
    start_time: float         # Transition start time
    duration_seconds: float   # Transition duration
    transition_type: str      # Type of transition applied
    alignment_quality: float  # Quality score (0.0-1.0)
```

#### Properties
```python
@property
def duration_samples(self) -> int:
    """Transition duration in samples."""
    return int(self.duration_seconds * self.track1.sr)

@property
def end_time(self) -> float:
    """Transition end time."""
    return self.start_time + self.duration_seconds

@property
def tempo_ratio(self) -> float:
    """BPM ratio between tracks."""
    return self.track2.bpm / self.track1.bpm
```

## Serialization Support

### JSON Serialization
```python
def track_to_dict(track: Track) -> dict:
    """Convert Track to JSON-serializable dictionary."""
    return {
        'filepath': str(track.filepath),
        'sr': track.sr,
        'beat_info': {
            'beats': track.beats.tolist(),
            'downbeats': track.downbeats.tolist(),
            'bpm': track.bpm,
            'confidence': track.beat_info.confidence
        },
        'key_info': {
            'key': track.key,
            'confidence': track.key_info.confidence,
            'chroma': track.key_info.chroma.tolist()
        },
        'metadata': track.metadata
    }

def track_from_dict(data: dict) -> Track:
    """Create Track from dictionary (excluding audio data)."""
    # Reconstruct Track object from serialized data
    # Audio data loaded separately for performance
```

### Cache Integration
```python
# Automatic serialization for caching
cache_data = {
    'track_info': track_to_dict(track),
    'analysis_version': ANALYSIS_VERSION,
    'timestamp': datetime.now().isoformat(),
    'file_mtime': os.path.getmtime(track.filepath)
}
```

## Type Safety and Validation

### Input Validation
```python
def validate_audio_data(audio: np.ndarray, sr: int) -> None:
    """Validate audio data format and properties."""
    if audio.ndim > 2:
        raise ValueError("Audio must be mono or stereo")
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if len(audio) == 0:
        raise ValueError("Audio data cannot be empty")
```

### Type Hints
```python
from typing import List, Optional, Dict, Any, Tuple, Union

# Strong typing throughout the models
def process_tracks(tracks: List[Track]) -> List[TransitionInfo]:
    """Process tracks with full type safety."""
```

## Usage Examples

### Creating a Track
```python
# From audio analysis
track = Track(
    filepath=Path("song.wav"),
    audio=audio_data,
    sr=sample_rate,
    beat_info=BeatInfo(
        beats=beat_times,
        downbeats=downbeat_times,
        bpm=detected_bpm,
        confidence=0.85
    ),
    key_info=KeyInfo(
        key="A minor",
        confidence=0.92,
        chroma=chroma_data
    ),
    metadata={}
)
```

### Working with Beat Information
```python
# Access beat data
print(f"BPM: {track.bpm:.1f}")
print(f"Beat count: {len(track.beats)}")
print(f"Downbeat count: {len(track.downbeats)}")
print(f"Beat regularity: {track.beat_info.beat_regularity:.2f}")

# Time calculations
beat_interval = track.beat_info.beat_interval
measures_count = len(track.downbeats)  # Assuming 4/4 time
```

### Key Analysis
```python
# Key information
print(f"Key: {track.key}")
print(f"Root note: {track.key_info.root_note}")
print(f"Is major: {track.key_info.is_major}")
print(f"Key number: {track.key_info.key_number}")
```

### Audio Segments
```python
# Extract segment
start_samples = int(30.0 * track.sr)  # 30 seconds
end_samples = int(60.0 * track.sr)    # 60 seconds

segment = AudioSegment(
    data=track.audio[start_samples:end_samples],
    start_sample=start_samples,
    end_sample=end_samples,
    sample_rate=track.sr
)

print(f"Segment duration: {segment.duration:.1f}s")
```

## Performance Considerations

### Memory Efficiency
- Audio data stored as numpy arrays (memory-mapped when possible)
- Lazy loading of large data structures
- Efficient serialization without audio data for caching

### Validation Overhead
- Validation performed once at construction time
- No runtime validation overhead for property access
- Clear error messages for invalid data

### Type Safety Benefits
- Compile-time error detection with type checkers
- Clear interfaces and documentation
- Reduced runtime errors through validation

The data models provide a robust foundation for the DJ Mix Generator, ensuring data integrity, type safety, and clear interfaces throughout the application.