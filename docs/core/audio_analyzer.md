# Audio Analyzer

The `AudioAnalyzer` class handles comprehensive audio analysis including BPM detection, beat tracking, key detection, and downbeat identification. It's the foundation for all audio processing in the DJ Mix Generator.

## Overview

The AudioAnalyzer processes audio files and extracts musical information essential for DJ mixing. It uses advanced signal processing techniques and integrates with caching for performance optimization.

## Class: `AudioAnalyzer`

### Constructor
```python
def __init__(
    self, 
    use_cache: bool = True,
    manual_downbeats: bool = False,
    allow_irregular_tempo: bool = False
):
    """
    Initialize audio analyzer with configuration options.
    
    Args:
        use_cache: Enable intelligent caching of analysis results
        manual_downbeats: Enable GUI for manual downbeat selection
        allow_irregular_tempo: Allow non-integer BPM values
    """
```

### Key Methods

#### `analyze_track()`
```python
def analyze_track(self, filepath: str) -> Track:
    """
    Perform complete audio analysis on a track.
    
    Args:
        filepath: Path to audio file (WAV format)
        
    Returns:
        Track: Complete track object with analysis results
        
    Raises:
        ValueError: If file is invalid or analysis fails
    """
```

## Analysis Pipeline

### 1. Audio Loading
```python
def _load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load and validate audio file.
    
    - Supports WAV format
    - Automatic mono conversion
    - Sample rate standardization
    - Duration validation
    """
```

### 2. BPM Detection
```python
def _detect_bpm(self, audio: np.ndarray, sr: int) -> float:
    """
    Advanced BPM detection with multiple algorithms.
    
    Methods:
    - Onset detection with spectral analysis
    - Autocorrelation of beat patterns  
    - Harmonic analysis for complex rhythms
    - Confidence weighting and validation
    
    Returns:
        BPM value (rounded to integer unless allow_irregular_tempo=True)
    """
```

### 3. Beat Tracking
```python
def _detect_beats(self, audio: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """
    Precise beat location detection.
    
    - Dynamic programming beat tracker
    - Onset strength analysis
    - Tempo consistency validation
    - Sub-sample precision timing
    
    Returns:
        Array of beat timestamps in seconds
    """
```

### 4. Downbeat Detection
```python
def _detect_downbeats(self, audio: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    """
    Enhanced downbeat detection with multiple strategies.
    
    Primary Method - Enhanced Percussion Analysis:
    - Isolates percussion elements using spectral filtering
    - Analyzes low-frequency emphasis patterns
    - Detects characteristic downbeat signatures
    - Validates against beat grid consistency
    
    Fallback Method - Spectral Analysis:
    - Chroma and tonal analysis
    - Harmonic change detection
    - Structural boundary identification
    
    Returns:
        Array of downbeat timestamps in seconds
    """
```

### 5. Key Detection
```python
def _detect_key(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Musical key detection using chromagram analysis.
    
    Process:
    - Short-time Fourier transform
    - Chromagram extraction (12 pitch classes)
    - Template matching against major/minor profiles
    - Confidence calculation based on correlation strength
    
    Returns:
        Tuple of (key_name, confidence_score)
        Key format: "C major", "A minor", etc.
    """
```

## Advanced Features

### Manual Downbeat Selection
```python
def _get_manual_downbeats(self, audio: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    """
    Interactive GUI for precise downbeat selection.
    
    Features:
    - Visual waveform display with beat markers
    - Audio playback for verification
    - Click-to-select interface
    - Beat snapping for accuracy
    - Real-time validation feedback
    """
```

### Caching Integration
```python
def _get_cached_analysis(self, filepath: str) -> Optional[Track]:
    """
    Intelligent cache retrieval with validation.
    
    - File modification time checking
    - Cache version compatibility
    - Automatic cache invalidation
    - Thread-safe operations
    """

def _cache_analysis(self, track: Track) -> None:
    """
    Store analysis results for future use.
    
    - JSON serialization of analysis data
    - Automatic cache directory management
    - Size limit enforcement
    - Orphaned file cleanup
    """
```

## Analysis Quality Validation

### Beat Quality Assessment
```python
def _validate_beat_analysis(self, beats: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    """
    Assess quality of beat detection results.
    
    Metrics:
    - Beat interval consistency
    - Spectral energy alignment
    - Onset correlation strength
    - Tempo stability analysis
    
    Returns:
        Quality metrics dictionary
    """
```

### Downbeat Confidence Scoring
```python
def _calculate_downbeat_confidence(self, downbeats: np.ndarray, beats: np.ndarray) -> float:
    """
    Calculate confidence in downbeat detection.
    
    Factors:
    - Regularity of downbeat spacing (4/4 time assumption)
    - Spectral energy at downbeat locations
    - Consistency with harmonic analysis
    - Pattern recognition strength
    
    Returns:
        Confidence score (0.0-1.0)
    """
```

## Error Handling

### File Validation
```python
def _validate_audio_file(self, filepath: str) -> None:
    """
    Comprehensive audio file validation.
    
    Checks:
    - File existence and readability
    - WAV format validation
    - Minimum duration requirements
    - Sample rate compatibility
    - Audio data integrity
    """
```

### Graceful Degradation
- Automatic fallback algorithms for difficult tracks
- Clear error messages with context
- Partial analysis results when possible
- Robust handling of corrupted files

## Performance Optimization

### Efficient Processing
```python
def _optimize_audio_for_analysis(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """
    Optimize audio for faster analysis.
    
    - Downsampling for BPM detection
    - Mono conversion
    - Normalization
    - Pre-filtering for noise reduction
    """
```

### Parallel Processing Support
- Thread-safe caching operations
- Concurrent analysis of multiple tracks
- Efficient memory management
- Progress tracking for long operations

## Configuration Options

### Analysis Parameters
```python
# BPM detection range
MIN_BPM = 60
MAX_BPM = 200

# Beat tracking sensitivity
BEAT_TRACK_SENSITIVITY = 0.8

# Downbeat detection threshold
DOWNBEAT_THRESHOLD = 0.6

# Key detection window size
KEY_ANALYSIS_WINDOW = 30  # seconds
```

### Audio Processing Constants
```python
# Default sample rate for analysis
DEFAULT_SAMPLE_RATE = 22050

# Hop length for spectral analysis
HOP_LENGTH = 512

# Frame size for STFT
FRAME_SIZE = 2048
```

## Usage Examples

### Basic Analysis
```python
from core.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer(use_cache=True)
track = analyzer.analyze_track("song.wav")

print(f"Title: {track.name}")
print(f"BPM: {track.bpm:.1f}")
print(f"Key: {track.key}")
print(f"Duration: {track.duration:.1f}s")
print(f"Beats detected: {len(track.beats)}")
print(f"Downbeats detected: {len(track.downbeats)}")
```

### Manual Downbeat Selection
```python
analyzer = AudioAnalyzer(manual_downbeats=True)
track = analyzer.analyze_track("complex_rhythm.wav")
# Opens interactive GUI for downbeat selection
```

### Irregular Tempo Support
```python
analyzer = AudioAnalyzer(allow_irregular_tempo=True)
track = analyzer.analyze_track("variable_tempo.wav")
print(f"Precise BPM: {track.bpm:.2f}")  # e.g., 127.34 BPM
```

### Batch Analysis
```python
analyzer = AudioAnalyzer(use_cache=True)
tracks = []

for filepath in audio_files:
    try:
        track = analyzer.analyze_track(filepath)
        tracks.append(track)
        print(f"✓ Analyzed: {track.name}")
    except Exception as e:
        print(f"✗ Failed: {filepath} - {e}")
```

### Cache Management
```python
# View cache information
if analyzer.cache:
    analyzer.cache.print_info()

# Clear cache
analyzer.cache.clear_cache()

# Disable caching for one-off analysis
analyzer = AudioAnalyzer(use_cache=False)
```

## Data Structures

The AudioAnalyzer produces `Track` objects containing:

```python
@dataclass
class Track:
    filepath: Path           # Original file path
    audio: np.ndarray       # Audio data
    sr: int                 # Sample rate
    beat_info: BeatInfo     # Beat/downbeat/BPM data
    key_info: KeyInfo       # Key detection results
    metadata: dict          # Additional metadata
```

The AudioAnalyzer is designed for accuracy, performance, and robustness, forming the foundation for high-quality DJ mixing capabilities.