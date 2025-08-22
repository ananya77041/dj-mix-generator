# Development Guide

This document provides a comprehensive guide for developers working on the DJ Mix Generator project. It covers architecture, development setup, coding standards, and contribution guidelines.

## 📁 Project Structure

```
dj-mix-generator/
├── src/                     # Source code
│   ├── cli/                 # Command-line interface
│   │   ├── args_parser.py   # Argument parsing and validation
│   │   └── main.py          # Main CLI application
│   ├── core/                # Core audio processing engine
│   │   ├── audio_analyzer.py # Audio analysis (BPM, beats, key)
│   │   ├── beat_utils.py    # Beat alignment and correction
│   │   ├── config.py        # Configuration management
│   │   ├── mix_generator.py # Main mixing engine
│   │   └── models.py        # Data models and structures
│   ├── gui/                 # Interactive user interfaces
│   │   ├── base_gui.py      # Common GUI utilities
│   │   ├── downbeat_gui.py  # Manual downbeat selection
│   │   ├── transition_gui.py # Transition timing GUI
│   │   └── beatgrid/        # Interactive beatgrid alignment
│   │       ├── advanced_gui.py    # GPU-accelerated interface
│   │       ├── fallback_gui.py    # Matplotlib fallback
│   │       └── simple_gui.py      # Basic alignment interface
│   └── utils/               # Utilities and external integrations
│       ├── audio_processing.py    # Audio processing helpers
│       ├── cache.py         # Intelligent caching system
│       ├── key_matching.py  # Harmonic mixing algorithms
│       └── spotify_downloader.py # Spotify playlist integration
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── comprehensive/      # End-to-end tests
├── docs/                   # Documentation
│   ├── core/              # Core component docs
│   ├── cli/               # CLI documentation
│   ├── utils/             # Utilities documentation
│   └── architecture.md    # System architecture
├── data/                   # Test data and examples
├── scripts/                # Development and deployment scripts
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # User documentation
```

## 🏗️ Architecture Overview

### Layered Architecture
```
┌─────────────────┐
│   CLI Layer     │  # User interface and argument handling
├─────────────────┤
│   Core Engine   │  # Audio analysis and mixing logic
├─────────────────┤
│   Utilities     │  # External services and optimization
└─────────────────┘
```

### Data Flow
```
Input → Analysis → Processing → Mixing → Output
  ↓       ↓          ↓         ↓       ↓
Audio   Track     Beat      Mix    WAV File
Files   Objects   Alignment  Gen.   + Metadata
```

## 🚀 Development Setup

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd dj-mix-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Install spotdl for Spotify integration
spotdl --download-ffmpeg
```

### 2. IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.typing.mypy-enabled": true
}
```

#### PyCharm Configuration
- Set Python interpreter to `./venv/bin/python`
- Enable type checking with mypy
- Configure code style to use Black formatter

### 3. Pre-commit Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📋 Coding Standards

### Python Style Guide
- **PEP 8 compliance** with Black formatting
- **Type hints** for all public functions
- **Docstrings** for all classes and methods (Google style)
- **Maximum line length**: 88 characters (Black default)

### Example Code Style
```python
from typing import List, Optional
import numpy as np


class AudioAnalyzer:
    """Comprehensive audio analysis for DJ mixing applications.
    
    This class handles BPM detection, beat tracking, and key analysis
    using advanced signal processing techniques.
    
    Attributes:
        use_cache: Enable intelligent caching of analysis results
        cache: Cache instance for storing analysis data
    """
    
    def __init__(self, use_cache: bool = True) -> None:
        """Initialize audio analyzer with caching options.
        
        Args:
            use_cache: Enable caching for performance optimization
        """
        self.use_cache = use_cache
        self.cache = TrackCache() if use_cache else None
    
    def analyze_track(self, filepath: str) -> Track:
        """Perform complete audio analysis on a track.
        
        Args:
            filepath: Path to audio file in WAV format
            
        Returns:
            Complete Track object with all analysis results
            
        Raises:
            ValueError: If file is invalid or analysis fails
            FileNotFoundError: If audio file doesn't exist
        """
        # Implementation with proper error handling
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        try:
            # Analysis logic here
            return track
        except Exception as e:
            raise ValueError(f"Analysis failed for {filepath}: {e}")
```

### Error Handling Patterns
```python
# Specific exception types
class AudioAnalysisError(Exception):
    """Raised when audio analysis fails."""
    pass

# Graceful degradation
try:
    result = expensive_operation()
except ExternalServiceError:
    logger.warning("External service failed, using fallback")
    result = fallback_operation()

# Clear error messages
if not tracks:
    raise ValueError(
        "No valid tracks found. Ensure WAV files are valid and not corrupted."
    )
```

## 🧪 Testing Strategy

### Test Structure
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_audio_analyzer.py
│   ├── test_mix_generator.py
│   └── test_models.py
├── integration/             # Integration tests
│   ├── test_cli_workflow.py
│   └── test_spotify_integration.py
└── comprehensive/          # End-to-end tests
    ├── test_full_workflow.py
    └── test_performance.py
```

### Test Examples
```python
import pytest
import numpy as np
from core.models import Track, BeatInfo, KeyInfo


class TestAudioAnalyzer:
    """Test suite for AudioAnalyzer class."""
    
    @pytest.fixture
    def sample_track(self):
        """Create sample track for testing."""
        return Track(
            filepath=Path("test.wav"),
            audio=np.random.randn(44100),  # 1 second of audio
            sr=44100,
            beat_info=BeatInfo(
                beats=np.array([0.0, 0.5, 1.0]),
                downbeats=np.array([0.0]),
                bpm=120.0,
                confidence=0.85
            ),
            key_info=KeyInfo(
                key="C major",
                confidence=0.90,
                chroma=np.random.randn(12)
            ),
            metadata={}
        )
    
    def test_bpm_detection(self, sample_track):
        """Test BPM detection accuracy."""
        assert sample_track.bpm == 120.0
        assert sample_track.beat_info.confidence >= 0.8
    
    def test_key_detection(self, sample_track):
        """Test key detection functionality."""
        assert "major" in sample_track.key or "minor" in sample_track.key
        assert sample_track.key_info.confidence >= 0.5
    
    @pytest.mark.parametrize("bpm,expected_interval", [
        (120.0, 0.5),
        (140.0, 0.43),
        (100.0, 0.6)
    ])
    def test_beat_intervals(self, bpm, expected_interval):
        """Test beat interval calculations."""
        beat_info = BeatInfo(
            beats=np.array([]),
            downbeats=np.array([]),
            bpm=bpm,
            confidence=1.0
        )
        assert abs(beat_info.beat_interval - expected_interval) < 0.01
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_audio_analyzer.py

# Run tests with verbose output
pytest -v

# Run performance tests
pytest tests/comprehensive/ -m "not slow"
```

## 🔧 Component Development Guide

### Adding New Audio Analysis Features

1. **Extend Track Model**
```python
# In models.py
@dataclass
class Track:
    # ... existing fields ...
    tempo_stability: Optional[float] = None  # New analysis result
```

2. **Update AudioAnalyzer**
```python
# In audio_analyzer.py
def _analyze_tempo_stability(self, audio: np.ndarray, sr: int) -> float:
    """Analyze tempo stability throughout the track."""
    # Implementation here
    return stability_score

def analyze_track(self, filepath: str) -> Track:
    # ... existing analysis ...
    tempo_stability = self._analyze_tempo_stability(audio, sr)
    # Include in Track creation
```

3. **Add Caching Support**
```python
# Update cache serialization in cache.py
def _serialize_track_data(self, track: Track) -> dict:
    data = {
        # ... existing fields ...
        'tempo_stability': track.tempo_stability
    }
    return data
```

### Adding New Mixing Features

1. **Extend Configuration**
```python
# In config.py
@dataclass
class TransitionSettings:
    # ... existing settings ...
    enable_tempo_smoothing: bool = False
```

2. **Update MixGenerator**
```python
# In mix_generator.py
def _apply_tempo_smoothing(self, track1_audio, track2_audio):
    """Apply tempo smoothing during transitions."""
    if not self.config.transition_settings.enable_tempo_smoothing:
        return track1_audio, track2_audio
    # Implementation here
```

3. **Add CLI Arguments**
```python
# In args_parser.py
parser.add_argument('--tempo-smoothing', action='store_true',
                   help='Enable tempo smoothing during transitions')
```

### Adding External Service Integrations

1. **Create Service Module**
```python
# In utils/new_service.py
class NewServiceDownloader:
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or ".")
    
    def download_playlist(self, url: str) -> List[str]:
        """Download playlist from new service."""
        # Validation, download, error handling
        return file_paths
```

2. **Integrate with CLI**
```python
# In cli/main.py
def _download_new_service_playlist(self, url: str) -> List[str]:
    """Download from new service with error handling."""
    try:
        downloader = NewServiceDownloader()
        return downloader.download_playlist(url)
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        raise
```

## 📊 Performance Guidelines

### Memory Management
```python
# Good: Stream processing for large files
def process_large_audio(filepath: str):
    with sf.SoundFile(filepath) as f:
        for block in f.blocks(blocksize=1024):
            yield process_block(block)

# Bad: Loading entire file into memory
def process_large_audio_bad(filepath: str):
    audio, sr = sf.read(filepath)  # Loads entire file
    return process_audio(audio)
```

### CPU Optimization
```python
# Use numpy vectorized operations
def apply_gain_vectorized(audio: np.ndarray, gain: float) -> np.ndarray:
    return audio * gain  # Vectorized multiplication

# Avoid loops where possible
def apply_gain_slow(audio: np.ndarray, gain: float) -> np.ndarray:
    result = np.zeros_like(audio)
    for i in range(len(audio)):  # Slow loop
        result[i] = audio[i] * gain
    return result
```

### Caching Strategy
```python
# Cache expensive operations
@lru_cache(maxsize=128)
def expensive_calculation(param: int) -> float:
    # Expensive operation here
    return result

# Use file-based caching for persistence
def get_analysis_with_cache(filepath: str) -> Track:
    cache_key = f"{filepath}_{os.path.getmtime(filepath)}"
    if cached := cache.get(cache_key):
        return cached
    
    result = expensive_analysis(filepath)
    cache.set(cache_key, result)
    return result
```

## 🐛 Debugging Guidelines

### Logging Setup
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed diagnostic information")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("Serious problem occurred")
logger.critical("Program cannot continue")
```

### Common Issues and Solutions

#### Audio Analysis Problems
```python
# Debug BPM detection
def debug_bpm_detection(audio, sr):
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    print(f"Detected tempo: {tempo}")
    print(f"Beat count: {len(beats)}")
    print(f"Beat interval std: {np.std(np.diff(beats))}")
```

#### Memory Issues
```python
# Monitor memory usage
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
```

### Profiling
```bash
# Profile CPU usage
python -m cProfile -o profile.stats dj_mix_generator.py args...

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

# Memory profiling
pip install memory_profiler
python -m memory_profiler dj_mix_generator.py
```

## 📚 Documentation Standards

### Code Documentation
- **Classes**: Purpose, usage examples, key attributes
- **Methods**: Parameters, return values, exceptions, examples
- **Complex algorithms**: High-level explanation and references

### API Documentation
```python
def align_beats(
    track1: Track, 
    track2: Track, 
    transition_duration: float = 30.0
) -> Tuple[Track, Track]:
    """Align beats between two tracks for smooth transitions.
    
    This function performs intelligent beat alignment by analyzing
    the beat positions within the transition region and applying
    minimal time-stretching to achieve perfect alignment.
    
    Args:
        track1: The first track (will be aligned to)
        track2: The second track (will be aligned from)
        transition_duration: Length of transition in seconds
        
    Returns:
        A tuple of (aligned_track1, aligned_track2) where track2
        has been time-stretched to match track1's beat positions
        
    Raises:
        ValueError: If tracks have incompatible sample rates
        AudioProcessingError: If beat alignment fails
        
    Example:
        >>> track1 = analyzer.analyze_track("song1.wav")
        >>> track2 = analyzer.analyze_track("song2.wav") 
        >>> aligned1, aligned2 = align_beats(track1, track2, 30.0)
        >>> print(f"Alignment quality: {calculate_alignment_quality(aligned1, aligned2)}")
    """
```

## 🔀 Git Workflow

### Branch Strategy
```bash
# Feature development
git checkout -b feature/spotify-integration
git commit -am "Add Spotify playlist downloading"
git push origin feature/spotify-integration
# Create pull request

# Bug fixes
git checkout -b fix/audio-analysis-crash
git commit -am "Fix crash in BPM detection for short tracks"
git push origin fix/audio-analysis-crash

# Releases
git checkout -b release/v1.2.0
git tag v1.2.0
git push origin v1.2.0
```

### Commit Messages
```bash
# Good commit messages
git commit -m "Add custom play time feature with track filtering"
git commit -m "Fix memory leak in parallel track analysis"
git commit -m "Update Spotify downloader error handling"

# Bad commit messages
git commit -m "Fix stuff"
git commit -m "WIP"
git commit -m "Updates"
```

## 🚀 Release Process

### Version Management
```python
# In setup.py
version="1.2.0"

# In __init__.py
__version__ = "1.2.0"
```

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] CHANGELOG.md updated
- [ ] Performance regression testing
- [ ] Security review completed

### Deployment
```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (if applicable)
pip install twine
twine upload dist/*
```

This development guide provides the foundation for maintaining high code quality and contributor productivity. For specific implementation details, refer to the component documentation in the respective directories.