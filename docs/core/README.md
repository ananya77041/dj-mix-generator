# Core Components Documentation

The core package contains the fundamental audio processing and mixing engine components. This directory provides detailed technical documentation for developers working with the core functionality.

## Overview

The core components handle all audio analysis, processing, and mixing operations. They are designed to be independent of the user interface and can be used programmatically or through the CLI.

## Components

### [Audio Analysis](audio_analyzer.md)
**`audio_analyzer.py`**
- BPM detection and beat analysis
- Key detection using chromagram analysis
- Caching integration for performance
- Manual downbeat selection support

### [Mix Generation](mix_generator.md)
**`mix_generator.py`**
- Complete mixing engine with transition processing
- Tempo strategies and beat alignment
- Audio quality enhancement features
- Metadata generation

### [Beat Utilities](beat_utils.md)
**`beat_utils.py`**
- Beat alignment and tempo correction
- Interactive beatgrid adjustment
- Piecewise audio stretching
- Beat quality validation

### [Data Models](models.md)
**`models.py`**
- Track data structures
- Beat and key information models
- Audio processing metadata
- Type-safe data containers

### [Configuration](config.md)
**`config.py`**
- System configuration management
- Audio quality settings
- Transition parameters
- Constants and enums

## Core Processing Pipeline

```
Input Audio → Analysis → Beat Alignment → Mixing → Output
     ↓           ↓            ↓           ↓        ↓
AudioAnalyzer → BeatAligner → MixGenerator → WAV File
     ↓           ↓            ↓           ↓        ↓  
   Cache    → Interactive → Transitions → Metadata
             Adjustment
```

## Quick Start for Developers

### Basic Audio Analysis
```python
from core.audio_analyzer import AudioAnalyzer
from core.models import Track

analyzer = AudioAnalyzer(use_cache=True)
track = analyzer.analyze_track("audio.wav")
print(f"BPM: {track.bpm}, Key: {track.key}")
```

### Simple Mix Generation
```python
from core.mix_generator import MixGenerator
from core.config import MixConfiguration

config = MixConfiguration()
mixer = MixGenerator(config)
mixer.generate_mix(tracks, "output.wav", transition_duration=30.0)
```

### Interactive Beat Alignment
```python
from core.beat_utils import BeatAligner

aligner = BeatAligner(interactive_beats=True)
aligned_tracks = aligner.align_tracks(tracks)
```

## Key Concepts

### Track Model
All audio files are represented as `Track` objects containing:
- Audio data and sample rate
- Beat information (beats, downbeats, BPM)
- Key information and confidence
- Metadata dictionary

### Beat Alignment
The system uses sophisticated beat alignment to ensure smooth transitions:
- Intelligent beat shifting without artifacts
- Piecewise stretching for tempo correction
- Interactive adjustment with visual feedback

### Tempo Strategies
Three approaches to handling tempo differences:
- **Sequential**: Use first track's BPM throughout
- **Uniform**: Stretch all tracks to average BPM
- **Match-track**: Gradual tempo ramping during transitions

### Caching System
Expensive audio analysis is cached automatically:
- File-based persistence with JSON metadata
- Automatic cache cleanup and validation
- Configurable cache size and location

## Error Handling

The core components implement robust error handling:
- Graceful degradation for corrupted audio files
- Clear error messages with context
- Automatic fallback for failed operations
- Validation of audio file formats and properties

## Performance Optimization

Key performance features:
- Parallel processing for multiple track analysis
- Efficient numpy array operations
- Memory-conscious audio processing
- Intelligent caching to avoid re-computation

## Thread Safety

Core components are designed for thread safety:
- Thread-safe caching operations
- Parallel analysis support
- Proper resource management
- No shared mutable state

For detailed information about each component, see the individual documentation files.