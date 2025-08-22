# Mix Generator

The `MixGenerator` class is the core mixing engine that creates seamless DJ mixes from analyzed tracks. It handles tempo alignment, beat matching, transitions, and audio quality enhancement.

## Overview

The MixGenerator processes a list of analyzed tracks and creates a continuous mix with professional-quality transitions. It supports multiple tempo strategies, advanced audio processing, and comprehensive metadata generation.

## Class: `MixGenerator`

### Constructor
```python
def __init__(self, config=None, **kwargs):
    """
    Initialize the mix generator with configuration.
    
    Args:
        config (MixConfiguration): Complete configuration object
        **kwargs: Individual settings for backward compatibility
    """
```

### Key Methods

#### `generate_mix()`
```python
def generate_mix(
    self, 
    tracks: List[Track], 
    output_path: str, 
    transition_duration: float = None,
    transition_measures: int = None, 
    transitions_only: bool = False
) -> None:
    """
    Generate complete DJ mix or transition preview.
    
    Args:
        tracks: List of analyzed Track objects
        output_path: Output file path for the mix
        transition_duration: Transition length in seconds
        transition_measures: Transition length in measures (overrides duration)
        transitions_only: Generate preview with transitions only
    """
```

## Tempo Strategies

### Sequential Strategy
- Uses the first track's BPM as the reference
- All subsequent tracks are stretched to match
- Simple and predictable behavior

### Uniform Strategy  
- Calculates average BPM of all tracks
- Stretches all tracks to the uniform tempo
- Balanced approach for mixed-tempo sets

### Match-Track Strategy
- Each track plays at its native tempo
- Gradual tempo ramping during transitions
- Most natural sound preservation

## Transition Processing

### Beat Alignment
```python
def _align_beats_intelligent(self, track1, track2, transition_duration):
    """
    Intelligent beat alignment with artifact prevention.
    
    - Calculates optimal beat positions within transition segments
    - Applies piecewise stretching for precise alignment
    - Validates alignment quality and reverts if needed
    """
```

### Tempo Ramping
```python
def _apply_tempo_ramping(self, track1_audio, track2_audio, ...):
    """
    Smooth tempo transitions between tracks.
    
    - Chunk-based processing for ultra-smooth ramping
    - Configurable resolution (200-400 chunks)
    - Real-time progress feedback
    """
```

### Frequency Transitions
```python
def _apply_frequency_transitions(self, track1_segment, track2_segment, ...):
    """
    Frequency-specific blending during transitions.
    
    Low-frequency (20-200 Hz):
    - Prevents kick drum clashing
    - Linear crossfading from 100% track1 to 100% track2
    
    Mid-frequency (200-2000 Hz):
    - Smoother melodic transitions
    - Gradual harmonic content blending
    
    High-frequency (2000+ Hz):
    - Optional high-end enhancement
    - Maintains clarity and presence
    """
```

## Audio Quality Features

### Volume Matching
- RMS-based volume normalization
- Prevents jarring volume changes
- Maintains dynamic range

### EQ Matching
- 3-band frequency analysis and correction
- Configurable matching strength (0.0-1.0)
- Smooth tonal transitions

### Peak Alignment
- Micro-alignment of beat peaks
- Sub-sample precision timing
- Eliminates phase issues

## Custom Play Time Feature

### Track Cutting
```python
def _apply_custom_play_time(self, tracks, custom_play_time, transition_duration):
    """
    Cut tracks to specified duration while preserving transitions.
    
    - Finds optimal transition start point
    - Aligns to nearest downbeat
    - Updates beat/downbeat arrays
    - Maintains audio quality
    """
```

### Smart Filtering
- Automatically excludes tracks shorter than custom play time
- Provides clear user feedback about excluded tracks
- Ensures sufficient audio for transitions

## Metadata Generation

### Track Information
```python
def _extract_track_info(self, track: Track) -> dict:
    """
    Extract comprehensive track metadata.
    
    Returns:
        - Artist and title (parsed from filename)
        - Original and stretched BPM
        - Key information
        - File path and timing data
    """
```

### Timing Calculation
```python
def _calculate_track_timing(self, processed_tracks, transition_duration, sample_rate):
    """
    Calculate precise timing for each track in the mix.
    
    - Start/end times accounting for transitions
    - Duration in mix vs. original duration
    - Transition overlap handling
    """
```

### Output Files
- **Main mix**: `dj_mix.wav` or `[Playlist Name].wav`
- **Metadata**: `dj_mix_metadata.txt`
- **Transitions preview**: `dj_transitions_preview.wav` (with `--transitions-only`)

## Error Handling and Validation

### Input Validation
```python
def _validate_tracks(self, tracks):
    """Ensure tracks are suitable for mixing."""
    - Minimum 2 tracks required
    - Sample rate consistency
    - Valid audio data
```

### Graceful Degradation
- Continues mixing if individual tracks fail
- Automatic fallback for failed beat alignment
- Clear error reporting with context

### Quality Assurance
- Audio normalization to prevent clipping
- Validation of transition quality
- Automatic reversion of poor alignments

## Configuration Integration

### Audio Quality Settings
```python
@dataclass
class AudioQualitySettings:
    eq_matching: bool = True
    volume_matching: bool = True
    peak_alignment: bool = True
    tempo_correction: bool = True
    eq_strength: float = 0.5
```

### Transition Settings
```python
@dataclass  
class TransitionSettings:
    measures: Optional[int] = 8
    seconds: Optional[float] = None
    enable_lf_transition: bool = True
    enable_mf_transition: bool = True
    enable_hf_transition: bool = False
    use_downbeat_mapping: bool = False
```

## Performance Characteristics

### Memory Usage
- Efficient numpy array operations
- Streaming processing where possible
- Automatic garbage collection of large buffers

### Processing Speed
- Parallel analysis support
- Optimized audio algorithms
- Progress feedback for long operations

### Output Quality
- 32-bit float internal processing
- High-resolution tempo ramping
- Professional-grade crossfading

## Usage Examples

### Basic Mix Generation
```python
from core.mix_generator import MixGenerator
from core.config import MixConfiguration

config = MixConfiguration()
mixer = MixGenerator(config)
mixer.generate_mix(tracks, "output.wav")
```

### Advanced Configuration
```python
from core.config import MixConfiguration, TransitionSettings, AudioQualitySettings

transition_settings = TransitionSettings(
    measures=16,
    enable_lf_transition=True,
    enable_mf_transition=True
)

audio_quality = AudioQualitySettings(
    eq_matching=True,
    eq_strength=0.75,
    peak_alignment=True
)

config = MixConfiguration(
    transition_settings=transition_settings,
    audio_quality=audio_quality,
    custom_play_time=120.0  # 2 minutes per track
)

mixer = MixGenerator(config)
mixer.generate_mix(tracks, "professional_mix.wav")
```

### Transitions Only Preview
```python
mixer.generate_mix(
    tracks, 
    "preview.wav", 
    transition_duration=30.0,
    transitions_only=True
)
```

The MixGenerator is the heart of the DJ Mix Generator system, providing professional-quality mixing capabilities with extensive customization options and robust error handling.