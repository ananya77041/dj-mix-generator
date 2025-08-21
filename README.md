# DJ Mix Generator

Professional DJ mixing tool that creates seamless transitions between tracks with beat-perfect alignment and intelligent processing.

## Features

- **Professional beat alignment** with intelligent beat shifting and artifact prevention
- **Dynamic tempo ramping** for smooth BPM transitions between tracks with large differences
- **Harmonic mixing** with Circle of Fifths key matching and automatic track reordering
- **Interactive beatgrid alignment** with GPU-accelerated Dear PyGui interface
- **Intelligent track caching** with automatic analysis persistence
- **Measure-based transitions** for musically consistent mixing
- **Professional audio quality** with volume normalization, EQ matching, and peak alignment
- **Transition preview mode** with 2-measure buffers for testing quality
- **Multiple tempo strategies** (sequential/uniform) and smart BPM/key sorting
- **Manual downbeat selection** for precision timing control

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **For GUI features** (`--manual-downbeats`, `--interactive-beats`):
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Optional: Audio playback during alignment
pip install sounddevice
```

## Usage

### Basic Usage

```bash
python dj_mix_generator.py track1.wav track2.wav track3.wav
```

Creates seamless 8-measure transitions using the first track's BPM, outputs `dj_mix.wav`.

### Harmonic Mixing

```bash
python dj_mix_generator.py --reorder-by-key track1.wav track2.wav track3.wav
```

Reorders tracks for optimal harmonic compatibility using Circle of Fifths relationships.

### Transition Preview

```bash
python dj_mix_generator.py --transitions-only track1.wav track2.wav track3.wav
```

Creates `dj_transitions_preview.wav` with 2-measure buffers around each transition for quality testing.

### Caching

```bash
python dj_mix_generator.py --cache-info      # View cache
python dj_mix_generator.py --clear-cache     # Clear cache
python dj_mix_generator.py --no-cache [...]  # Disable caching
```

Automatically caches track analysis for faster repeated usage.

### Manual Downbeat Selection

```bash
python dj_mix_generator.py --manual-downbeats track1.wav track2.wav track3.wav
```

Interactive GUI for precise first downbeat selection with beat snapping and visual feedback.

### Interactive Beatgrid Alignment

```bash
python dj_mix_generator.py --interactive-beats track1.wav track2.wav track3.wav
```

**GPU-Accelerated Interface (Dear PyGui):**
- 60+ FPS real-time performance with professional dark theme
- 3-step workflow with live audio playback and beat stretching
- Real-time BPM adjustment with immediate visual feedback

**Fallback Interface (matplotlib):**
- Interactive beat alignment with drag-and-drop functionality
- Real-time alignment quality feedback

```bash
pip install dearpygui>=1.10.0  # For GPU acceleration
```

### Audio Quality Features

**Volume & EQ Matching**
```bash
# Default: Volume + 50% EQ matching
python dj_mix_generator.py track1.wav track2.wav track3.wav

# Custom EQ strength
python dj_mix_generator.py --eq-strength=0.25 track1.wav track2.wav track3.wav

# Disable features
python dj_mix_generator.py --no-volume-matching --no-eq-matching track1.wav track2.wav track3.wav
```

**Peak Alignment**
```bash
# Default: Peak alignment enabled
python dj_mix_generator.py track1.wav track2.wav track3.wav

# Disable for faster processing
python dj_mix_generator.py --no-peak-alignment track1.wav track2.wav track3.wav
```

Features: RMS volume normalization, 3-band EQ matching, and micro peak-to-beat alignment for professional results.

### Tempo Strategies

```bash
# Sequential: Use first track's BPM (default)
python dj_mix_generator.py --tempo-strategy=sequential track1.wav track2.wav track3.wav

# Uniform: Use average BPM of all tracks
python dj_mix_generator.py --tempo-strategy=uniform track1.wav track2.wav track3.wav

# Match-track: Each track plays at native tempo with tempo ramping during transitions
python dj_mix_generator.py --tempo-strategy=match-track track1.wav track2.wav track3.wav
```

### Transition Length

```bash
# Measure-based (recommended) - 8 measures default
python dj_mix_generator.py --transition-measures=16 track1.wav track2.wav track3.wav

# Time-based (legacy) - fixed seconds
python dj_mix_generator.py --transition-seconds=30 track1.wav track2.wav track3.wav
```

## Example Usage

```bash
# Basic mixing
python dj_mix_generator.py track1.wav track2.wav track3.wav

# Harmonic mixing with preview
python dj_mix_generator.py --reorder-by-key --transitions-only track1.wav track2.wav track3.wav

# Manual precision
python dj_mix_generator.py --manual-downbeats --interactive-beats track1.wav track2.wav

# Professional quality
python dj_mix_generator.py --tempo-strategy=uniform --transition-measures=16 track1.wav track2.wav
```

## How It Works

1. **Track Analysis**: Extracts BPM, key, beats, and downbeats using librosa
2. **Beat Matching**: Time-stretches tracks to match tempo while preserving pitch
3. **Enhanced Detection**: Optimized percussion analysis for accurate downbeat detection
4. **Tempo Correction**: Piecewise stretching eliminates drift with sub-5ms precision
5. **Perfect Alignment**: Beat-by-beat matching for professional transitions
6. **Crossfading**: Equal-power curves with beat-aligned segments
7. **Quality Processing**: Volume normalization, EQ matching, and peak alignment

## Limitations

- Simple key detection using chromagram analysis
- Assumes 4/4 time signature
- WAV files only

## Project Structure

Modular components:
- `models.py` - Track dataclass
- `audio_analyzer.py` - BPM/key detection 
- `beat_utils.py` - Beat alignment
- `mix_generator.py` - Mixing engine
- `key_matcher.py` - Harmonic mixing
- `track_cache.py` - Analysis caching
- `dj_mix_generator.py` - Main CLI


## Troubleshooting

- **ModuleNotFoundError**: `pip install -r requirements.txt`
- **Empty file errors**: Ensure WAV files are valid and not corrupted
- **No GUI backend**: Install tkinter or PyQt5 for interactive features
- **No audio playback**: `pip install sounddevice` for alignment audio
- **Long processing**: Large files take time; BPM detection is slowest step

Enjoy your automated DJ mixes! 🎧