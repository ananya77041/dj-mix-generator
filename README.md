# DJ Mix Generator

Professional DJ mixing tool that creates seamless transitions between tracks with beat-perfect alignment and intelligent processing.

## Features

- **Professional beat alignment** with intelligent beat shifting and artifact prevention
- **Dynamic tempo ramping** for smooth BPM transitions between tracks with large differences
- **Harmonic mixing** with Circle of Fifths key matching and automatic track reordering
- **Random track selection** with customizable count for variety in mixes
- **BPM-based sorting** for building energy progression in sets
- **Interactive beatgrid alignment** with GPU-accelerated Dear PyGui interface
- **Intelligent track caching** with automatic analysis persistence
- **Measure-based transitions** for musically consistent mixing
- **Professional audio quality** with volume normalization, EQ matching, and peak alignment
- **Frequency-specific transitions** (LF/MF/HF) for enhanced mixing control
- **Transition preview mode** with 2-measure buffers for testing quality
- **Multiple tempo strategies** (sequential/uniform/match-track) and flexible track ordering
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

### Track Ordering Options

**Harmonic Mixing**
```bash
python dj_mix_generator.py --reorder-by-key track1.wav track2.wav track3.wav
```
Reorders tracks for optimal harmonic compatibility using Circle of Fifths relationships.

**BPM Sorting**
```bash
python dj_mix_generator.py --bpm-sort track1.wav track2.wav track3.wav
```
Sorts tracks by BPM in ascending order for energy progression.

**Random Selection**
```bash
# Select 5 tracks at random and randomize their order
python dj_mix_generator.py --random-order 5 track1.wav track2.wav track3.wav track4.wav track5.wav track6.wav

# Combine with BPM sort (sorting applies to final selected tracks)
python dj_mix_generator.py --random-order 3 --bpm-sort track1.wav track2.wav track3.wav track4.wav
```
Randomly selects N tracks from the full selection and randomizes their order, overriding all other ordering options. Can be combined with `--bpm-sort` to sort the final selected tracks.

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

### Transition Downbeat Mapping

```bash
python dj_mix_generator.py --transition-downbeats track1.wav track2.wav track3.wav

# Combine with other features for ultimate precision
python dj_mix_generator.py --tempo-strategy=match-track --transition-downbeats --mf-transition track1.wav track2.wav track3.wav
```

Interactive GUI for precise downbeat selection within transition segments. The 2-step process allows you to select the optimal downbeat for each track's transition portion, enabling perfect alignment without changing track tempos.

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

Features: RMS volume normalization, 3-band EQ matching, micro peak-to-beat alignment, and low-frequency blending for professional results.

### Frequency-Specific Transitions

**Low-Frequency (LF) Transition** - *Enabled by default*
```bash
# LF transitions are enabled by default, disable with:
python dj_mix_generator.py --no-lf-transition track1.wav track2.wav track3.wav

# Explicitly enable (redundant since it's default):
python dj_mix_generator.py --lf-transition track1.wav track2.wav track3.wav
```
Prevents kick drum and bass clashing by gradually blending low frequencies (20-200 Hz) during transitions.

**Mid-Frequency (MF) Transition** - *Enabled by default*
```bash
# MF transitions are enabled by default, disable with:
python dj_mix_generator.py --no-mf-transition track1.wav track2.wav track3.wav

# Explicitly enable (redundant since it's default):
python dj_mix_generator.py --mf-transition track1.wav track2.wav track3.wav
```
Smooth transitions for melodic content by blending mid frequencies (200-2000 Hz) including vocals and melody lines.

**High-Frequency (HF) Transition** - *Disabled by default*
```bash
# Enable high-frequency blending:
python dj_mix_generator.py --hf-transition track1.wav track2.wav track3.wav

# Disable (redundant since it's default):
python dj_mix_generator.py --no-hf-transition track1.wav track2.wav track3.wav
```
Blends high frequencies (2000-8000 Hz) for enhanced transition smoothness when needed.

### Tempo Strategies

```bash
# Match-track: Each track plays at native tempo with ramping during transitions (default)
python dj_mix_generator.py --tempo-strategy=match-track track1.wav track2.wav track3.wav

# Sequential: Use first track's BPM for all tracks
python dj_mix_generator.py --tempo-strategy=sequential track1.wav track2.wav track3.wav

# Uniform: Use average BPM of all tracks
python dj_mix_generator.py --tempo-strategy=uniform track1.wav track2.wav track3.wav
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
# Basic mixing (LF/MF transitions enabled by default)
python dj_mix_generator.py track1.wav track2.wav track3.wav

# Harmonic mixing with preview
python dj_mix_generator.py --reorder-by-key --transitions-only track1.wav track2.wav track3.wav

# Random selection with BPM sorting
python dj_mix_generator.py --random-order 4 --bpm-sort *.wav

# Manual precision
python dj_mix_generator.py --manual-downbeats --transition-downbeats track1.wav track2.wav

# Professional quality with all transitions
python dj_mix_generator.py --tempo-strategy=uniform --transition-measures=16 --hf-transition track1.wav track2.wav
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

```
src/
├── cli/
│   ├── main.py           # Main CLI application
│   └── args_parser.py    # Command-line argument parsing
├── core/
│   ├── config.py         # Configuration classes
│   ├── models.py         # Track dataclass
│   ├── audio_analyzer.py # BPM/key detection
│   ├── mix_generator.py  # Mixing engine
│   └── beat_utils.py     # Beat alignment utilities
├── utils/
│   ├── key_matching.py   # Harmonic mixing
│   └── cache.py          # Analysis caching
└── gui/                  # Interactive GUI components

dj_mix_generator.py       # Legacy entry point (backward compatibility)
```


## Troubleshooting

- **ModuleNotFoundError**: `pip install -r requirements.txt`
- **Empty file errors**: Ensure WAV files are valid and not corrupted
- **No GUI backend**: Install tkinter or PyQt5 for interactive features
- **No audio playback**: `pip install sounddevice` for alignment audio
- **Long processing**: Large files take time; BPM detection is slowest step

Enjoy your automated DJ mixes! 🎧