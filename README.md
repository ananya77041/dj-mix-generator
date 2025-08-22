# DJ Mix Generator

A professional-grade DJ mixing tool that creates seamless transitions between tracks with beat-perfect alignment, intelligent tempo management, and advanced audio processing.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd dj-mix-generator
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install spotdl for Spotify integration:**
```bash
spotdl --download-ffmpeg  # Downloads FFmpeg automatically
```

### Basic Usage

**Mix local audio files:**
```bash
python dj_mix_generator.py track1.wav track2.wav track3.wav
```

**Mix Spotify playlist:**
```bash
python dj_mix_generator.py --spotify-playlist="https://open.spotify.com/playlist/..." --transition-measures=16
```

**Custom track duration:**
```bash
python dj_mix_generator.py --custom-play-time=2:30 --transition-measures=8 *.wav
```

## ✨ Key Features

### 🎵 Spotify Integration
- Download and mix playlists directly from Spotify URLs
- Multi-source audio (YouTube Music, YouTube, SoundCloud)
- Automatic playlist naming for output files
- Smart track filtering and duplicate handling

### 🎛️ Advanced Mixing
- **Beat-perfect alignment** with intelligent shifting
- **Tempo strategies**: Sequential, Uniform, Match-track
- **Harmonic mixing** with Circle of Fifths key matching
- **Frequency-specific transitions** (low, mid, high frequency blending)
- **Custom play time** per track with intelligent cutting

### 🔧 Professional Tools
- **Interactive beatgrid alignment** with GPU-accelerated interface
- **Manual downbeat selection** for precision timing
- **Transition preview mode** for quality testing
- **Intelligent caching** for faster repeated processing
- **Comprehensive metadata** generation

### 🎚️ Audio Quality
- Volume normalization and EQ matching
- Peak alignment for professional sound
- Piecewise tempo correction (sub-5ms precision)
- Multiple crossfading algorithms

## 📖 Usage Examples

### Basic Mixing
```bash
# Simple mix with default settings
python dj_mix_generator.py song1.wav song2.wav song3.wav

# Output: dj_mix.wav (seamless mix with 8-measure transitions)
```

### Spotify Playlist Mixing
```bash
# Download and mix Spotify playlist with custom timing
python dj_mix_generator.py \
  --spotify-playlist="https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd" \
  --custom-play-time=2:00 \
  --transition-measures=16

# Output: [Playlist Name].wav
```

### Advanced Features
```bash
# Harmonic mixing with key reordering
python dj_mix_generator.py --reorder-by-key --transition-measures=16 *.wav

# Random selection with BPM sorting
python dj_mix_generator.py --random-order=10 --bpm-sort *.wav

# Transition preview for testing
python dj_mix_generator.py --transitions-only --transition-seconds=30 *.wav
```

### Interactive Mode
```bash
# Manual precision with GUI interfaces
python dj_mix_generator.py \
  --manual-downbeats \
  --interactive-beats \
  --transition-downbeats \
  *.wav
```

## 🎯 Command-Line Options

### Input Sources
- `track1.wav track2.wav ...` - Local audio files
- `--spotify-playlist=URL` - Spotify playlist URL
- `--demo` - Use built-in demo tracks

### Mixing Control
- `--tempo-strategy=match-track` - Tempo handling (sequential/uniform/match-track)
- `--transition-measures=16` - Transition length in measures
- `--transition-seconds=30` - Transition length in seconds
- `--custom-play-time=2:30` - Maximum duration per track

### Track Organization
- `--reorder-by-key` - Optimize track order for harmonic mixing
- `--bpm-sort` - Sort tracks by BPM
- `--random-order=N` - Select N random tracks

### Audio Processing
- `--no-eq-matching` - Disable EQ matching
- `--eq-strength=0.5` - EQ matching strength (0.0-1.0)
- `--no-volume-matching` - Disable volume normalization
- `--lf-transition` / `--mf-transition` / `--hf-transition` - Frequency-specific blending

### Interactive Features
- `--manual-downbeats` - GUI for downbeat selection
- `--interactive-beats` - GPU-accelerated beatgrid alignment
- `--transition-downbeats` - GUI for transition timing

### Output Options
- `--transitions-only` - Generate preview with transitions only
- `--irregular-tempo` - Allow non-integer BPM values

### Cache Management
- `--no-cache` - Disable analysis caching
- `--cache-info` - Show cache statistics
- `--clear-cache` - Clear analysis cache

## 📁 Output Files

The tool generates several output files:

- **`dj_mix.wav`** (or `[Playlist Name].wav`) - Complete mix
- **`dj_mix_metadata.txt`** - Detailed track information and timing
- **`dj_transitions_preview.wav`** - Transition-only preview (with `--transitions-only`)

### Metadata File Contents
- Track listing with artist/title information
- Precise timing for each track in the mix
- Transition details and audio processing settings
- Mix statistics and generation information

## 🛠️ Requirements

- **Python 3.8+**
- **FFmpeg** (installed automatically with spotdl)
- **Audio files in WAV format** (for local files)

### Optional Dependencies
```bash
# For interactive GUI features
pip install dearpygui>=1.10.0

# For audio playback during alignment
pip install sounddevice

# For additional GUI backends
sudo apt-get install python3-tk  # Ubuntu/Debian
```

## 🔧 Troubleshooting

**Common Issues:**
- **No module errors**: Run `pip install -r requirements.txt`
- **Spotify download fails**: Check internet connection and playlist privacy
- **Empty output**: Verify input WAV files are valid
- **Slow processing**: Large files and many tracks increase processing time

**Performance Tips:**
- Use `--no-cache` for one-off mixes
- Enable caching for repeated processing of same tracks
- Use `--transitions-only` to test settings quickly

## 📚 Developer Documentation

For detailed technical documentation, see the [`docs/`](docs/) directory:
- [Architecture Overview](docs/architecture.md)
- [Core Components](docs/core/)
- [CLI Implementation](docs/cli/)
- [Utilities](docs/utils/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Enjoy creating professional DJ mixes!** 🎧✨