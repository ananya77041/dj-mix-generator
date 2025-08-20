# DJ Mix Generator

A modular Python tool to create continuous DJ mixes from WAV files. The program analyzes your tracks for BPM and key information, then creates seamless transitions between them.

## Features

- **Automatic BPM detection** using librosa's beat tracking
- **Musical key estimation** with chromagram analysis
- **Beat matching** via time-stretching to align tempos
- **Enhanced downbeat detection** optimized for kick drums and percussive elements
- **Visual downbeat selection** with interactive waveform interface for manual precision
- **Professional DJ transitions** with tracks entering on their first downbeat
- **Transition preview mode** to test-listen only the transition sections
- **Intelligent track caching** to avoid re-analyzing the same tracks
- **Smooth crossfade transitions** with equal-power curves (30 seconds default)
- **Harmonic mixing** with optional key-based track reordering
- **Audio normalization** to prevent clipping
- **Support for different sample rates** with automatic resampling
- **Modular architecture** for easy customization and extension

## Installation

1. Make sure you have Python 3.7+ installed
2. Install dependencies:

```bash
cd /Users/andy.mishra/workspace/dj-mix-generator
pip install -r requirements.txt
```

3. **For visual downbeat selection** (`--manual-downbeats`), you need a GUI backend:

**Option A: tkinter (recommended, usually pre-installed)**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS/Windows: tkinter should already be included
```

**Option B: PyQt5 (alternative)**
```bash
pip install PyQt5
```

Note: Installing librosa may take a few minutes as it has several dependencies including scipy and scikit-learn.

## Usage

### Basic Usage

```bash
python dj_mix_generator.py track1.wav track2.wav track3.wav
```

This will:
1. Analyze each track for BPM and key
2. Create seamless 30-second crossfade transitions
3. Use sequential tempo strategy (first track's BPM)
4. Output a file called `dj_mix.wav`

### Harmonic Mixing (Key-Based Reordering)

```bash
python dj_mix_generator.py --reorder-by-key track1.wav track2.wav track3.wav
```

This will:
1. Analyze each track for BPM and key
2. **Reorder tracks for optimal harmonic compatibility**
3. Create seamless transitions with better key matching
4. Output a file called `dj_mix.wav`

The key matching uses Circle of Fifths relationships:
- **Perfect matches**: Same key or relative major/minor
- **Compatible**: Adjacent keys in Circle of Fifths
- **Semitone**: Keys one semitone apart (energy boost)
- **Clashes**: Avoid awkward key combinations

### Transition Testing (Preview Mode)

```bash
python dj_mix_generator.py --transitions-only track1.wav track2.wav track3.wav
```

This creates a preview file (`dj_transitions_preview.wav`) containing only the transition sections:
- 5 seconds from the end of each track
- Full 30-second transition
- 5 seconds from the start of the next track
- 1-second silence gap between each transition

Perfect for testing downbeat alignment and transition quality before generating the full mix!

### Track Analysis Caching

The tool automatically caches track analysis results to speed up repeated usage:

```bash
# View cache information
python dj_mix_generator.py --cache-info

# Clear all cached analyses
python dj_mix_generator.py --clear-cache

# Clean up orphaned cache files
python dj_mix_generator.py --cleanup-cache

# Disable caching for this run
python dj_mix_generator.py --no-cache track1.wav track2.wav track3.wav
```

**Cache Benefits:**
- **Fast re-analysis**: Previously analyzed tracks load instantly from cache
- **Intelligent identification**: Uses file hash + metadata to detect identical tracks
- **Automatic management**: Cache is stored in `~/.dj_mix_generator_cache/`
- **Safe storage**: Metadata (JSON) and audio data (pickle) stored separately

### Visual Downbeat Selection

For maximum precision, manually select the first downbeat using an interactive waveform:

```bash
python dj_mix_generator.py --manual-downbeats track1.wav track2.wav track3.wav
```

**Interactive Features:**
- **Waveform Display**: Shows first 10 seconds of each track with detected beats
- **Visual Selection**: Click on the waveform where the first downbeat should occur
- **Beat Snapping**: Clicks near detected beats automatically snap for precision
- **Auto-Generation**: Creates regular downbeat pattern from your selection
- **Fallback Options**: Choose automatic detection or cancel if needed

The GUI opens for each track during analysis, allowing you to:
1. See the waveform with detected beats (orange dashed lines)
2. Click where the first downbeat should be (red line shows selection)
3. Choose "Confirm" to use your selection, "Auto-Detect" for automatic, or "Cancel"

Manual selections are cached separately from automatic detection!

### Tempo Alignment Strategies

Choose between two tempo alignment strategies:

**Sequential Strategy (Default)**
```bash
python dj_mix_generator.py --tempo-strategy=sequential track1.wav track2.wav track3.wav
```

- Each track is stretched to match the **previous track's tempo**
- First track plays at its native BPM
- Final mix BPM = first track's BPM
- Results in gradual tempo changes throughout the mix
- Best for preserving the energy and feel of the opening track

**Uniform Strategy**
```bash
python dj_mix_generator.py --tempo-strategy=uniform track1.wav track2.wav track3.wav
```

- All tracks are stretched to match the **average BPM** of all tracks
- Consistent tempo throughout the entire mix
- Final mix BPM = average of all track BPMs
- Best for maintaining steady energy across the entire set

**Example:**
- Track1: 120 BPM, Track2: 128 BPM, Track3: 132 BPM
- Sequential: Mix plays at 120 BPM (track1's tempo)
- Uniform: Mix plays at 126.7 BPM (average tempo)

### Example Output

```
Loading playlist with 3 tracks...

Analyzing: track1.wav
  Performing fresh analysis...
  Detected 42 downbeats using enhanced percussion analysis
  Cached analysis for track1.wav
  [1/3] BPM: 128.0, Key: G major, Duration: 180.5s, Downbeats: 42

Analyzing: track2.wav
  ✓ Loaded from cache (automatic downbeats)
  [2/3] BPM: 132.1, Key: A minor, Duration: 210.2s, Downbeats: 48

Analyzing: track3.wav
  Performing fresh analysis...
  Detected 45 downbeats using enhanced percussion analysis
  Cached analysis for track3.wav
  [3/3] BPM: 126.8, Key: F major, Duration: 195.7s, Downbeats: 45

Successfully loaded 3 tracks.

Original track order - Average compatibility: 1.5/3.0
Reordering tracks for optimal key matching...
Starting with: track1.wav (G major)
  Next: track3.wav (D major) - compatible match  
  Next: track2.wav (A minor) - compatible match
Reordered track order - Average compatibility: 2.0/3.0
Improvement: +0.5 compatibility points

Generating mix with 3 tracks...
Transition duration: 30.0s

Track 1: track1.wav
Track 2: track2.wav
  BPM adjustment: 128.0 -> 132.1
  Time-stretching track2 by ratio: 1.032
  Aligning to downbeats...
  Track 1 outro starts at: 172.5s, ends at: 202.5s
  Track 2 intro starts at: 4.2s
  Mix length so far: 6.2 minutes

Track 3: track3.wav
  BPM adjustment: 132.1 -> 126.8
  Time-stretching track3 by ratio: 0.960
  Aligning to downbeats...
  Track 2 outro starts at: 202.1s, ends at: 232.1s
  Track 3 intro starts at: 3.8s
  Mix length so far: 9.4 minutes

Saving mix to: dj_mix.wav

Mix complete! 🎵
Duration: 9.4 minutes
Sample rate: 44100 Hz
File size: 82.3 MB
```

## How It Works

1. **Track Analysis**: Each WAV file is analyzed using librosa to extract:
   - BPM (beats per minute) using onset detection
   - Musical key using chromagram analysis
   - Beat positions for precise alignment
   - Downbeat positions (strong beats marking measure boundaries)

2. **Beat Matching**: When transitioning between tracks:
   - Time-stretches the incoming track to match the current tempo
   - Preserves pitch while adjusting speed

3. **Enhanced Downbeat Detection**: Optimized for kick drums and percussion:
   - Separates percussive and harmonic content using HPSS (Harmonic-Percussive Source Separation)
   - Analyzes low-frequency content (60-120 Hz) for kick drum detection
   - Uses spectral flux analysis for percussive onset detection
   - Combines multiple metrics with weighted scoring system
   - Pattern validation using autocorrelation for consistent measure detection
   
4. **Downbeat Alignment**: Professional DJ-style transitions:
   - Finds optimal outro point in current track (ending on downbeat)
   - Finds optimal intro point in next track (starting on downbeat)
   - Ensures next track enters precisely on its first strong downbeat

5. **Crossfading**: Creates smooth transitions using:
   - Equal-power crossfade curves (cosine/sine)
   - 30-second default transition duration
   - Beat-aligned segments for seamless mixing

6. **Audio Processing**: 
   - Handles different sample rates via resampling
   - Normalizes final output to prevent clipping
   - Maintains audio quality throughout the process

## Limitations

This is a basic implementation with some limitations:

- **Simple key detection**: Uses basic chromagram analysis (not as accurate as professional tools)
- **Basic downbeat detection**: Uses spectral analysis but may not be 100% accurate for complex music
- **Fixed crossfade**: Uses simple equal-power crossfades (no EQ matching or advanced techniques)
- **Time signature assumption**: Assumes 4/4 time signature for downbeat detection
- **WAV only**: Currently only supports WAV files

## Project Structure

The project is organized into modular components:

- `models.py` - Track dataclass definition
- `audio_analyzer.py` - AudioAnalyzer class for BPM, key detection, and downbeat analysis
- `beat_utils.py` - BeatAligner class for precise downbeat alignment
- `mix_generator.py` - MixGenerator class for beat-aligned transitions and mixing  
- `key_matcher.py` - KeyMatcher class for harmonic mixing and track reordering
- `track_cache.py` - TrackCache class for persistent analysis caching
- `dj_mix_generator.py` - Main DJMixGenerator class and CLI interface
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Next Steps for Enhancement

- Better key detection algorithms (Krumhansl-Schmuckler, etc.)
- Advanced harmonic mixing algorithms (considering energy levels, mood)
- Automatic cue point detection for optimal transition timing
- EQ matching between tracks
- Support for MP3 and other audio formats
- GUI interface for easier use
- Advanced time-stretching with pyrubberband
- Machine learning for better track ordering based on user preferences

## Troubleshooting

**"ModuleNotFoundError: No module named 'librosa'"**
- Install dependencies: `pip install -r requirements.txt`

**"Error analyzing track.wav: Input file is empty"**
- Make sure your WAV files are valid and not corrupted
- Try converting the file with a tool like ffmpeg if needed

**"No valid tracks loaded!"**
- Check that your file paths are correct
- Ensure WAV files are readable and in a supported format

**Very long processing time**
- Large audio files take time to analyze
- Consider using shorter clips for testing
- BPM detection is the most time-consuming step

**"No working GUI backend found" or "tkinter not installed"**
- Install tkinter: `sudo apt-get install python3-tk` (Ubuntu/Debian)
- Alternative: `pip install PyQt5` 
- Visual downbeat selection requires a GUI backend
- The tool will automatically fall back to automatic detection

## Example Test

To test with your own tracks:

```bash
# Basic mixing (preserves original order)
python dj_mix_generator.py your_track1.wav your_track2.wav your_track3.wav

# Harmonic mixing (reorders for better key compatibility)
python dj_mix_generator.py --reorder-by-key your_track1.wav your_track2.wav your_track3.wav

# Test transitions only (for quick previewing)
python dj_mix_generator.py --transitions-only your_track1.wav your_track2.wav your_track3.wav

# Manual downbeat selection for precision
python dj_mix_generator.py --manual-downbeats your_track1.wav your_track2.wav your_track3.wav

# Combine options
python dj_mix_generator.py --reorder-by-key --transitions-only your_track1.wav your_track2.wav your_track3.wav
python dj_mix_generator.py --manual-downbeats --reorder-by-key your_track1.wav your_track2.wav

# Output files: dj_mix.wav (full mix) or dj_transitions_preview.wav (transitions only)
```

Enjoy your automated DJ mixes! 🎧