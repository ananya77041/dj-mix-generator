# CLI Main Application

The `DJMixGeneratorCLI` class in `main.py` is the primary application controller that orchestrates the complete workflow from command-line input to mix output.

## Overview

The main CLI application coordinates all components of the DJ Mix Generator, managing the flow from track loading through analysis, processing, and final mix generation. It provides user feedback, handles errors gracefully, and supports both local files and Spotify playlist integration.

## Class: `DJMixGeneratorCLI`

### Constructor
```python
def __init__(self):
    """Initialize CLI application with empty state."""
    self.tracks: List[Track] = []
    self.config: MixConfiguration = None
    self.analyzer: AudioAnalyzer = None
    self.mixer: MixGenerator = None
    self.key_matcher: KeyMatcher = None
    self.spotify_downloader: SpotifyPlaylistDownloader = None
```

### Main Entry Point
```python
def run(self, args: List[str] = None) -> int:
    """
    Main application entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
```

## Workflow Management

### 1. Initialization Phase
```python
def _initialize_components(self):
    """Initialize analysis and mixing components with configuration."""
    self.analyzer = AudioAnalyzer(
        use_cache=self.config.use_cache,
        manual_downbeats=self.config.manual_downbeats,
        allow_irregular_tempo=self.config.allow_irregular_tempo
    )
    
    self.mixer = MixGenerator(config=self.config)
    self.key_matcher = KeyMatcher()
```

### 2. Track Loading
```python
def _load_playlist(self, filepaths: List[str]):
    """
    Load and analyze all tracks in the playlist.
    
    Features:
    - Parallel processing for automatic analysis
    - Sequential processing for manual mode
    - Progress tracking and reporting
    - Error handling for individual tracks
    """
```

### 3. Track Processing
```python
def _filter_tracks_by_duration(self):
    """
    Filter out tracks shorter than custom play time.
    
    - Clear user feedback about excluded tracks
    - Detailed reporting of kept vs. excluded tracks
    - Validation that sufficient tracks remain
    """
```

## Spotify Integration

### Playlist Download
```python
def _download_spotify_playlist(self, spotify_url: str) -> List[str]:
    """
    Download tracks from Spotify playlist and return file paths.
    
    Process:
    1. Initialize SpotifyPlaylistDownloader
    2. Download playlist with progress feedback
    3. Extract playlist name for output file naming
    4. Return list of downloaded track file paths
    
    Features:
    - Real-time download progress
    - Multi-source audio downloading
    - Automatic playlist directory creation
    - Error handling with clear messages
    """
```

### Output File Naming
```python
def _get_output_path(self) -> str:
    """
    Get appropriate output file path using playlist name when available.
    
    Logic:
    1. Check if Spotify downloader is available with playlist name
    2. Sanitize playlist name for filesystem compatibility
    3. Generate appropriate filename for mix or transitions
    4. Fallback to default names if no playlist name
    
    Examples:
    - "Jazzy Disco Funk.wav" (from playlist name)
    - "My Playlist_transitions_preview.wav"
    - "dj_mix.wav" (default fallback)
    """
```

## Track Organization

### BPM and Key Sorting
```python
def _sort_tracks_by_bpm_and_key(self):
    """
    Sort tracks by BPM (ascending), then by key within same BPM.
    
    - Shows original and sorted order
    - Uses KeyMatcher for intelligent key sorting
    - Provides clear before/after comparison
    """
```

### Randomization
```python
def _randomize_tracks(self):
    """
    Randomly select and randomize track order.
    
    Features:
    - Configurable number of tracks to select
    - Random sampling from available tracks
    - Random shuffle of selected tracks
    - Clear feedback about selection process
    """
```

### Harmonic Reordering
```python
def _reorder_tracks_by_key(self):
    """
    Reorder tracks for optimal harmonic mixing.
    
    Process:
    1. Analyze original track flow compatibility
    2. Apply KeyMatcher reordering algorithm
    3. Analyze new track flow compatibility
    4. Report improvement metrics
    """
```

## Parallel Processing

### Automatic Analysis Mode
```python
def _load_playlist_parallel(self, filepaths: List[str]):
    """
    Load tracks in parallel for automatic analysis mode.
    
    Features:
    - ThreadPoolExecutor for concurrent analysis
    - Thread-safe caching operations
    - Progress tracking as tasks complete
    - Error handling for individual tracks
    - Optimal CPU utilization
    """
```

### Thread Safety
```python
def analyze_single_track(filepath: str) -> tuple[int, Track]:
    """
    Analyze a single track with thread-safe cache access.
    
    - Thread-safe cache operations using locks
    - Individual AudioAnalyzer instances per thread
    - Exception handling for individual track failures
    - Original ordering preservation
    """
```

## User Experience Features

### Progress Reporting
```python
# Real-time progress updates
print(f"Loading playlist with {len(filepaths)} tracks...\n")
print(f"  [{completed}/{total}] {filename}: BPM: {bpm:.1f}, Key: {key}")
print(f"✅ Successfully downloaded {len(files)} tracks")
```

### Error Handling
```python
try:
    track = self.analyzer.analyze_track(filepath)
    self.tracks.append(track)
except Exception as e:
    print(f"Skipping {filepath} due to error: {e}\n")
    # Continue processing other tracks
```

### Configuration Display
```python
# Clear configuration feedback
print(f"🎵 Using default 16-measure transitions for Spotify playlist")
print(f"📏 Custom play time: {int(custom_play_time // 60)}:{int(custom_play_time % 60):02d}")
print(f"🚫 Excluded {len(excluded_tracks)} tracks shorter than custom play time")
```

## Advanced Features

### Custom Play Time Integration
```python
# Filter tracks by minimum duration
if self.config.custom_play_time is not None:
    self._filter_tracks_by_duration()

# Pass custom play time to mix generator
# (Mix generator handles track cutting logic)
```

### Interactive Mode Support
- Manual downbeat selection coordination
- Interactive beatgrid alignment management
- GUI component initialization and cleanup

### Cache Management
- Automatic cache directory management
- Progress tracking for cached vs. fresh analysis
- Cache statistics and cleanup operations

## Configuration Integration

### MixConfiguration Usage
```python
# Parse command line to configuration object
self.config, track_input = parse_command_line()

# Apply configuration to components
self.analyzer = AudioAnalyzer(
    use_cache=self.config.use_cache,
    manual_downbeats=self.config.manual_downbeats,
    allow_irregular_tempo=self.config.allow_irregular_tempo
)
```

### Default Behavior
```python
# Spotify playlist defaults
if isinstance(track_input, str):  # Spotify URL
    track_paths = self._download_spotify_playlist(track_input)
    # Set default 16-measure transitions if not specified
    if self.config.transition_settings.measures is None and self.config.transition_settings.seconds is None:
        self.config.transition_settings.measures = 16
```

## Error Recovery

### Graceful Degradation
```python
# Continue processing even if individual tracks fail
except Exception as e:
    print(f"Skipping {filepath} due to error: {e}")
    # Application continues with remaining tracks
```

### Validation
```python
# Ensure minimum requirements are met
if not self.tracks:
    raise ValueError("No valid tracks loaded!")

if len(self.tracks) < 2:
    raise ValueError("Need at least 2 tracks to create a mix")
```

## Usage Examples

### Basic CLI Usage
```python
# Entry point
def main():
    cli = DJMixGeneratorCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
```

### Programmatic Usage
```python
# Direct instantiation
cli = DJMixGeneratorCLI()
result = cli.run(["--spotify-playlist=URL", "--transition-measures=16"])
```

### Advanced Configuration
```python
# The CLI handles all configuration through command-line arguments
# No direct configuration API - use args_parser for customization
```

## Performance Characteristics

### Memory Management
- Efficient track list management
- Automatic cleanup of large audio buffers
- Progress tracking without memory leaks

### CPU Utilization
- Parallel analysis when beneficial
- CPU count-based worker allocation
- Automatic fallback to sequential processing

### I/O Optimization
- Batch file operations
- Efficient temporary file handling
- Streaming audio processing integration

The CLI main application provides a robust, user-friendly interface that coordinates all aspects of the DJ mixing workflow while maintaining high performance and reliability.