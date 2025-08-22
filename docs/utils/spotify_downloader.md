# Spotify Playlist Downloader

The `SpotifyPlaylistDownloader` class provides comprehensive Spotify playlist integration, downloading tracks from multiple audio sources and organizing them into playlist-specific directories.

## Overview

The Spotify downloader acts as a bridge between Spotify playlists and the DJ Mix Generator's audio processing pipeline. It leverages the `spotdl` library to download high-quality audio from YouTube Music, YouTube, and SoundCloud.

## Class: `SpotifyPlaylistDownloader`

### Constructor
```python
def __init__(self, base_dir: Optional[str] = None):
    """
    Initialize the downloader.
    
    Args:
        base_dir: Base directory where playlist folders will be created.
                 If None, uses current working directory.
    """
```

### Key Methods

#### `download_playlist()`
```python
def download_playlist(self, spotify_url: str, format: str = "wav") -> List[str]:
    """
    Download a Spotify playlist to local audio files.
    
    Args:
        spotify_url: Spotify playlist URL or URI
        format: Audio format to download (default: wav for best quality)
        
    Returns:
        List of paths to downloaded audio files in playlist order
        
    Raises:
        ValueError: If URL is invalid or download fails
    """
```

## URL Processing

### URL Validation
```python
def _is_valid_spotify_playlist_url(self, url: str) -> bool:
    """
    Validate if URL is a valid Spotify playlist URL.
    
    Supported formats:
    - https://open.spotify.com/playlist/[playlist_id]
    - spotify:playlist:[playlist_id]
    - URLs with query parameters (automatically cleaned)
    
    Returns:
        bool: True if URL is valid Spotify playlist format
    """
```

### URL Normalization
```python
def _normalize_spotify_url(self, url: str) -> str:
    """
    Normalize Spotify URL to standard format.
    
    Process:
    1. Remove query parameters (e.g., ?si=...)
    2. Convert spotify: URIs to https URLs
    3. Ensure consistent format for processing
    
    Example:
        Input: "spotify:playlist:37i9dQZF1DX0XUsuxWHRQd"
        Output: "https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd"
    """
```

## Download Process

### Multi-Source Audio Downloading
```python
# spotdl command configuration
cmd = [
    "spotdl", "download", normalized_url,
    "--output", output_path,
    "--format", format,
    "--bitrate", "auto",           # Highest available bitrate
    "--threads", "2",              # Moderate threading for stability
    "--overwrite", "skip",         # Skip existing files
    "--audio", "youtube-music", "youtube", "soundcloud",  # Fallback sources
    "--config"                     # Use modified config file
]
```

### Progress Monitoring
```python
def _stream_download_progress(self, process):
    """
    Stream real-time download progress to user.
    
    Features:
    - Real-time output display
    - Progress tracking per track
    - Error detection and reporting
    - Success/failure statistics
    """
```

### Output Organization
```python
# Automatic playlist directory creation
output_template = "{list-name}/{artists} - {title}.{output-ext}"
# Results in: "Playlist Name/Artist - Song Title.wav"
```

## Playlist Metadata Extraction

### Playlist Name Detection
```python
def _extract_playlist_name_from_output(self, output_text: str) -> Optional[str]:
    """
    Extract playlist name from spotdl output text.
    
    Patterns recognized:
    - "Found 25 songs in Jazzy Disco Funk (Playlist)"
    - "Playlist: My Awesome Mix"
    - "Processing playlist: Chill Vibes"
    - JSON metadata fields
    
    Returns:
        Playlist name if found, None otherwise
    """
```

### Directory Name Sanitization
```python
def _sanitize_directory_name(self, name: str) -> str:
    """
    Sanitize playlist name for use as directory name.
    
    Process:
    1. Replace problematic characters with underscores
    2. Remove leading/trailing dots and spaces
    3. Limit length to prevent filesystem issues
    4. Ensure non-empty result
    
    Example:
        Input: 'My Playlist: The "Best" Songs! <2025>'
        Output: 'My_Playlist__The__Best__Songs!__2025_'
    """
```

## File Management

### Existing Track Detection
```python
def _check_for_existing_tracks(self, format: str):
    """
    Check for existing playlist directories and provide feedback.
    
    Features:
    - Scan base directory for audio files
    - Count tracks in each potential playlist directory
    - Provide user feedback about existing content
    - Support for multiple audio formats
    """
```

### Smart File Handling
```python
def _get_downloaded_files_in_dir(self, directory: Path, format: str) -> List[str]:
    """
    Get list of downloaded files in a specific directory.
    
    Process:
    1. Scan directory for audio files
    2. Support multiple audio formats (wav, mp3, flac, m4a)
    3. Sort files to maintain playlist order
    4. Return absolute file paths
    
    Returns:
        List of file paths in playlist order
    """
```

## Error Handling

### Network Error Recovery
```python
# Robust error handling for network issues
try:
    result = subprocess.run(cmd, ...)
    if result.returncode != 0:
        raise ValueError(f"spotdl download failed with return code {result.returncode}")
except FileNotFoundError:
    raise ValueError("spotdl is not installed. Install it with: pip install spotdl")
```

### Playlist Matching Logic
```python
def _find_playlist_directory(self, expected_name: str) -> Optional[Path]:
    """
    Find the correct playlist directory with fallback logic.
    
    Strategy:
    1. Try exact name match first
    2. Try fuzzy matching (case insensitive, partial matches)
    3. Fall back to directory with most audio files (with warning)
    4. Provide clear error if no suitable directory found
    
    Error Messages:
    - Clear indication when wrong playlist might be selected
    - Helpful suggestions for manual correction
    - Statistics about what was found vs. expected
    """
```

## Advanced Features

### Duplicate Handling
```python
# Automatic duplicate detection and skipping
"--overwrite", "skip"  # Skip files that already exist
# Provides feedback about skipped tracks
"🔄 Existing files will be automatically skipped"
```

### Quality Optimization
```python
# Audio quality settings
"--format", "wav",      # Lossless format for DJ mixing
"--bitrate", "auto",    # Highest available bitrate
# Multi-source fallback for maximum track availability
```

### Progress Feedback
```python
# Real-time user feedback
print("⬇️  Running spotdl download with multiple audio sources...")
print("🎵 Audio sources: YouTube Music → YouTube → SoundCloud")
print(f"🎧 Format: {format.upper()} (highest quality available)")
print("📥 Download progress:")
print("-" * 80)
# ... real-time progress output ...
print(f"📊 Found {found_count} songs in playlist, {lookup_errors} failed to find")
```

## Configuration Integration

### spotdl Configuration
```python
# Modified spotdl config file
~/.spotdl/config.json
{
    "lyrics_providers": ["genius"],  # Removed problematic providers
    "audio_providers": ["youtube-music", "youtube", "soundcloud"],
    "format": "wav",
    "bitrate": "auto"
}
```

### Directory Structure
```
Base Directory/
├── Playlist Name 1/
│   ├── Artist1 - Track1.wav
│   ├── Artist2 - Track2.wav
│   └── Artist3 - Track3.wav
├── Playlist Name 2/
│   ├── Artist4 - Track4.wav
│   └── Artist5 - Track5.wav
└── ...
```

## Usage Examples

### Basic Playlist Download
```python
from utils.spotify_downloader import SpotifyPlaylistDownloader

downloader = SpotifyPlaylistDownloader()
files = downloader.download_playlist(
    "https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd"
)

print(f"Downloaded {len(files)} tracks:")
for file_path in files:
    print(f"  - {os.path.basename(file_path)}")
```

### Custom Base Directory
```python
downloader = SpotifyPlaylistDownloader(base_dir="/path/to/music")
files = downloader.download_playlist(spotify_url, format="wav")
```

### Integration with DJ Mix Generator
```python
# Automatic integration through CLI
python dj_mix_generator.py --spotify-playlist="<URL>" --transition-measures=16

# Programmatic usage
cli = DJMixGeneratorCLI()
files = cli._download_spotify_playlist(spotify_url)
```

### Error Handling
```python
try:
    files = downloader.download_playlist(url)
    if not files:
        print("No tracks were downloaded")
    else:
        print(f"Successfully downloaded to: {downloader.download_dir}")
except ValueError as e:
    print(f"Download failed: {e}")
    # Handle specific error cases
```

## Performance Characteristics

### Download Speed
- Moderate threading (2 threads) for stability
- Automatic source fallback for failed tracks
- Skip existing files to avoid re-download
- Efficient progress tracking

### Resource Usage
- Memory-efficient streaming download
- Automatic cleanup of temporary files
- Controlled concurrent downloads
- Progress reporting without memory leaks

### Reliability
- Multiple audio source fallback
- Robust error handling and recovery
- Clear user feedback about failures
- Automatic retry logic through spotdl

The Spotify downloader provides a robust, user-friendly interface for integrating Spotify playlists into the DJ mixing workflow, with comprehensive error handling and progress feedback.