# Utilities Documentation

The utilities package provides supporting functionality for external integrations, caching, audio processing helpers, and specialized features like key matching.

## Overview

The utilities are designed to be modular, reusable components that enhance the core functionality without being essential to basic operation. They handle external services, optimization features, and specialized DJ techniques.

## Components

### [Spotify Integration](spotify_downloader.md)
**`spotify_downloader.py`**
- Spotify playlist URL parsing and validation
- Multi-source audio downloading via spotdl
- Automatic playlist directory organization
- Progress tracking and error handling

### [Key Matching](key_matching.md)
**`key_matching.py`**
- Circle of Fifths harmonic analysis
- Track compatibility scoring
- Optimal track ordering for harmonic mixing
- Key signature processing and validation

### [Caching System](cache.md)
**`cache.py`**
- Intelligent track analysis caching
- File-based persistence with JSON metadata
- Automatic cache validation and cleanup
- Thread-safe operations for parallel processing

### [Audio Processing](audio_processing.md)
**`audio_processing.py`**
- Low-level audio manipulation utilities
- Signal processing helpers
- Format conversion and validation
- Performance optimization functions

## Design Philosophy

### 1. Optional Enhancement
- Core functionality works without utilities
- Utilities provide performance and feature enhancements
- Graceful degradation when utilities fail

### 2. External Integration
- Clean interfaces to external services
- Robust error handling for network operations
- Configurable timeout and retry logic

### 3. Performance Optimization
- Intelligent caching to reduce computation
- Memory-efficient processing
- Parallel operation support

### 4. Modularity
- Independent utility modules
- Minimal inter-dependencies
- Easy to extend or replace

## Integration Patterns

### Service Integration
```python
# External service wrapper pattern
class SpotifyPlaylistDownloader:
    def download_playlist(self, url: str) -> List[str]:
        # Validate input
        # Call external service
        # Handle errors gracefully
        # Return standardized result
```

### Caching Pattern
```python
# Transparent caching layer
class TrackCache:
    def get_cached_analysis(self, filepath: str) -> Optional[Track]:
        # Check cache validity
        # Return cached result or None
        
    def cache_analysis(self, track: Track) -> None:
        # Store analysis result
        # Manage cache size
```

### Utility Helper Pattern
```python
# Stateless utility functions
def process_audio_segment(audio: np.ndarray, **kwargs) -> np.ndarray:
    # Apply processing
    # Return result
    # No side effects
```

## Common Usage Patterns

### Error-Safe External Calls
```python
try:
    result = external_service.call()
    return process_result(result)
except ExternalServiceError as e:
    logger.warning(f"External service failed: {e}")
    return fallback_result()
```

### Performance-Conscious Caching
```python
# Check cache first
cached_result = cache.get(key)
if cached_result and cache.is_valid(cached_result):
    return cached_result

# Perform expensive operation
result = expensive_operation()
cache.store(key, result)
return result
```

### Configurable Behavior
```python
# Allow configuration override
default_config = get_default_config()
user_config = user_preferences()
final_config = {**default_config, **user_config}
```

## Performance Characteristics

### Caching Efficiency
- File-based persistence survives application restarts
- Intelligent cache invalidation based on file modification
- Automatic cleanup of orphaned entries
- Configurable cache size limits

### Network Operations
- Timeout handling for external service calls
- Retry logic with exponential backoff
- Progress reporting for long operations
- Graceful handling of network failures

### Memory Management
- Streaming operations where possible
- Efficient data structures
- Automatic cleanup of temporary resources
- Memory usage monitoring and reporting

For detailed information about each utility component, see the individual documentation files.