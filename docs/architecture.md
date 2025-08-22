# Architecture Overview

The DJ Mix Generator is designed with a modular architecture that separates concerns and enables extensibility. This document provides a high-level overview of the system architecture and component interactions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Interface                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   args_parser   │  │      main       │  │   Interactive   │ │
│  │                 │  │                 │  │      GUIs       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Engine                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ audio_analyzer  │  │  mix_generator  │  │   beat_utils    │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │     models      │  │     config      │  │   Cache Layer   │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Utilities & External                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ spotify_downloader│ │  key_matching   │ │audio_processing │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    External     │  │     librosa     │  │   soundfile     │ │
│  │   (spotdl)      │  │    (analysis)   │  │    (I/O)        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. CLI Interface Layer

**`src/cli/`**
- **Entry Point**: Handles command-line argument parsing and user interaction
- **Responsibility**: User interface, input validation, workflow orchestration
- **Key Files**: `main.py`, `args_parser.py`

### 2. Core Engine Layer

**`src/core/`**
- **Processing Engine**: Core audio analysis and mixing functionality
- **Responsibility**: Audio processing, beat detection, mix generation
- **Key Files**: `mix_generator.py`, `audio_analyzer.py`, `beat_utils.py`

### 3. Utilities Layer

**`src/utils/`**
- **Support Services**: Auxiliary functionality and external integrations
- **Responsibility**: Caching, Spotify integration, key matching, audio utilities
- **Key Files**: `spotify_downloader.py`, `key_matching.py`, `cache.py`

### 4. GUI Layer

**`src/gui/`**
- **Interactive Interfaces**: User interfaces for manual control and visualization
- **Responsibility**: Beat selection, beatgrid alignment, transition mapping
- **Key Files**: `downbeat_gui.py`, `beatgrid/advanced_gui.py`

## Data Flow

### 1. Input Processing

```
Audio Files / Spotify URL
         ↓
   CLI Argument Parser
         ↓
   Track Loading & Analysis
         ↓
   Beat & Key Detection
         ↓
   Caching (Optional)
```

### 2. Mix Generation

```
Analyzed Tracks
         ↓
   Track Ordering (Optional)
         ↓
   Tempo Strategy Application
         ↓
   Beat Alignment
         ↓
   Transition Generation
         ↓
   Audio Mixing & Output
```

### 3. External Integrations

```
Spotify URL
         ↓
   SpotifyPlaylistDownloader
         ↓
   spotdl (External Tool)
         ↓
   YouTube/SoundCloud Download
         ↓
   Local WAV Files
```

## Key Design Principles

### 1. Separation of Concerns
- **CLI**: User interface and workflow
- **Core**: Audio processing logic
- **Utils**: External integrations and caching
- **GUI**: Interactive user interfaces

### 2. Modularity
- Each component has a single responsibility
- Components communicate through well-defined interfaces
- Easy to test and extend individual components

### 3. Configuration-Driven
- Centralized configuration system (`config.py`)
- Dataclasses for type safety and validation
- Easy to extend with new options

### 4. Caching Strategy
- Intelligent caching of expensive audio analysis
- File-based persistence with automatic cleanup
- Configurable cache behavior

### 5. Error Handling
- Graceful degradation when optional features fail
- Clear error messages with actionable guidance
- Robust handling of malformed audio files

## Component Dependencies

### Internal Dependencies
```
cli → core → utils
gui → core
core → models, config
utils → models
```

### External Dependencies
```
librosa    → Audio analysis and processing
soundfile  → Audio I/O operations
numpy      → Numerical computations
scipy      → Signal processing
spotdl     → Spotify playlist downloading
dearpygui  → GPU-accelerated GUI (optional)
matplotlib → Fallback GUI (optional)
```

## Extension Points

### 1. New Audio Sources
- Implement new downloaders in `utils/`
- Follow `SpotifyPlaylistDownloader` interface
- Add CLI argument in `args_parser.py`

### 2. New Tempo Strategies
- Add to `TempoStrategy` enum in `config.py`
- Implement logic in `MixGenerator`
- Update CLI help text

### 3. New Analysis Features
- Extend `Track` model in `models.py`
- Add analysis logic in `AudioAnalyzer`
- Update caching if needed

### 4. New Transition Types
- Add configuration to `TransitionSettings`
- Implement in `MixGenerator._process_track_transition()`
- Add CLI arguments

## Performance Considerations

### 1. Audio Analysis
- Most expensive operation (BPM/beat detection)
- Parallelized processing for multiple tracks
- Intelligent caching to avoid re-analysis

### 2. Memory Management
- Streaming audio processing where possible
- Efficient numpy array operations
- Garbage collection of large audio buffers

### 3. I/O Optimization
- Batch file operations
- Avoid unnecessary file reads
- Efficient temporary file handling

## Security Considerations

### 1. Input Validation
- File path sanitization
- Audio format validation
- URL validation for Spotify integration

### 2. External Tool Integration
- Safe subprocess execution
- Input sanitization for external tools
- Error handling for external tool failures

### 3. File System Access
- Controlled file creation/deletion
- Safe temporary file handling
- Proper permission management

## Testing Strategy

### 1. Unit Tests
- Individual component testing
- Mock external dependencies
- Edge case coverage

### 2. Integration Tests
- End-to-end workflow testing
- Real audio file processing
- External service integration

### 3. Performance Tests
- Large file handling
- Memory usage validation
- Processing speed benchmarks

This architecture enables the DJ Mix Generator to be maintainable, extensible, and performant while providing a clean separation between user interface, core processing, and external integrations.