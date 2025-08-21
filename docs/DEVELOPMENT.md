# DJ Mix Generator - Development Guide

## Project Structure

The codebase has been refactored for better maintainability, following SOLID principles and DRY methodology:

```
dj-mix-generator/
├── src/                          # Main source code
│   ├── core/                     # Core business logic
│   │   ├── config.py            # Configuration and constants
│   │   ├── models.py            # Enhanced data models
│   │   ├── audio_analyzer.py    # Audio analysis logic
│   │   ├── mix_generator.py     # Main mixing engine
│   │   └── beat_utils.py        # Beat alignment utilities
│   ├── gui/                     # GUI components
│   │   ├── base_gui.py          # Base GUI classes and utilities
│   │   ├── downbeat_gui.py      # Downbeat selection GUI
│   │   ├── transition_gui.py    # Transition downbeat GUI
│   │   └── beatgrid/            # Beatgrid alignment GUIs
│   ├── utils/                   # Utilities and helpers
│   │   ├── audio_processing.py  # Common audio processing functions
│   │   ├── cache.py             # Caching functionality
│   │   ├── key_matching.py      # Key/harmonic matching
│   │   └── file_utils.py        # File I/O utilities
│   └── cli/                     # Command-line interface
│       ├── main.py              # Main CLI application
│       └── args_parser.py       # Command-line argument parsing
├── tests/                       # All test files
├── data/                        # Test data and samples
└── docs/                        # Documentation
```

## Architecture Principles

### 1. Separation of Concerns
- **Core Logic**: Pure business logic with no UI dependencies
- **GUI Components**: Reusable GUI components with common base classes
- **Utilities**: Shared functionality used across components
- **CLI**: Command-line interface separate from core logic

### 2. DRY (Don't Repeat Yourself)
- `AudioProcessor`: Common audio processing functions
- `FrequencyProcessor`: Frequency domain operations
- `TransitionProcessor`: Specialized transition effects
- `TempoProcessor`: Tempo and rhythm processing
- `BaseGuiComponent`: Common GUI functionality

### 3. Configuration-Driven
- `MixConfiguration`: Type-safe configuration object
- `AudioQualitySettings`: Audio enhancement settings
- `TransitionSettings`: Transition configuration
- Centralized constants in `config.py`

### 4. Enhanced Models
- `Track`: Enhanced with better organization and methods
- `AudioSegment`: Represents audio segments
- `BeatInfo`: Beat and rhythm information
- `KeyInfo`: Key and harmonic information
- `MixResult`: Complete mix generation result

## Key Improvements

### Code Organization
- ✅ Modular components with clear responsibilities
- ✅ Base classes to eliminate code duplication
- ✅ Type hints throughout for better IDE support
- ✅ Comprehensive docstrings and comments

### Audio Processing
- ✅ Centralized audio processing utilities
- ✅ Reusable frequency separation functions
- ✅ Improved tempo ramping with synchronization
- ✅ Enhanced crossfading algorithms

### GUI Framework
- ✅ Base GUI classes for consistency
- ✅ Multi-step workflow support
- ✅ Backend management for cross-platform compatibility
- ✅ Standardized plotting utilities

### Configuration Management
- ✅ Type-safe configuration objects
- ✅ Validation at multiple levels
- ✅ Centralized argument parsing
- ✅ Clear separation of concerns

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd dj-mix-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/comprehensive/

# Run with coverage
python -m pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking (if mypy is installed)
mypy src/
```

## Adding New Features

### 1. Core Functionality
- Add new classes to `src/core/`
- Update `models.py` if new data structures are needed
- Add constants to `config.py`
- Update configuration classes if new settings are needed

### 2. Audio Processing
- Add new processors to `src/utils/audio_processing.py`
- Follow existing patterns for processor classes
- Add comprehensive docstrings and type hints

### 3. GUI Components
- Inherit from `BaseGuiComponent` or `MultiStepGui`
- Use `AudioWaveformPlotter` for consistent plotting
- Add to appropriate subdirectory in `src/gui/`

### 4. CLI Options
- Update `args_parser.py` to add new arguments
- Update configuration classes to handle new options
- Add validation logic

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Mock external dependencies
- Focus on edge cases and error conditions

### Integration Tests (`tests/integration/`)
- Test component interactions
- Use small test data files
- Verify end-to-end workflows

### Comprehensive Tests (`tests/comprehensive/`)
- Test all parameter combinations
- Performance and stress testing
- Real-world scenario validation

## Performance Considerations

### Memory Management
- Use generators for large datasets
- Clear audio data when no longer needed
- Monitor peak memory usage in tests

### Processing Speed
- Profile critical paths with `cProfile`
- Use NumPy vectorized operations
- Consider parallel processing for independent operations

### Caching Strategy
- Cache expensive analysis results
- Implement cache invalidation
- Monitor cache size and cleanup

## Debugging and Profiling

### Debug Mode
```bash
# Enable debug logging
export DJ_MIX_DEBUG=1
python dj_mix_generator.py --debug track1.wav track2.wav
```

### Profiling
```bash
# Profile execution
python -m cProfile -o profile.stats dj_mix_generator.py track1.wav track2.wav

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run the full test suite
6. Update documentation as needed
7. Submit a pull request

## Migration from Legacy Code

The refactored codebase maintains backwards compatibility:
- Old entry point still works: `python dj_mix_generator.py`
- All command-line options preserved
- Same output file formats and quality
- Existing cache files compatible

Legacy files are preserved but should not be modified. All new development should use the refactored structure.