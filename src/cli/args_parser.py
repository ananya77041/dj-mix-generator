#!/usr/bin/env python3
"""
Command-line argument parser
Centralized argument parsing with validation
"""

import sys
import argparse
from typing import List, Optional
from pathlib import Path
try:
    from core.config import MixConfiguration, TempoStrategy, TransitionSettings, AudioQualitySettings
except ImportError:
    from ..core.config import MixConfiguration, TempoStrategy, TransitionSettings, AudioQualitySettings


class ArgumentParser:
    """Enhanced argument parser with validation and configuration building"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options"""
        parser = argparse.ArgumentParser(
            description='Professional DJ Mix Generator with advanced beat matching and transitions',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )
        
        # Positional arguments
        parser.add_argument('tracks', nargs='*', help='Audio track files (.wav)')
        
        # Demo mode
        parser.add_argument('--demo', action='store_true',
                           help='Run demo mode with example tracks')
        
        # Tempo strategies
        parser.add_argument('--tempo-strategy', choices=['sequential', 'uniform', 'match-track'],
                           default='uniform', help='Tempo alignment strategy (default: uniform)')
        
        # Transition settings
        transition_group = parser.add_argument_group('Transition Settings')
        transition_group.add_argument('--transition-measures', type=int,
                                    help='Transition length in measures (overrides seconds)')
        transition_group.add_argument('--transition-seconds', type=float,
                                    help='Transition length in seconds (default: 30)')
        
        # Audio quality settings
        quality_group = parser.add_argument_group('Audio Quality')
        quality_group.add_argument('--no-eq-matching', action='store_true',
                                 help='Disable EQ matching during transitions')
        quality_group.add_argument('--no-volume-matching', action='store_true',
                                 help='Disable volume normalization during transitions')
        quality_group.add_argument('--no-peak-alignment', action='store_true',
                                 help='Disable micro peak-to-beat alignment')
        quality_group.add_argument('--no-tempo-correction', action='store_true',
                                 help='Disable piecewise tempo correction')
        quality_group.add_argument('--eq-strength', type=float, default=0.5,
                                 help='EQ matching strength: 0.0-1.0 (default: 0.5)')
        
        # Frequency transitions
        freq_group = parser.add_argument_group('Frequency Transitions')
        freq_group.add_argument('--lf-transition', action='store_true',
                              help='Enable low-frequency blending to prevent bass/kick clashing')
        freq_group.add_argument('--mf-transition', action='store_true',
                              help='Enable mid-frequency blending for smoother melodic transitions')
        
        # Advanced features
        advanced_group = parser.add_argument_group('Advanced Features')
        advanced_group.add_argument('--reorder-by-key', action='store_true',
                                  help='Reorder tracks for optimal harmonic mixing')
        advanced_group.add_argument('--transitions-only', action='store_true',
                                  help='Generate only transition sections for testing')
        advanced_group.add_argument('--manual-downbeats', action='store_true',
                                  help='Use visual interface to manually select downbeats')
        advanced_group.add_argument('--irregular-tempo', action='store_true',
                                  help='Allow non-integer BPM values')
        advanced_group.add_argument('--interactive-beats', action='store_true',
                                  help='Use interactive beatgrid alignment GUI')
        advanced_group.add_argument('--transition-downbeats', action='store_true',
                                  help='Use interactive GUI to select downbeats for transitions')
        
        # Cache settings
        cache_group = parser.add_argument_group('Cache Management')
        cache_group.add_argument('--no-cache', action='store_true',
                               help='Disable track analysis caching')
        cache_group.add_argument('--cache-info', action='store_true',
                               help='Show cache information and exit')
        cache_group.add_argument('--clear-cache', action='store_true',
                               help='Clear track analysis cache and exit')
        cache_group.add_argument('--cleanup-cache', action='store_true',
                               help='Clean up orphaned cache files and exit')
        
        return parser
    
    def _get_examples_text(self) -> str:
        """Get examples text for help"""
        return """
Examples:
  # Basic mixing
  python dj_mix_generator.py track1.wav track2.wav track3.wav
  
  # Harmonic mixing with preview
  python dj_mix_generator.py --reorder-by-key --transitions-only track1.wav track2.wav
  
  # Advanced features
  python dj_mix_generator.py --tempo-strategy=match-track --mf-transition track1.wav track2.wav
  
  # Manual precision
  python dj_mix_generator.py --manual-downbeats --transition-downbeats track1.wav track2.wav
        """
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse arguments with validation"""
        parsed = self.parser.parse_args(args)
        
        # Validate arguments
        self._validate_args(parsed)
        
        return parsed
    
    def _validate_args(self, args: argparse.Namespace):
        """Validate parsed arguments"""
        # Validate EQ strength
        if not 0.0 <= args.eq_strength <= 1.0:
            raise ValueError("EQ strength must be between 0.0 and 1.0")
        
        # Validate transition settings
        if args.transition_measures is not None and args.transition_measures < 1:
            raise ValueError("Transition measures must be at least 1")
        
        if args.transition_seconds is not None and args.transition_seconds < 1.0:
            raise ValueError("Transition seconds must be at least 1.0")
        
        # Note: Both frequency transitions can be enabled simultaneously
        
        # Validate tracks
        if not args.demo and not args.cache_info and not args.clear_cache and not args.cleanup_cache:
            if not args.tracks:
                raise ValueError("No track files specified")
            
            # Check if files exist
            for track_path in args.tracks:
                if not Path(track_path).exists():
                    print(f"Warning: File not found: {track_path}")
    
    def create_configuration(self, args: argparse.Namespace) -> MixConfiguration:
        """Create MixConfiguration from parsed arguments"""
        # Parse tempo strategy
        tempo_strategy = TempoStrategy(args.tempo_strategy)
        
        # Create transition settings
        transition_settings = TransitionSettings(
            measures=args.transition_measures,
            seconds=args.transition_seconds,
            enable_lf_transition=args.lf_transition,
            enable_mf_transition=args.mf_transition,
            use_downbeat_mapping=args.transition_downbeats
        )
        
        # Create audio quality settings
        audio_quality = AudioQualitySettings(
            eq_matching=not args.no_eq_matching,
            volume_matching=not args.no_volume_matching,
            peak_alignment=not args.no_peak_alignment,
            tempo_correction=not args.no_tempo_correction,
            eq_strength=args.eq_strength
        )
        
        # If EQ strength is 0, disable EQ matching
        if args.eq_strength == 0.0:
            audio_quality.eq_matching = False
        
        # Create main configuration
        config = MixConfiguration(
            tempo_strategy=tempo_strategy,
            transition_settings=transition_settings,
            audio_quality=audio_quality,
            reorder_by_key=args.reorder_by_key,
            interactive_beats=args.interactive_beats,
            manual_downbeats=args.manual_downbeats,
            allow_irregular_tempo=args.irregular_tempo,
            transitions_only=args.transitions_only,
            use_cache=not args.no_cache
        )
        
        # Validate the complete configuration
        config.validate()
        
        return config
    
    def handle_special_commands(self, args: argparse.Namespace) -> bool:
        """Handle special commands that don't require full processing. Returns True if handled."""
        if args.cache_info or args.clear_cache or args.cleanup_cache:
            try:
                from utils.cache import TrackCache
            except ImportError:
                from ..utils.cache import TrackCache
            cache = TrackCache()
            
            if args.cache_info:
                cache.print_info()
            elif args.clear_cache:
                cache.clear_cache()
            elif args.cleanup_cache:
                cache.cleanup_orphaned_files()
            
            return True
        
        return False


def parse_command_line() -> tuple[MixConfiguration, List[str]]:
    """Convenience function to parse command line and return config + track paths"""
    parser = ArgumentParser()
    args = parser.parse_args()
    
    # Handle special commands
    if parser.handle_special_commands(args):
        sys.exit(0)
    
    # Handle demo mode
    if args.demo:
        demo_tracks = [
            "example_tracks/track1.wav",
            "example_tracks/track2.wav", 
            "example_tracks/track3.wav"
        ]
        return parser.create_configuration(args), demo_tracks
    
    return parser.create_configuration(args), args.tracks