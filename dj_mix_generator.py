#!/usr/bin/env python3
"""
Simple DJ Mix Generator
Takes a playlist of WAV files and creates seamless transitions
"""

import sys
import os
from typing import List
from audio_analyzer import AudioAnalyzer
from mix_generator import MixGenerator
from key_matcher import KeyMatcher
from models import Track


class DJMixGenerator:
    """Main DJ Mix Generator class that coordinates audio analysis and mix generation"""
    
    def __init__(self, use_cache: bool = True, manual_downbeats: bool = False, allow_irregular_tempo: bool = False, tempo_strategy: str = "sequential", interactive_beats: bool = False, enable_eq_matching: bool = True, enable_volume_matching: bool = True, enable_peak_alignment: bool = True, eq_strength: float = 0.5):
        self.tracks: List[Track] = []
        self.analyzer = AudioAnalyzer(use_cache=use_cache, manual_downbeats=manual_downbeats, allow_irregular_tempo=allow_irregular_tempo)
        self.mixer = MixGenerator(tempo_strategy=tempo_strategy, interactive_beats=interactive_beats, enable_eq_matching=enable_eq_matching, enable_volume_matching=enable_volume_matching, enable_peak_alignment=enable_peak_alignment, eq_strength=eq_strength)
        self.key_matcher = KeyMatcher()
        self.tempo_strategy = tempo_strategy
        self.interactive_beats = interactive_beats
    
    def load_playlist(self, filepaths: List[str]):
        """Load and analyze all tracks in the playlist"""
        self.tracks = []
        print(f"Loading playlist with {len(filepaths)} tracks...\n")
        
        for i, filepath in enumerate(filepaths, 1):
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
                
            try:
                track = self.analyzer.analyze_track(filepath)
                self.tracks.append(track)
                downbeat_count = len(track.downbeats) if len(track.downbeats) > 0 else 0
                print(f"  [{i}/{len(filepaths)}] BPM: {track.bpm:.1f}, Key: {track.key}, Duration: {track.duration:.1f}s, Downbeats: {downbeat_count}\n")
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}\n")
        
        if not self.tracks:
            raise ValueError("No valid tracks loaded!")
        
        print(f"Successfully loaded {len(self.tracks)} tracks.\n")
    
    def reorder_by_key(self):
        """Reorder tracks for optimal harmonic mixing"""
        if len(self.tracks) <= 1:
            print("Not enough tracks to reorder.\n")
            return
        
        # Show original flow analysis
        original_flow = self.key_matcher.analyze_track_flow(self.tracks)
        print(f"Original track order - Average compatibility: {original_flow['average_score']:.1f}/3.0")
        
        # Reorder tracks
        self.tracks = self.key_matcher.reorder_tracks_by_key(self.tracks)
        
        # Show new flow analysis
        new_flow = self.key_matcher.analyze_track_flow(self.tracks)
        print(f"Reordered track order - Average compatibility: {new_flow['average_score']:.1f}/3.0")
        
        if new_flow['average_score'] > original_flow['average_score']:
            print(f"Improvement: +{new_flow['average_score'] - original_flow['average_score']:.1f} compatibility points\n")
        else:
            print("Note: Reordering did not improve overall compatibility\n")
    
    def generate_mix(self, output_path: str, transition_duration: float = None, transition_measures: int = None, transitions_only: bool = False):
        """Generate the complete DJ mix or just transitions preview"""
        # Default to 8 measures if neither specified
        if transition_duration is None and transition_measures is None:
            transition_measures = 8
            
        self.mixer.generate_mix(self.tracks, output_path, transition_duration=transition_duration, 
                               transition_measures=transition_measures, transitions_only=transitions_only)
    
    def get_cache_info(self):
        """Display information about the track analysis cache"""
        if self.analyzer.cache:
            info = self.analyzer.cache.get_cache_info()
            print("\n📁 Track Analysis Cache Info:")
            print(f"  Cached tracks: {info['cached_tracks']}")
            print(f"  Cache size: {info['cache_size_mb']:.1f} MB")
            print(f"  Cache directory: {info['cache_directory']}")
        else:
            print("Cache is disabled")
    
    def clear_cache(self):
        """Clear the track analysis cache"""
        if self.analyzer.cache:
            self.analyzer.cache.clear_cache()
        else:
            print("Cache is disabled")
    
    def cleanup_cache(self):
        """Clean up orphaned cache files"""
        if self.analyzer.cache:
            self.analyzer.cache.cleanup_orphaned_files()
        else:
            print("Cache is disabled")


def main():
    """Main function with example usage"""
    if len(sys.argv) < 2:
        print("Usage: python dj_mix_generator.py [options] <track1.wav> <track2.wav> [track3.wav] ...")
        print("   or: python dj_mix_generator.py --demo")
        print("\nOptions:")
        print("  --reorder-by-key       Reorder tracks for optimal harmonic mixing")
        print("  --transitions-only     Generate only transition sections for testing (with 5s buffers)")
        print("  --manual-downbeats     Use visual interface to manually select downbeats and BPM")
        print("  --irregular-tempo      Allow non-integer BPM values (use with --manual-downbeats)")
        print("  --tempo-strategy=MODE  Tempo alignment strategy: 'sequential' or 'uniform' (default: sequential)")
        print("  --transition-measures=N Transition length in measures (default: 8, overrides seconds)")
        print("  --transition-seconds=N  Transition length in seconds (default: 30, used if measures not specified)")
        print("  --interactive-beats     Use interactive beatgrid alignment GUI for transitions")
        print("  --no-eq-matching       Disable EQ matching during transitions (faster processing)")
        print("  --no-volume-matching   Disable volume normalization during transitions")
        print("  --no-peak-alignment    Disable micro peak-to-beat alignment (faster processing)")
        print("  --eq-strength=N        EQ matching strength: 0.0-1.0 (default: 0.5, 0=off)")
        print("  --no-cache             Disable track analysis caching")
        print("  --cache-info           Show cache information and exit")
        print("  --clear-cache          Clear track analysis cache and exit")
        print("  --cleanup-cache        Clean up orphaned cache files and exit")
        return
    
    # Parse command line options
    reorder_by_key = False
    transitions_only = False
    manual_downbeats = False
    allow_irregular_tempo = False
    use_cache = True
    tempo_strategy = "sequential"
    transition_measures = None
    transition_seconds = 30.0
    interactive_beats = False
    enable_eq_matching = True
    enable_volume_matching = True
    enable_peak_alignment = True
    eq_strength = 0.5
    playlist = []
    output_path = "dj_mix.wav"
    
    # Handle cache management commands first
    if "--cache-info" in sys.argv:
        dj = DJMixGenerator(use_cache=True)
        dj.get_cache_info()
        return 0
        
    if "--clear-cache" in sys.argv:
        dj = DJMixGenerator(use_cache=True)
        dj.clear_cache()
        return 0
        
    if "--cleanup-cache" in sys.argv:
        dj = DJMixGenerator(use_cache=True)
        dj.cleanup_cache()
        return 0
    
    if sys.argv[1] == "--demo":
        # Demo mode - use example tracks (you'll need to provide these)
        playlist = [
            "example_tracks/track1.wav",
            "example_tracks/track2.wav", 
            "example_tracks/track3.wav"
        ]
        output_path = "demo_mix.wav"
    else:
        # Parse arguments
        args = sys.argv[1:]
        
        # Check for options
        if "--reorder-by-key" in args:
            reorder_by_key = True
            args.remove("--reorder-by-key")
        
        if "--transitions-only" in args:
            transitions_only = True
            args.remove("--transitions-only")
            output_path = "dj_transitions_preview.wav"  # Different filename for transitions
        
        if "--manual-downbeats" in args:
            manual_downbeats = True
            args.remove("--manual-downbeats")
        
        if "--irregular-tempo" in args:
            allow_irregular_tempo = True
            args.remove("--irregular-tempo")
        
        if "--no-cache" in args:
            use_cache = False
            args.remove("--no-cache")
        
        if "--interactive-beats" in args:
            interactive_beats = True
            args.remove("--interactive-beats")
        
        # Audio quality options
        if "--no-eq-matching" in args:
            enable_eq_matching = False
            args.remove("--no-eq-matching")
        
        if "--no-volume-matching" in args:
            enable_volume_matching = False
            args.remove("--no-volume-matching")
        
        if "--no-peak-alignment" in args:
            enable_peak_alignment = False
            args.remove("--no-peak-alignment")
        
        # Check for EQ strength setting
        eq_strength_args = [arg for arg in args if arg.startswith("--eq-strength=")]
        if eq_strength_args:
            try:
                eq_strength = float(eq_strength_args[0].split("=", 1)[1])
                if not 0.0 <= eq_strength <= 1.0:
                    print("Error: EQ strength must be between 0.0 and 1.0.")
                    return 1
                if eq_strength == 0.0:
                    enable_eq_matching = False  # Disable EQ matching if strength is 0
                args.remove(eq_strength_args[0])
            except ValueError:
                print("Error: EQ strength must be a valid number between 0.0 and 1.0.")
                return 1
        
        # Check for tempo strategy
        tempo_strategy_args = [arg for arg in args if arg.startswith("--tempo-strategy=")]
        if tempo_strategy_args:
            tempo_strategy = tempo_strategy_args[0].split("=", 1)[1].lower()
            if tempo_strategy not in ["sequential", "uniform"]:
                print(f"Error: Invalid tempo strategy '{tempo_strategy}'. Use 'sequential' or 'uniform'.")
                return 1
            args.remove(tempo_strategy_args[0])
        
        # Check for transition measures
        measures_args = [arg for arg in args if arg.startswith("--transition-measures=")]
        if measures_args:
            try:
                transition_measures = int(measures_args[0].split("=", 1)[1])
                if transition_measures < 1:
                    print("Error: Transition measures must be at least 1.")
                    return 1
                args.remove(measures_args[0])
            except ValueError:
                print("Error: Transition measures must be a valid integer.")
                return 1
        
        # Check for transition seconds (only used if measures not specified)
        seconds_args = [arg for arg in args if arg.startswith("--transition-seconds=")]
        if seconds_args:
            try:
                transition_seconds = float(seconds_args[0].split("=", 1)[1])
                if transition_seconds < 1.0:
                    print("Error: Transition seconds must be at least 1.0.")
                    return 1
                args.remove(seconds_args[0])
            except ValueError:
                print("Error: Transition seconds must be a valid number.")
                return 1
        
        # Remaining arguments are track files
        playlist = args
        
        if not playlist:
            print("Error: No track files specified")
            return 1
    
    try:
        dj = DJMixGenerator(use_cache=use_cache, manual_downbeats=manual_downbeats, allow_irregular_tempo=allow_irregular_tempo, tempo_strategy=tempo_strategy, interactive_beats=interactive_beats, enable_eq_matching=enable_eq_matching, enable_volume_matching=enable_volume_matching, enable_peak_alignment=enable_peak_alignment, eq_strength=eq_strength)
        dj.load_playlist(playlist)
        
        # Optionally reorder by key
        if reorder_by_key:
            dj.reorder_by_key()
        
        # Generate mix with measure-based or second-based transitions
        if transition_measures is not None:
            print(f"Using {transition_measures} measure transitions")
            dj.generate_mix(output_path, transition_measures=transition_measures, transitions_only=transitions_only)
        else:
            print(f"Using {transition_seconds} second transitions")
            dj.generate_mix(output_path, transition_duration=transition_seconds, transitions_only=transitions_only)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())