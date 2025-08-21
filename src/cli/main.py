#!/usr/bin/env python3
"""
Main CLI entry point for DJ Mix Generator
Refactored for better organization and maintainability
"""

import sys
import os
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import based on execution context
try:
    # Try relative imports first (when imported as package)
    from .args_parser import parse_command_line
    from ..core.config import MixConfiguration, FileConstants
    from ..core.audio_analyzer import AudioAnalyzer
    from ..core.mix_generator import MixGenerator
    from ..utils.key_matching import KeyMatcher
    from ..core.models import Track
except ImportError:
    # Fallback for direct execution - ensure src is in path
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from cli.args_parser import parse_command_line
    from core.config import MixConfiguration, FileConstants
    from core.audio_analyzer import AudioAnalyzer
    from core.mix_generator import MixGenerator
    from utils.key_matching import KeyMatcher
    from core.models import Track


class DJMixGeneratorCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.tracks: List[Track] = []
        self.config: MixConfiguration = None
        self.analyzer: AudioAnalyzer = None
        self.mixer: MixGenerator = None
        self.key_matcher: KeyMatcher = None
    
    def run(self, args: List[str] = None) -> int:
        """Main entry point"""
        try:
            # Parse command line arguments
            self.config, track_paths = parse_command_line()
            
            # Initialize components
            self._initialize_components()
            
            # Load and analyze tracks
            self._load_playlist(track_paths)
            
            # Apply track ordering (random order overrides all others)
            if self.config.random_order is not None:
                self._randomize_tracks()
                # Apply additional sorting to the final selected tracks if requested
                if self.config.bpm_sort:
                    self._sort_by_bpm()
                elif self.config.reorder_by_key:
                    self._reorder_tracks_by_key()
            elif self.config.bpm_sort:
                self._sort_by_bpm()
            elif self.config.reorder_by_key:
                self._reorder_tracks_by_key()
            
            # Generate the mix
            output_path = self._get_output_path()
            self._generate_mix(output_path)
            
            print("✅ Mix generation completed successfully!")
            return 0
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    def _initialize_components(self):
        """Initialize analysis and mixing components"""
        self.analyzer = AudioAnalyzer(
            use_cache=self.config.use_cache,
            manual_downbeats=self.config.manual_downbeats,
            allow_irregular_tempo=self.config.allow_irregular_tempo
        )
        
        self.mixer = MixGenerator(
            config=self.config
        )
        
        self.key_matcher = KeyMatcher()
    
    def _load_playlist(self, filepaths: List[str]):
        """Load and analyze all tracks in the playlist"""
        self.tracks = []
        print(f"Loading playlist with {len(filepaths)} tracks...\\n")
        
        # Check if we should use parallel processing
        if not self.config.manual_downbeats and len(filepaths) > 1:
            self._load_playlist_parallel(filepaths)
        else:
            self._load_playlist_sequential(filepaths)
        
        if not self.tracks:
            raise ValueError("No valid tracks loaded!")
        
        print(f"Successfully loaded {len(self.tracks)} tracks.\\n")
        
        # Sort tracks by BPM and key for optimal mixing flow
        self._sort_tracks_by_bpm_and_key()
    
    def _load_playlist_sequential(self, filepaths: List[str]):
        """Load tracks sequentially (for manual mode or single track)"""
        for i, filepath in enumerate(filepaths, 1):
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
                
            try:
                track = self.analyzer.analyze_track(filepath)
                self.tracks.append(track)
                downbeat_count = len(track.downbeats) if len(track.downbeats) > 0 else 0
                print(f"  [{i}/{len(filepaths)}] BPM: {track.bpm:.1f}, Key: {track.key}, "
                      f"Duration: {track.duration:.1f}s, Downbeats: {downbeat_count}\\n")
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}\\n")
    
    def _load_playlist_parallel(self, filepaths: List[str]):
        """Load tracks in parallel for automatic analysis mode"""
        print("Using parallel processing for maximum compute power...")
        
        # Filter out non-existent files first
        valid_filepaths = []
        for filepath in filepaths:
            if os.path.exists(filepath):
                valid_filepaths.append(filepath)
            else:
                print(f"Warning: File not found: {filepath}")
        
        if not valid_filepaths:
            return
        
        # Use thread lock for thread-safe cache operations
        cache_lock = threading.Lock()
        
        def analyze_single_track(filepath: str) -> tuple[int, Track]:
            """Analyze a single track with thread-safe cache access"""
            try:
                # Create analyzer instance with thread-safe cache access
                analyzer = AudioAnalyzer(
                    use_cache=self.config.use_cache,
                    manual_downbeats=False,  # Always false for parallel processing
                    allow_irregular_tempo=self.config.allow_irregular_tempo
                )
                
                # Use cache lock for thread safety
                if analyzer.use_cache and analyzer.cache:
                    with cache_lock:
                        cached_track = analyzer.cache.get_cached_analysis(filepath, False)
                    if cached_track is not None:
                        print(f"  ✓ {os.path.basename(filepath)} loaded from cache")
                        return (filepaths.index(filepath), cached_track)
                
                # Perform analysis
                track = analyzer.analyze_track(filepath)
                
                # Cache results with thread safety
                if analyzer.use_cache and analyzer.cache:
                    with cache_lock:
                        analyzer.cache.cache_analysis(track, False)
                
                return (filepaths.index(filepath), track)
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}")
                return (filepaths.index(filepath), None)
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(len(valid_filepaths), os.cpu_count() or 4)
        track_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_filepath = {executor.submit(analyze_single_track, filepath): filepath 
                                for filepath in valid_filepaths}
            
            # Collect results as they complete
            completed = 0
            total = len(valid_filepaths)
            
            for future in as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    original_index, track = future.result()
                    if track is not None:
                        track_results[original_index] = track
                        downbeat_count = len(track.downbeats) if len(track.downbeats) > 0 else 0
                        completed += 1
                        print(f"  [{completed}/{total}] {os.path.basename(filepath)}: "
                              f"BPM: {track.bpm:.1f}, Key: {track.key}, "
                              f"Duration: {track.duration:.1f}s, Downbeats: {downbeat_count}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        
        # Sort tracks by original order and add to self.tracks
        for i in sorted(track_results.keys()):
            self.tracks.append(track_results[i])
        
        print(f"\\nParallel analysis completed: {len(self.tracks)} tracks loaded")
    
    def _sort_tracks_by_bpm_and_key(self):
        """Sort tracks by BPM (ascending), then by key within same BPM"""
        if len(self.tracks) <= 1:
            return
            
        print("Sorting tracks by BPM and key...")
        
        # Show original order
        print("Original order:")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        
        # Sort by BPM first, then by key within same BPM
        self.tracks.sort(key=lambda track: (track.bpm, self.key_matcher._get_key_sort_value(track.key)))
        
        print("\\nSorted order (by BPM ↑, then by key):")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        print()
    
    def _randomize_tracks(self):
        """Randomly select and randomize track order"""
        if len(self.tracks) <= 1:
            print("Not enough tracks to randomize.\\n")
            return
        
        num_tracks = self.config.random_order
        total_tracks = len(self.tracks)
        
        if num_tracks >= total_tracks:
            print(f"Requested {num_tracks} tracks, but only {total_tracks} available. Using all tracks.")
            num_tracks = total_tracks
        
        print(f"Randomly selecting {num_tracks} tracks from {total_tracks} available...")
        
        # Show original order
        print("Available tracks:")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        
        # Randomly select the specified number of tracks
        selected_tracks = random.sample(self.tracks, num_tracks)
        
        # Randomize the order of selected tracks
        random.shuffle(selected_tracks)
        
        # Update the track list
        self.tracks = selected_tracks
        
        print(f"\\nSelected and randomized {len(self.tracks)} tracks:")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        print()

    def _sort_by_bpm(self):
        """Sort tracks by BPM only (ascending order)"""
        if len(self.tracks) <= 1:
            print("Not enough tracks to sort by BPM.\\n")
            return
        
        print("Sorting tracks by BPM...")
        
        # Show original order
        print("Original order:")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        
        # Sort by BPM only (ascending)
        self.tracks.sort(key=lambda track: track.bpm)
        
        print("\\nSorted by BPM (ascending):")
        for i, track in enumerate(self.tracks, 1):
            print(f"  [{i}] {track.filepath.name}: {track.bpm:.1f} BPM, {track.key}")
        print()

    def _reorder_tracks_by_key(self):
        """Reorder tracks for optimal harmonic mixing"""
        if len(self.tracks) <= 1:
            print("Not enough tracks to reorder.\\n")
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
            print(f"Improvement: +{new_flow['average_score'] - original_flow['average_score']:.1f} compatibility points\\n")
        else:
            print("Note: Reordering did not improve overall compatibility\\n")
    
    def _get_output_path(self) -> str:
        """Get the appropriate output file path"""
        if self.config.transitions_only:
            return FileConstants.TRANSITIONS_OUTPUT_NAME
        else:
            return FileConstants.DEFAULT_OUTPUT_NAME
    
    def _generate_mix(self, output_path: str):
        """Generate the complete DJ mix or transitions preview"""
        transition_duration = None
        transition_measures = None
        
        # Determine transition length
        if self.config.transition_settings.measures is not None:
            transition_measures = self.config.transition_settings.measures
            print(f"Using {transition_measures} measure transitions")
        else:
            transition_duration = self.config.transition_settings.seconds or 30.0
            print(f"Using {transition_duration} second transitions")
        
        # Generate the mix
        self.mixer.generate_mix(
            self.tracks,
            output_path,
            transition_duration=transition_duration,
            transition_measures=transition_measures,
            transitions_only=self.config.transitions_only
        )


def main():
    """Main entry point for CLI"""
    cli = DJMixGeneratorCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())