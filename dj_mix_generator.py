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
    
    def __init__(self):
        self.tracks: List[Track] = []
        self.analyzer = AudioAnalyzer()
        self.mixer = MixGenerator()
        self.key_matcher = KeyMatcher()
    
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
    
    def generate_mix(self, output_path: str, transition_duration: float = 30.0):
        """Generate the complete DJ mix"""
        self.mixer.generate_mix(self.tracks, output_path, transition_duration)


def main():
    """Main function with example usage"""
    if len(sys.argv) < 2:
        print("Usage: python dj_mix_generator.py [options] <track1.wav> <track2.wav> [track3.wav] ...")
        print("   or: python dj_mix_generator.py --demo")
        print("\nOptions:")
        print("  --reorder-by-key    Reorder tracks for optimal harmonic mixing")
        return
    
    # Parse command line options
    reorder_by_key = False
    playlist = []
    output_path = "dj_mix.wav"
    
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
        
        # Remaining arguments are track files
        playlist = args
        
        if not playlist:
            print("Error: No track files specified")
            return 1
    
    try:
        dj = DJMixGenerator()
        dj.load_playlist(playlist)
        
        # Optionally reorder by key
        if reorder_by_key:
            dj.reorder_by_key()
        
        dj.generate_mix(output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())