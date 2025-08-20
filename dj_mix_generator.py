#!/usr/bin/env python3
"""
Simple DJ Mix Generator
Takes a playlist of WAV files and creates seamless transitions
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import sys
import os


@dataclass
class Track:
    """Represents an audio track with its analysis data"""
    filepath: Path
    audio: np.ndarray
    sr: int
    bpm: float
    key: str
    beats: np.ndarray
    duration: float


class DJMixGenerator:
    def __init__(self):
        self.tracks: List[Track] = []
    
    def analyze_track(self, filepath: str) -> Track:
        """Analyze a single audio file for BPM, key, and beat positions"""
        print(f"Analyzing: {os.path.basename(filepath)}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(filepath, sr=None)
            duration = len(audio) / sr
            
            # BPM detection
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            bpm = float(tempo)
            
            # Key detection (simplified chromagram approach)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            key = self._estimate_key(chroma)
            
            return Track(
                filepath=Path(filepath),
                audio=audio,
                sr=sr,
                bpm=bpm,
                key=key,
                beats=beats,
                duration=duration
            )
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            raise
    
    def _estimate_key(self, chroma) -> str:
        """Simple key estimation using chromagram"""
        # This is a simplified approach - you might want to use more sophisticated methods
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_mean = np.mean(chroma, axis=1)
        estimated_key = key_names[np.argmax(chroma_mean)]
        
        # Simple major/minor detection based on chord patterns
        # This is very basic - real key detection is much more complex
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Rotate profiles to match the detected root note
        root_idx = np.argmax(chroma_mean)
        major_rotated = np.roll(major_profile, root_idx)
        minor_rotated = np.roll(minor_profile, root_idx)
        
        # Calculate correlation with major and minor profiles
        major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
        
        # Add major/minor suffix
        if major_corr > minor_corr:
            return f"{estimated_key} major"
        else:
            return f"{estimated_key} minor"
    
    def load_playlist(self, filepaths: List[str]):
        """Load and analyze all tracks in the playlist"""
        self.tracks = []
        print(f"Loading playlist with {len(filepaths)} tracks...\n")
        
        for i, filepath in enumerate(filepaths, 1):
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
                
            try:
                track = self.analyze_track(filepath)
                self.tracks.append(track)
                print(f"  [{i}/{len(filepaths)}] BPM: {track.bpm:.1f}, Key: {track.key}, Duration: {track.duration:.1f}s\n")
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}\n")
        
        if not self.tracks:
            raise ValueError("No valid tracks loaded!")
        
        print(f"Successfully loaded {len(self.tracks)} tracks.\n")
    
    def create_transition(self, track1: Track, track2: Track, transition_duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create a crossfade transition between two tracks"""
        print(f"  BPM adjustment: {track1.bpm:.1f} -> {track2.bpm:.1f}")
        
        # Simple beatmatching - align BPMs by time-stretching
        bpm_ratio = track2.bpm / track1.bpm  # Ratio to stretch track2 to match track1's tempo
        
        # Time-stretch track2 to match track1's BPM (simplified approach)
        if abs(bpm_ratio - 1.0) > 0.05:  # Only stretch if BPMs differ significantly
            print(f"  Time-stretching track2 by ratio: {bpm_ratio:.3f}")
            track2_stretched = librosa.effects.time_stretch(track2.audio, rate=1/bpm_ratio)
        else:
            track2_stretched = track2.audio.copy()
        
        # Calculate transition samples
        transition_samples = int(transition_duration * track1.sr)
        
        # Get the outro of track1 and intro of track2
        track1_outro = track1.audio[-transition_samples:]
        track2_intro = track2_stretched[:transition_samples]
        
        # Ensure both segments are the same length
        min_length = min(len(track1_outro), len(track2_intro))
        if min_length < transition_samples // 2:
            print(f"  Warning: Short transition ({min_length/track1.sr:.1f}s)")
        
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create crossfade curves (equal power crossfade)
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade
        transition = track1_outro * fade_out + track2_intro * fade_in
        
        return transition, track2_stretched
    
    def generate_mix(self, output_path: str, transition_duration: float = 8.0):
        """Generate the complete DJ mix"""
        if len(self.tracks) < 2:
            raise ValueError("Need at least 2 tracks to create a mix")
        
        print(f"Generating mix with {len(self.tracks)} tracks...")
        print(f"Transition duration: {transition_duration}s\n")
        
        # Start with the first track
        mix_audio = self.tracks[0].audio.copy()
        current_sr = self.tracks[0].sr
        
        print(f"Track 1: {self.tracks[0].filepath.name}")
        
        # Add each subsequent track with transitions
        for i in range(1, len(self.tracks)):
            current_track = self.tracks[i]
            
            print(f"Track {i+1}: {current_track.filepath.name}")
            
            # Verify sample rates match
            if current_track.sr != current_sr:
                print(f"  Resampling from {current_track.sr}Hz to {current_sr}Hz")
                current_track.audio = librosa.resample(current_track.audio, 
                                                     orig_sr=current_track.sr, 
                                                     target_sr=current_sr)
                current_track.sr = current_sr
            
            # Create a temporary track object for the current mix state
            prev_track = Track(
                filepath=self.tracks[i-1].filepath,
                audio=mix_audio,
                sr=current_sr,
                bpm=self.tracks[i-1].bpm,
                key=self.tracks[i-1].key,
                beats=self.tracks[i-1].beats,
                duration=len(mix_audio) / current_sr
            )
            
            # Create transition
            transition, stretched_track = self.create_transition(prev_track, current_track, transition_duration)
            
            # Remove the outro from the mix and add the transition + new track
            transition_samples = len(transition)
            mix_audio = mix_audio[:-transition_samples]  # Remove outro
            mix_audio = np.concatenate([mix_audio, transition, stretched_track[transition_samples:]])
            
            print(f"  Mix length so far: {len(mix_audio) / current_sr / 60:.1f} minutes\n")
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(mix_audio))
        if max_val > 0.95:
            print(f"Normalizing audio (peak: {max_val:.3f})")
            mix_audio = mix_audio * (0.95 / max_val)
        
        # Save the final mix
        print(f"Saving mix to: {output_path}")
        sf.write(output_path, mix_audio, current_sr)
        
        duration_minutes = len(mix_audio) / current_sr / 60
        print(f"\nMix complete! 🎵")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")


def main():
    """Main function with example usage"""
    if len(sys.argv) < 2:
        print("Usage: python dj_mix_generator.py <track1.wav> <track2.wav> [track3.wav] ...")
        print("   or: python dj_mix_generator.py --demo")
        return
    
    if sys.argv[1] == "--demo":
        # Demo mode - use example tracks (you'll need to provide these)
        playlist = [
            "example_tracks/track1.wav",
            "example_tracks/track2.wav", 
            "example_tracks/track3.wav"
        ]
        output_path = "demo_mix.wav"
    else:
        # Use command line arguments as playlist
        playlist = sys.argv[1:]
        output_path = "dj_mix.wav"
    
    try:
        dj = DJMixGenerator()
        dj.load_playlist(playlist)
        dj.generate_mix(output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())