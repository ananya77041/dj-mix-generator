#!/usr/bin/env python3
"""
Mix generation functionality for DJ Mix Generator
"""

import librosa
import soundfile as sf
import numpy as np
import os
from typing import List, Tuple
from models import Track


class MixGenerator:
    """Handles DJ mix generation with transitions and beatmatching"""
    
    def __init__(self):
        pass
        
    def create_transition(self, track1: Track, track2: Track, transition_duration: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def generate_mix(self, tracks: List[Track], output_path: str, transition_duration: float = 30.0):
        """Generate the complete DJ mix"""
        if len(tracks) < 2:
            raise ValueError("Need at least 2 tracks to create a mix")
        
        print(f"Generating mix with {len(tracks)} tracks...")
        print(f"Transition duration: {transition_duration}s\n")
        
        # Start with the first track
        mix_audio = tracks[0].audio.copy()
        current_sr = tracks[0].sr
        
        print(f"Track 1: {tracks[0].filepath.name}")
        
        # Add each subsequent track with transitions
        for i in range(1, len(tracks)):
            current_track = tracks[i]
            
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
                filepath=tracks[i-1].filepath,
                audio=mix_audio,
                sr=current_sr,
                bpm=tracks[i-1].bpm,
                key=tracks[i-1].key,
                beats=tracks[i-1].beats,
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