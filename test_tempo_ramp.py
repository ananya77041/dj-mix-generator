#!/usr/bin/env python3
"""
Test script for dynamic tempo ramping with large BPM differences
"""

import numpy as np
import librosa
import soundfile as sf
import os

# Create test tracks directory
os.makedirs("tempo_ramp_tracks", exist_ok=True)

def create_synthetic_track(filename, duration_seconds, bpm, bass_freq=80):
    """Create a synthetic test track with specified BPM and bass frequency"""
    sr = 44100
    samples = int(duration_seconds * sr)
    
    # Create a simple bass drum pattern with a distinguishable bass frequency
    beat_duration = 60.0 / bpm  # Duration of one beat in seconds
    samples_per_beat = int(beat_duration * sr)
    
    # Create track with strong bass beats
    track = np.zeros(samples)
    
    for beat_sample in range(0, samples, samples_per_beat):
        if beat_sample + samples_per_beat < samples:
            # Create bass drum hit
            t = np.linspace(0, 0.1, int(0.1 * sr))  # 0.1 second bass drum
            bass_hit = np.sin(2 * np.pi * bass_freq * t) * np.exp(-t * 50)
            
            # Add the bass hit to the track
            end_idx = min(beat_sample + len(bass_hit), len(track))
            track[beat_sample:end_idx] = bass_hit[:end_idx-beat_sample]
    
    # Add subtle background texture
    background = 0.1 * np.random.normal(0, 0.01, samples)
    track += background
    
    # Normalize
    track = track / np.max(np.abs(track))
    
    # Save as WAV
    sf.write(filename, track, sr)
    
    print(f"Created {filename}: {duration_seconds}s at {bpm} BPM (bass at {bass_freq}Hz)")

# Create test tracks with BPM differences > 5 BPM to trigger tempo ramping
create_synthetic_track("tempo_ramp_tracks/track_1.wav", 120, 120, 80)   # 120 BPM
create_synthetic_track("tempo_ramp_tracks/track_2.wav", 150, 130, 100)  # 130 BPM (diff = 10)  
create_synthetic_track("tempo_ramp_tracks/track_3.wav", 180, 140, 120)  # 140 BPM (diff = 10)
create_synthetic_track("tempo_ramp_tracks/track_4.wav", 135, 110, 140)  # 110 BPM (diff = 30!)

print(f"\nTest tracks created in tempo_ramp_tracks/")
print("BPM differences that should trigger tempo ramping:")
print("  120 -> 130 BPM: diff = 10 (should trigger)")
print("  130 -> 140 BPM: diff = 10 (should trigger)")  
print("  140 -> 110 BPM: diff = 30 (should trigger)")
print("Run the DJ mix generator with:")
print("python dj_mix_generator.py tempo_ramp_tracks/track_1.wav tempo_ramp_tracks/track_2.wav tempo_ramp_tracks/track_3.wav tempo_ramp_tracks/track_4.wav")