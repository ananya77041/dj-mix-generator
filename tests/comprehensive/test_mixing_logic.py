#!/usr/bin/env python3
"""
Test script to debug the mixing logic with synthetic tracks
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_track(filename, duration_seconds=30, bpm=120, sr=44100):
    """Create a synthetic house music track for testing"""
    
    # Generate a simple house beat pattern
    samples_per_beat = int(sr * 60 / bpm)
    beats_per_bar = 4
    samples_per_bar = samples_per_beat * beats_per_bar
    
    total_samples = int(duration_seconds * sr)
    audio = np.zeros(total_samples)
    
    # Add kick drums on beats 1 and 3
    kick_samples = int(0.1 * sr)  # 0.1 second kick
    kick_wave = np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, kick_samples)) * np.exp(-np.linspace(0, 5, kick_samples))
    
    # Add hi-hat on beats 2 and 4  
    hihat_samples = int(0.05 * sr)  # 0.05 second hi-hat
    hihat_wave = np.random.normal(0, 0.1, hihat_samples) * np.exp(-np.linspace(0, 10, hihat_samples))
    
    # Add a different bass tone for each track to distinguish them
    filename_str = str(filename)
    if 'track_' in filename_str:
        track_num = int(filename_str.split('track_')[1].split('.')[0])  # Extract track number
    else:
        track_num = 1  # Default fallback
    bass_freq = 80 + (track_num - 1) * 20  # Different bass frequencies
    
    for beat in range(0, total_samples, samples_per_beat):
        beat_in_bar = (beat // samples_per_beat) % beats_per_bar
        
        if beat_in_bar == 0 or beat_in_bar == 2:  # Beats 1 and 3
            end_pos = min(beat + kick_samples, total_samples)
            audio[beat:end_pos] += kick_wave[:end_pos-beat] * 0.8
            
        if beat_in_bar == 1 or beat_in_bar == 3:  # Beats 2 and 4
            end_pos = min(beat + hihat_samples, total_samples)
            audio[beat:end_pos] += hihat_wave[:end_pos-beat] * 0.3
    
    # Add continuous bass line to distinguish tracks
    t = np.linspace(0, duration_seconds, total_samples)
    bass_line = 0.2 * np.sin(2 * np.pi * bass_freq * t) * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))
    audio += bass_line
    
    # Add some variation to make it sound more interesting
    audio += 0.05 * np.random.normal(0, 1, total_samples)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV
    sf.write(filename, audio, sr)
    print(f"Created {filename}: {duration_seconds}s at {bpm} BPM (bass at {bass_freq}Hz)")

def main():
    """Create test tracks"""
    test_dir = Path("test_tracks")
    test_dir.mkdir(exist_ok=True)
    
    # Create 4 test tracks with different characteristics - much longer for valid testing
    create_test_track(test_dir / "track_1.wav", duration_seconds=120, bpm=120)  # 2 minutes
    create_test_track(test_dir / "track_2.wav", duration_seconds=150, bpm=124)  # 2.5 minutes  
    create_test_track(test_dir / "track_3.wav", duration_seconds=180, bpm=128)  # 3 minutes
    create_test_track(test_dir / "track_4.wav", duration_seconds=135, bpm=126)  # 2.25 minutes
    
    print(f"\nTest tracks created in {test_dir}/")
    print("Run the DJ mix generator with:")
    print(f"python dj_mix_generator.py {test_dir}/track_1.wav {test_dir}/track_2.wav {test_dir}/track_3.wav {test_dir}/track_4.wav")

if __name__ == "__main__":
    main()