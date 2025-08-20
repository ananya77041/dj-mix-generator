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
from beat_utils import BeatAligner


class MixGenerator:
    """Handles DJ mix generation with transitions and beatmatching"""
    
    def __init__(self, tempo_strategy: str = "sequential"):
        self.beat_aligner = BeatAligner()
        self.tempo_strategy = tempo_strategy
        self.target_bpm = None  # Will be set based on strategy
    
    def calculate_target_bpm(self, tracks: List[Track]):
        """Calculate target BPM based on tempo strategy"""
        if self.tempo_strategy == "sequential":
            # Sequential: use first track's BPM
            self.target_bpm = tracks[0].bpm
            print(f"Sequential tempo strategy: Target BPM = {self.target_bpm:.1f} (from first track)")
        elif self.tempo_strategy == "uniform":
            # Uniform: use average of all BPMs
            self.target_bpm = sum(track.bpm for track in tracks) / len(tracks)
            print(f"Uniform tempo strategy: Target BPM = {self.target_bpm:.1f} (average of all tracks)")
            bpm_list = [f"{track.bpm:.1f}" for track in tracks]
            print(f"  Individual BPMs: {', '.join(bpm_list)}")
        else:
            raise ValueError(f"Unknown tempo strategy: {self.tempo_strategy}")
        
    def create_transition(self, track1: Track, track2: Track, transition_duration: float = 30.0, 
                         stretch_track1: bool = False) -> Tuple[np.ndarray, np.ndarray, Track]:
        """Create a beat-aligned crossfade transition between two tracks"""
        if self.target_bpm is None:
            raise ValueError("Target BPM not set. Call calculate_target_bpm() first.")
        
        # Determine what needs to be stretched based on tempo strategy
        if self.tempo_strategy == "sequential":
            # Sequential: stretch track2 to match track1 (track1 stays at native tempo)
            print(f"  Sequential mode: {track2.bpm:.1f} -> {track1.bpm:.1f} (track2 stretched)")
            target_bpm_for_track2 = track1.bpm
            track1_stretched = track1  # No change to track1
            track2_stretched = self._stretch_track_to_bpm(track2, target_bpm_for_track2)
            
        elif self.tempo_strategy == "uniform":
            # Uniform: stretch both tracks to target BPM
            print(f"  Uniform mode: {track1.bpm:.1f} & {track2.bpm:.1f} -> {self.target_bpm:.1f} (both stretched)")
            
            if stretch_track1:
                # Stretch track1 to target BPM (for uniform mode in full mix)
                track1_stretched = self._stretch_track_to_bpm(track1, self.target_bpm)
            else:
                # Track1 already at target BPM (first track in uniform mode)
                track1_stretched = track1
                
            track2_stretched = self._stretch_track_to_bpm(track2, self.target_bpm)
        
        return self._create_crossfade(track1_stretched, track2_stretched, transition_duration)
    
    def _stretch_track_to_bpm(self, track: Track, target_bpm: float) -> Track:
        """Stretch a track to match target BPM"""
        bpm_ratio = track.bpm / target_bpm  # Ratio to stretch to target tempo
        
        if abs(bpm_ratio - 1.0) > 0.05:  # Only stretch if BPMs differ significantly
            print(f"    Time-stretching {track.filepath.name}: {track.bpm:.1f} -> {target_bpm:.1f} (ratio: {bpm_ratio:.3f})")
            stretched_audio = librosa.effects.time_stretch(track.audio, rate=bpm_ratio)
            
            # Adjust timing information
            stretched_beats = track.beats / bpm_ratio  # Beats get closer together when tempo increases
            stretched_downbeats = track.downbeats / bpm_ratio
            
            return Track(
                filepath=track.filepath,
                audio=stretched_audio,
                sr=track.sr,
                bpm=target_bpm,
                key=track.key,
                beats=stretched_beats,
                downbeats=stretched_downbeats,
                duration=len(stretched_audio) / track.sr
            )
        else:
            print(f"    {track.filepath.name}: No stretching needed ({track.bpm:.1f} ≈ {target_bpm:.1f})")
            return track
    
    def _create_crossfade(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, np.ndarray, Track]:
        """Create crossfade transition between two tempo-matched tracks"""
        # Find optimal transition points using downbeat alignment
        print(f"  Aligning to downbeats...")
        track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
            track1, track2, transition_duration
        )
        
        # Extract beat-aligned audio segments
        track1_outro, track2_intro = self.beat_aligner.align_beats_for_transition(
            track1, track2, track1_end_sample, track2_start_sample, transition_duration
        )
        
        # Report the alignment
        track1_end_time = track1_end_sample / track1.sr
        track2_start_time = track2_start_sample / track2.sr
        print(f"  Track 1 outro starts at: {track1_end_time - transition_duration:.1f}s, ends at: {track1_end_time:.1f}s")
        print(f"  Track 2 intro starts at: {track2_start_time:.1f}s")
        
        # Create crossfade with beat alignment
        if len(track1_outro) == 0 or len(track2_intro) == 0:
            print("  Warning: Empty audio segments, using fallback transition")
            return self._create_fallback_transition(track1, track2, transition_duration)
        
        # Ensure both segments are the same length
        min_length = min(len(track1_outro), len(track2_intro))
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create equal-power crossfade curves
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        # Apply crossfade
        transition = track1_outro * fade_out + track2_intro * fade_in
        
        return transition, track2
    
    def _create_fallback_transition(self, track1: Track, track2: Track, transition_duration: float) -> Tuple[np.ndarray, Track]:
        """
        Fallback transition method using simple time-based alignment
        Used when downbeat alignment fails
        """
        print("  Using fallback transition method")
        
        transition_samples = int(transition_duration * track1.sr)
        
        # Simple approach: use end of track1 and beginning of track2
        track1_outro = track1.audio[-transition_samples:] if len(track1.audio) >= transition_samples else track1.audio
        track2_intro = track2.audio[:transition_samples] if len(track2.audio) >= transition_samples else track2.audio
        
        # Ensure both segments are the same length
        min_length = min(len(track1_outro), len(track2_intro))
        track1_outro = track1_outro[:min_length]
        track2_intro = track2_intro[:min_length]
        
        # Create crossfade
        fade_samples = min_length
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
        
        transition = track1_outro * fade_out + track2_intro * fade_in
        
        return transition, track2
    
    def generate_mix(self, tracks: List[Track], output_path: str, transition_duration: float = 30.0, transitions_only: bool = False):
        """Generate the complete DJ mix or just the transitions"""
        if len(tracks) < 2:
            raise ValueError("Need at least 2 tracks to create a mix")
        
        # Calculate target BPM based on strategy
        self.calculate_target_bpm(tracks)
        
        if transitions_only:
            print("Generating transitions-only preview with 5s buffers...")
        else:
            print(f"Generating mix with {len(tracks)} tracks...")
        print(f"Transition duration: {transition_duration}s")
        print(f"Tempo strategy: {self.tempo_strategy}\n")
        
        if transitions_only:
            return self._generate_transitions_only(tracks, output_path, transition_duration)
        
        # For uniform strategy, stretch the first track to target BPM
        if self.tempo_strategy == "uniform":
            first_track = self._stretch_track_to_bpm(tracks[0], self.target_bpm)
            print(f"Track 1: {first_track.filepath.name} (stretched to target BPM)")
        else:
            first_track = tracks[0]
            print(f"Track 1: {first_track.filepath.name} (native BPM)")
        
        # Start with the first track
        mix_audio = first_track.audio.copy()
        current_sr = first_track.sr
        current_bpm = first_track.bpm
        
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
                bpm=current_bpm,  # Use current mix BPM
                key=tracks[i-1].key,
                beats=tracks[i-1].beats,
                downbeats=tracks[i-1].downbeats,
                duration=len(mix_audio) / current_sr
            )
            
            # Create transition (stretch_track1 is always False since prev_track is already processed)
            transition, stretched_track = self.create_transition(prev_track, current_track, transition_duration, stretch_track1=False)
            
            # Update current BPM for next iteration
            current_bpm = stretched_track.bpm
            
            # Remove the outro from the mix and add the transition + new track
            transition_samples = len(transition)
            mix_audio = mix_audio[:-transition_samples]  # Remove outro
            mix_audio = np.concatenate([mix_audio, transition, stretched_track.audio[transition_samples:]])
            
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
    
    def _generate_transitions_only(self, tracks: List[Track], output_path: str, transition_duration: float):
        """
        Generate only the transition sections with 5-second buffers for preview testing
        Each transition includes 5s from end of current track + transition + 5s from start of next track
        """
        print("Creating transitions-only preview for testing...")
        
        current_sr = tracks[0].sr
        buffer_duration = 5.0  # 5 seconds before/after each transition
        buffer_samples = int(buffer_duration * current_sr)
        
        all_transitions = []
        silence_gap = np.zeros(int(1.0 * current_sr))  # 1 second gap between transitions
        
        for i in range(len(tracks) - 1):
            current_track = tracks[i]
            next_track = tracks[i + 1]
            
            print(f"Transition {i+1}: {current_track.filepath.name} → {next_track.filepath.name}")
            
            # Ensure sample rates match
            if next_track.sr != current_sr:
                print(f"  Resampling {next_track.filepath.name} from {next_track.sr}Hz to {current_sr}Hz")
                next_track.audio = librosa.resample(
                    next_track.audio, 
                    orig_sr=next_track.sr, 
                    target_sr=current_sr
                )
                next_track.sr = current_sr
            
            # Create a mock "previous track state" for transition generation
            # Use the full current track as if it were the current mix state
            prev_track = Track(
                filepath=current_track.filepath,
                audio=current_track.audio,
                sr=current_sr,
                bpm=current_track.bpm,
                key=current_track.key,
                beats=current_track.beats,
                downbeats=current_track.downbeats,
                duration=current_track.duration
            )
            
            # Generate the transition
            transition, stretched_track = self.create_transition(prev_track, next_track, transition_duration, stretch_track1=False)
            
            # Find the transition points in the original tracks
            track1_end_sample, track2_start_sample = self.beat_aligner.find_optimal_transition_points(
                prev_track, stretched_track, transition_duration
            )
            
            # Extract 5s buffer before transition from current track
            pre_buffer_start = max(0, track1_end_sample - int(transition_duration * current_sr) - buffer_samples)
            pre_buffer_end = max(0, track1_end_sample - int(transition_duration * current_sr))
            pre_buffer = current_track.audio[pre_buffer_start:pre_buffer_end]
            
            # Extract 5s buffer after transition from next track
            post_buffer_start = track2_start_sample + int(transition_duration * current_sr)
            post_buffer_end = min(len(stretched_track.audio), post_buffer_start + buffer_samples)
            post_buffer = stretched_track.audio[post_buffer_start:post_buffer_end]
            
            # Combine: pre-buffer + transition + post-buffer
            if len(pre_buffer) > 0 and len(transition) > 0 and len(post_buffer) > 0:
                transition_segment = np.concatenate([pre_buffer, transition, post_buffer])
                all_transitions.append(transition_segment)
                
                segment_duration = len(transition_segment) / current_sr
                print(f"  Transition segment: {segment_duration:.1f}s (5s + {transition_duration}s + 5s)")
            else:
                print(f"  Warning: Could not create full transition segment")
                if len(transition) > 0:
                    all_transitions.append(transition)
        
        if not all_transitions:
            raise ValueError("No transitions could be generated")
        
        # Combine all transitions with silence gaps
        final_mix = all_transitions[0]
        for transition in all_transitions[1:]:
            final_mix = np.concatenate([final_mix, silence_gap, transition])
        
        # Normalize audio
        max_val = np.max(np.abs(final_mix))
        if max_val > 0.95:
            print(f"Normalizing transitions preview (peak: {max_val:.3f})")
            final_mix = final_mix * (0.95 / max_val)
        
        # Save transitions-only mix
        print(f"Saving transitions preview to: {output_path}")
        sf.write(output_path, final_mix, current_sr)
        
        duration_minutes = len(final_mix) / current_sr / 60
        print(f"\n🎵 Transitions preview complete!")
        print(f"Contains {len(all_transitions)} transitions")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Sample rate: {current_sr} Hz")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print("Listen to this file to test transition quality before generating full mix!")