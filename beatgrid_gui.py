#!/usr/bin/env python3
"""
Interactive beatgrid alignment GUI for DJ Mix Generator
Shows two tracks' beatgrids overlaid and allows manual alignment adjustment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import librosa
from typing import Optional, Callable, Tuple
from models import Track


class BeatgridAligner:
    """Interactive GUI for step-by-step beatgrid alignment: Track 1 → Track 2 → Final alignment"""
    
    def __init__(self, track1: Track, track2: Track, track1_outro: np.ndarray, track2_intro: np.ndarray,
                 outro_start: int, track2_start_sample: int, transition_duration: float):
        self.track1 = track1
        self.track2 = track2
        self.track1_outro = track1_outro
        self.track2_intro = track2_intro
        self.outro_start = outro_start
        self.track2_start_sample = track2_start_sample
        self.transition_duration = transition_duration
        
        # Calculate display parameters - show only first 4 measures for clarity
        self.beats_per_measure = 4
        self.measures_to_show = 4
        beats_per_second = track1.bpm / 60.0
        self.display_duration = (self.measures_to_show * self.beats_per_measure) / beats_per_second
        
        # Don't exceed the actual transition duration
        self.display_duration = min(self.display_duration, transition_duration)
        self.sr = track1.sr
        
        # Find the first downbeats in both tracks' transition sections
        self.track1_first_downbeat_offset = self._find_first_downbeat_offset(track1, outro_start)
        self.track2_first_downbeat_offset = self._find_first_downbeat_offset(track2, track2_start_sample)
        
        # Calculate the number of samples for the display duration
        display_samples = int(self.display_duration * self.sr)
        
        # Truncate audio to display duration, starting from first downbeat
        track1_start_sample = max(0, int(self.track1_first_downbeat_offset * self.sr))
        track2_start_sample = max(0, int(self.track2_first_downbeat_offset * self.sr))
        
        self.track1_outro_display = track1_outro[track1_start_sample:track1_start_sample + display_samples] if len(track1_outro) > track1_start_sample + display_samples else track1_outro[track1_start_sample:]
        self.track2_intro_display = track2_intro[track2_start_sample:track2_start_sample + display_samples] if len(track2_intro) > track2_start_sample + display_samples else track2_intro[track2_start_sample:]
        
        # Measure axis for the display (0 to 4 measures)
        self.measure_axis = np.linspace(0, self.measures_to_show, len(self.track1_outro_display))
        
        # Convert beat positions to time within the display window
        self.track1_beats_in_transition = self._get_beats_in_transition(track1, outro_start, display_samples)
        self.track2_beats_in_transition = self._get_beats_in_transition(track2, track2_start_sample, display_samples)
        
        # Store original beat data for precise alignment
        self.track1_all_beats = self._get_all_beats_as_time(track1)
        self.track2_all_beats = self._get_all_beats_as_time(track2)
        self.track1_all_downbeats = self._get_all_downbeats_as_time(track1) 
        self.track2_all_downbeats = self._get_all_downbeats_as_time(track2)
        
        # Workflow state management
        self.current_step = 1  # 1=Track 1 adjustment, 2=Track 2 adjustment, 3=Final alignment
        self.track1_adjusted_downbeat = None  # User-adjusted first downbeat position
        self.track2_adjusted_downbeat = None  # User-adjusted first downbeat position
        
        # Track alignment offsets (user adjustable)
        self.track1_offset = 0.0  # seconds
        self.track2_offset = 0.0  # seconds
        self.callback_func = None
        self.dragging = False
        self.dragging_track = None  # Which track is being dragged: 1 or 2
        
        # Audio playback state
        self.is_playing = False
        self.playback_position = 0.0  # Current playback position in seconds
        self.playback_start_time = None
        self.playback_line = None  # The scrolling playback indicator line
        self.playback_thread = None
        self.playback_audio = None
        self.playback_sr = None
        self.playback_stream = None  # sounddevice stream object
        self.stop_playback_flag = False  # Thread-safe stop flag
        
        # Beatgrid stretching state
        self.stretching = False
        self.stretch_anchor_measure = None
        self.original_bpm = None
        self.current_bpm = None
        
        # Setup the plot
        self._setup_plot()
    
    def _get_beats_in_transition(self, track: Track, start_sample: int, segment_length: int) -> np.ndarray:
        """Get beat times within the transition window"""
        # Convert beat frames to sample positions
        beat_samples = librosa.frames_to_samples(track.beats, hop_length=512)
        
        # Find beats within the segment
        segment_end = start_sample + segment_length
        beats_in_segment = beat_samples[
            (beat_samples >= start_sample) & (beat_samples <= segment_end)
        ]
        
        # Convert to time relative to segment start
        if len(beats_in_segment) > 0:
            beat_times = (beats_in_segment - start_sample) / self.sr
            return beat_times[beat_times <= self.display_duration]
        else:
            return np.array([])
    
    def _get_downbeats_in_transition(self, track: Track, start_sample: int, segment_length: int) -> np.ndarray:
        """Get downbeat times within the transition window"""
        # Convert downbeat frames to sample positions
        downbeat_samples = librosa.frames_to_samples(track.downbeats, hop_length=512)
        
        # Find downbeats within the segment
        segment_end = start_sample + segment_length
        downbeats_in_segment = downbeat_samples[
            (downbeat_samples >= start_sample) & (downbeat_samples <= segment_end)
        ]
        
        # Convert to time relative to segment start
        if len(downbeats_in_segment) > 0:
            downbeat_times = (downbeats_in_segment - start_sample) / self.sr
            return downbeat_times[downbeat_times <= self.display_duration]
        else:
            return np.array([])
    
    def _find_first_downbeat_offset(self, track: Track, start_sample: int) -> float:
        """Find the offset from start_sample to the first downbeat within the transition"""
        if len(track.downbeats) == 0:
            return 0.0
        
        # Convert downbeat frames to sample positions
        downbeat_samples = librosa.frames_to_samples(track.downbeats, hop_length=512)
        
        # Find the first downbeat at or after start_sample
        valid_downbeats = downbeat_samples[downbeat_samples >= start_sample]
        
        if len(valid_downbeats) > 0:
            first_downbeat_sample = valid_downbeats[0]
            offset_samples = first_downbeat_sample - start_sample
            return offset_samples / self.sr
        
        return 0.0
    
    def _get_all_beats_as_time(self, track: Track) -> np.ndarray:
        """Get all beats in the track as time values (seconds)"""
        if len(track.beats) == 0:
            return np.array([])
        beat_samples = librosa.frames_to_samples(track.beats, hop_length=512)
        return beat_samples / self.sr
    
    def _get_all_downbeats_as_time(self, track: Track) -> np.ndarray:
        """Get all downbeats in the track as time values (seconds)"""
        if len(track.downbeats) == 0:
            return np.array([])
        downbeat_samples = librosa.frames_to_samples(track.downbeats, hop_length=512)
        return downbeat_samples / self.sr
    
    def _get_beats_in_window(self, all_beats: np.ndarray, window_start: float, window_duration: float, offset: float = 0.0) -> np.ndarray:
        """Get beats within a specific time window, with optional offset, converted to measure positions"""
        if len(all_beats) == 0:
            return np.array([])
        
        # Apply offset to beat positions
        offset_beats = all_beats + offset
        
        # Find beats within the window
        window_end = window_start + window_duration
        beats_in_window = offset_beats[
            (offset_beats >= window_start) & (offset_beats <= window_end)
        ]
        
        # Convert to relative time within the window
        beats_relative_time = beats_in_window - window_start
        
        # Convert time to measures (assuming 4/4 time)
        # Use an average BPM if tracks differ slightly, or track1 BPM if they should be matched
        avg_bpm = (self.track1.bpm + self.track2.bpm) / 2.0
        beats_per_second = avg_bpm / 60.0
        beats_per_measure = self.beats_per_measure
        measures_per_second = beats_per_second / beats_per_measure
        
        beats_in_measures = beats_relative_time * measures_per_second
        
        return beats_in_measures
    
    def _get_step_instructions(self):
        """Get instructions text based on current workflow step"""
        if self.current_step == 1:
            return (
                f"STEP 1: Adjust Track 1 Beatgrid ({self.track1.filepath.name}):\\n"
                f"• CLICK & DRAG anywhere: Move the entire beatgrid for alignment\\n"
                f"• CLICK ON PURPLE LINE: Stretch/contract beatgrid to adjust tempo\\n"
                f"• PLAY BUTTON: Listen with live scrolling indicator (pause/resume)\\n" 
                f"• Blue lines = beats, Purple line = first downbeat (most important!)\\n"
                f"• Watch the BPM update live when stretching\\n"
                f"• Click 'Next Step' when perfectly aligned"
            )
        elif self.current_step == 2:
            return (
                f"STEP 2: Adjust Track 2 Beatgrid ({self.track2.filepath.name}):\\n"
                f"• CLICK & DRAG anywhere: Move the entire beatgrid for alignment\\n"
                f"• CLICK ON GREEN LINE: Stretch/contract beatgrid to adjust tempo\\n"
                f"• PLAY BUTTON: Listen with live scrolling indicator (pause/resume)\\n"
                f"• Orange lines = beats, Green line = first downbeat (most important!)\\n" 
                f"• Watch the BPM update live when stretching\\n"
                f"• Click 'Next Step' when perfectly aligned"
            )
        else:
            return (
                f"STEP 3: Final Transition Alignment:\\n"
                f"• Both tracks' downbeats and tempos are now properly adjusted\\n"
                f"• The system will create the final transition using your adjustments\\n"
                f"• Purple line (Track 1) and Green line (Track 2) mark the transition points\\n"
                f"• You can still fine-tune by clicking on downbeat lines to stretch\\n"
                f"• Click 'Confirm' to create the final aligned transition"
            )
    
    def _setup_track1_view(self):
        """Setup single track view for Track 1 adjustment"""
        # Plot track1 waveform
        self.ax_main.plot(self.measure_axis, self.track1_outro_display, color='blue', alpha=0.7, linewidth=1.0)
        self.ax_main.set_title(f'Step 1: Adjust Track 1 Beatgrid - {self.track1.filepath.name} (BPM: {self.track1.bpm:.1f})', 
                              fontsize=14, fontweight='bold', color='blue')
        self.ax_main.set_xlabel('Measures', fontsize=12)
        self.ax_main.set_ylabel('Amplitude', fontsize=10)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlim(0, self.measures_to_show)
    
    def _setup_track2_view(self):
        """Setup single track view for Track 2 adjustment"""
        # Plot track2 waveform
        self.ax_main.plot(self.measure_axis, self.track2_intro_display, color='red', alpha=0.7, linewidth=1.0)
        self.ax_main.set_title(f'Step 2: Adjust Track 2 Beatgrid - {self.track2.filepath.name} (BPM: {self.track2.bpm:.1f})', 
                              fontsize=14, fontweight='bold', color='red')
        self.ax_main.set_xlabel('Measures', fontsize=12)
        self.ax_main.set_ylabel('Amplitude', fontsize=10)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlim(0, self.measures_to_show)
    
    def _setup_dual_track_view(self):
        """Setup dual track view for final alignment"""
        tempo_matched = abs(self.track1.bpm - self.track2.bpm) < 0.1
        tempo_status = "Tempo-Matched" if tempo_matched else f"BPM: {self.track1.bpm:.1f}"
        
        # Plot track1 waveform
        self.ax_track1.plot(self.measure_axis, self.track1_outro_display, color='blue', alpha=0.7, linewidth=1.0)
        self.ax_track1.set_title(f'Step 3: Final Alignment - Track 1: {self.track1.filepath.name} ({tempo_status})', 
                                fontsize=12, fontweight='bold', color='blue')
        self.ax_track1.set_ylabel('Amplitude', fontsize=10)
        self.ax_track1.grid(True, alpha=0.3)
        self.ax_track1.set_xlim(0, self.measures_to_show)
        
        # Plot track2 waveform
        self.ax_track2.plot(self.measure_axis, self.track2_intro_display, color='red', alpha=0.7, linewidth=1.0)
        tempo_status2 = "Tempo-Matched" if tempo_matched else f"BPM: {self.track2.bpm:.1f}"
        self.ax_track2.set_title(f'Track 2: {self.track2.filepath.name} ({tempo_status2})', 
                                fontsize=12, fontweight='bold', color='red')
        self.ax_track2.set_xlabel('Measures', fontsize=12)
        self.ax_track2.set_ylabel('Amplitude', fontsize=10)
        self.ax_track2.grid(True, alpha=0.3)
        self.ax_track2.set_xlim(0, self.measures_to_show)
        
        # Link the x-axes so they zoom/pan together
        self.ax_track2.sharex(self.ax_track1)
    
    def _setup_plot(self):
        """Setup the matplotlib plot based on current workflow step"""
        if self.current_step in [1, 2]:
            # Steps 1 & 2: Single track view for individual adjustment
            self.fig, (self.ax_main, self.ax_controls) = plt.subplots(
                2, 1, figsize=(16, 10), 
                gridspec_kw={'height_ratios': [4, 1]}
            )
            self.ax_track1 = self.ax_main  # Alias for compatibility
            self.ax_track2 = None
        else:
            # Step 3: Dual track view for final alignment
            self.fig, (self.ax_track1, self.ax_track2, self.ax_controls) = plt.subplots(
                3, 1, figsize=(16, 12), 
                gridspec_kw={'height_ratios': [2.5, 2.5, 1]}
            )
            self.ax_main = self.ax_track1  # Alias for compatibility
        
        # Plot based on current workflow step
        if self.current_step == 1:
            self._setup_track1_view()
        elif self.current_step == 2:
            self._setup_track2_view()
        else:
            self._setup_dual_track_view()
        
        # Draw initial beatgrids
        self._draw_beatgrids()
        
        # Instructions based on current step
        instruction_text = self._get_step_instructions()
        
        self.ax_controls.text(0.02, 0.95, instruction_text, 
                            transform=self.ax_controls.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        # Add buttons
        self._add_buttons()
        
        # Connect interaction events
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        
        # Set window title
        self.fig.canvas.manager.set_window_title('Beatgrid Alignment Tool - Musical View (Measures)')
    
    def _draw_beatgrids(self):
        """Draw the beatgrid lines based on current workflow step"""
        # Clear existing beatgrid lines
        axes_to_clear = [self.ax_main] if self.current_step in [1, 2] else [self.ax_track1, self.ax_track2]
        
        for ax in axes_to_clear:
            if ax is not None:
                # Remove beat lines (keep only the waveform plot)
                lines_to_remove = []
                for line in ax.lines[1:]:  # Keep the first line (waveform)
                    lines_to_remove.append(line)
                for line in lines_to_remove:
                    line.remove()
        
        # Calculate window parameters for the displayed segment
        # Adjust window start to align first downbeats at measure 0
        track1_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
        track2_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
        
        # Get actual detected beats within the display window with user offsets applied
        track1_beats_in_display = self._get_beats_in_window(
            self.track1_all_beats, 
            track1_window_start, 
            self.display_duration,
            offset=self.track1_offset
        )
        
        track1_downbeats_in_display = self._get_beats_in_window(
            self.track1_all_downbeats,
            track1_window_start,
            self.display_duration,
            offset=self.track1_offset
        )
        
        # Get track2 beats with user offset applied
        track2_beats_in_display = self._get_beats_in_window(
            self.track2_all_beats,
            track2_window_start,
            self.display_duration,
            offset=self.track2_offset
        )
        
        track2_downbeats_in_display = self._get_beats_in_window(
            self.track2_all_downbeats,
            track2_window_start, 
            self.display_duration,
            offset=self.track2_offset
        )
        
        # Draw beatgrids based on current step
        if self.current_step == 1:
            self._draw_track1_beatgrid(track1_beats_in_display, track1_downbeats_in_display)
        elif self.current_step == 2:
            self._draw_track2_beatgrid(track2_beats_in_display, track2_downbeats_in_display)
        else:
            self._draw_dual_beatgrids(track1_beats_in_display, track1_downbeats_in_display, 
                                    track2_beats_in_display, track2_downbeats_in_display)
        
        # Add alignment quality indicator
        self._update_alignment_quality()
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def _draw_track1_beatgrid(self, beats_in_display, downbeats_in_display):
        """Draw Track 1 beatgrid for step 1"""
        track1_line_style = '--' if abs(self.track1_offset) > 0.01 else '-'
        track1_alpha = 0.8 if abs(self.track1_offset) > 0.01 else 0.6
        
        # Draw beats (blue lines)
        for beat_measure in beats_in_display:
            if 0 <= beat_measure <= self.measures_to_show:
                self.ax_main.axvline(x=beat_measure, color='blue', alpha=track1_alpha, 
                                   linestyle=track1_line_style, linewidth=1.5)
        
        # Draw downbeats (purple lines - most important!)
        for downbeat_measure in downbeats_in_display:
            if 0 <= downbeat_measure <= self.measures_to_show:
                self.ax_main.axvline(x=downbeat_measure, color='purple', alpha=track1_alpha + 0.2, 
                                   linestyle=track1_line_style, linewidth=3.0)
    
    def _draw_track2_beatgrid(self, beats_in_display, downbeats_in_display):
        """Draw Track 2 beatgrid for step 2"""
        track2_line_style = '--' if abs(self.track2_offset) > 0.01 else '-'
        track2_alpha = 0.8 if abs(self.track2_offset) > 0.01 else 0.6
        
        # Draw beats (orange lines)
        for beat_measure in beats_in_display:
            if 0 <= beat_measure <= self.measures_to_show:
                self.ax_main.axvline(x=beat_measure, color='orange', alpha=track2_alpha, 
                                   linestyle=track2_line_style, linewidth=1.5)
        
        # Draw downbeats (green lines - most important!)
        for downbeat_measure in downbeats_in_display:
            if 0 <= downbeat_measure <= self.measures_to_show:
                self.ax_main.axvline(x=downbeat_measure, color='green', alpha=track2_alpha + 0.2, 
                                   linestyle=track2_line_style, linewidth=3.0)
    
    def _draw_dual_beatgrids(self, track1_beats, track1_downbeats, track2_beats, track2_downbeats):
        """Draw both beatgrids for step 3 (final alignment)"""
        track1_line_style = '--' if abs(self.track1_offset) > 0.01 else '-'
        track1_alpha = 0.8 if abs(self.track1_offset) > 0.01 else 0.6
        
        # Draw Track 1 beatgrid on both plots
        for beat_measure in track1_beats:
            if 0 <= beat_measure <= self.measures_to_show:
                self.ax_track1.axvline(x=beat_measure, color='blue', alpha=track1_alpha, 
                                     linestyle=track1_line_style, linewidth=1.5)
                self.ax_track2.axvline(x=beat_measure, color='blue', alpha=track1_alpha, 
                                     linestyle=track1_line_style, linewidth=1.5)
        
        for downbeat_measure in track1_downbeats:
            if 0 <= downbeat_measure <= self.measures_to_show:
                self.ax_track1.axvline(x=downbeat_measure, color='purple', alpha=track1_alpha + 0.2, 
                                     linestyle=track1_line_style, linewidth=2.5)
                self.ax_track2.axvline(x=downbeat_measure, color='purple', alpha=track1_alpha + 0.2, 
                                     linestyle=track1_line_style, linewidth=2.5)
        
        # Draw Track 2 beatgrid on both plots
        for beat_measure in track2_beats:
            if 0 <= beat_measure <= self.measures_to_show:
                self.ax_track1.axvline(x=beat_measure, color='orange', alpha=0.6, 
                                     linestyle='--', linewidth=1.5)
                self.ax_track2.axvline(x=beat_measure, color='orange', alpha=0.6, 
                                     linestyle='--', linewidth=1.5)
        
        for downbeat_measure in track2_downbeats:
            if 0 <= downbeat_measure <= self.measures_to_show:
                self.ax_track1.axvline(x=downbeat_measure, color='green', alpha=0.8, 
                                     linestyle='--', linewidth=2.5)
                self.ax_track2.axvline(x=downbeat_measure, color='green', alpha=0.8, 
                                     linestyle='--', linewidth=2.5)
    
    def _update_alignment_quality(self):
        """Calculate and display alignment quality using actual detected beats"""
        # Calculate window parameters for the displayed segment
        # Adjust window start to align first downbeats at measure 0
        track1_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
        track2_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
        
        # Get actual detected beats within the display window with offsets
        track1_beats_in_display = self._get_beats_in_window(
            self.track1_all_beats, 
            track1_window_start, 
            self.display_duration,
            offset=self.track1_offset
        )
        
        track2_beats_in_display = self._get_beats_in_window(
            self.track2_all_beats,
            track2_window_start,
            self.display_duration,
            offset=self.track2_offset
        )
        
        if len(track1_beats_in_display) == 0 or len(track2_beats_in_display) == 0:
            return
        
        # Calculate average alignment offset
        total_offset = 0
        matches = 0
        
        for t1_beat in track1_beats_in_display:
            if 0 <= t1_beat <= self.measures_to_show:
                # Find closest t2 beat
                valid_t2_beats = track2_beats_in_display[
                    (track2_beats_in_display >= 0) & (track2_beats_in_display <= self.measures_to_show)
                ]
                if len(valid_t2_beats) > 0:
                    distances = np.abs(valid_t2_beats - t1_beat)
                    min_distance = np.min(distances)
                    # Convert measure distance to time distance for ms calculation
                    avg_bpm = (self.track1.bpm + self.track2.bpm) / 2.0
                    beats_per_second = avg_bpm / 60.0
                    measures_per_second = beats_per_second / self.beats_per_measure
                    time_distance = min_distance / measures_per_second
                    total_offset += time_distance * 1000  # Convert to ms
                    matches += 1
        
        if matches > 0:
            avg_offset = total_offset / matches
            
            # Update track2 title with alignment quality
            if avg_offset < 5:
                quality = "Excellent"
                color = "green"
            elif avg_offset < 15:
                quality = "Good"  
                color = "orange"
            else:
                quality = "Needs Improvement"
                color = "red"
            
            tempo_matched = abs(self.track1.bpm - self.track2.bpm) < 0.1
            tempo_status = "Tempo-Matched" if tempo_matched else f"BPM: {self.track2.bpm:.1f}"
            
            # Show offset information if tracks have been adjusted
            offset_info = ""
            if abs(self.track1_offset) > 0.01 or abs(self.track2_offset) > 0.01:
                offset_info = f" | T1:{self.track1_offset:.2f}s T2:{self.track2_offset:.2f}s"
            
            title = f'Track 2: {self.track2.filepath.name} ({tempo_status}) - {quality} Alignment ({avg_offset:.1f}ms){offset_info}'
            
            # Only update track2 title if we're in dual track view (step 3)
            if self.current_step == 3 and self.ax_track2 is not None:
                self.ax_track2.set_title(title, fontsize=12, fontweight='bold', color=color)
    
    def _add_buttons(self):
        """Add control buttons based on current workflow step"""
        if self.current_step in [1, 2]:
            # Steps 1 & 2: Individual track adjustment buttons
            play_ax = plt.axes([0.10, 0.02, 0.12, 0.06])
            next_ax = plt.axes([0.75, 0.02, 0.12, 0.06])
            reset_ax = plt.axes([0.49, 0.02, 0.12, 0.06])
            cancel_ax = plt.axes([0.36, 0.02, 0.12, 0.06])
            
            self.play_btn = Button(play_ax, 'Play Section', color='lightgreen', hovercolor='green')
            self.next_btn = Button(next_ax, 'Next Step', color='lightblue', hovercolor='blue')
            self.reset_btn = Button(reset_ax, 'Reset', color='lightyellow', hovercolor='yellow')
            self.cancel_btn = Button(cancel_ax, 'Cancel', color='lightcoral', hovercolor='red')
            
            self.play_btn.on_clicked(self._play_section)
            self.next_btn.on_clicked(self._next_step)
            self.reset_btn.on_clicked(self._reset_alignment)
            self.cancel_btn.on_clicked(self._cancel_alignment)
        else:
            # Step 3: Final alignment buttons
            confirm_ax = plt.axes([0.75, 0.02, 0.12, 0.06])
            auto_ax = plt.axes([0.62, 0.02, 0.12, 0.06]) 
            reset_ax = plt.axes([0.49, 0.02, 0.12, 0.06])
            cancel_ax = plt.axes([0.36, 0.02, 0.12, 0.06])
            
            self.confirm_btn = Button(confirm_ax, 'Confirm', color='lightgreen', hovercolor='green')
            self.auto_btn = Button(auto_ax, 'Auto-Align', color='lightblue', hovercolor='blue')
            self.reset_btn = Button(reset_ax, 'Reset', color='lightyellow', hovercolor='yellow')
            self.cancel_btn = Button(cancel_ax, 'Cancel', color='lightcoral', hovercolor='red')
            
            self.confirm_btn.on_clicked(self._confirm_alignment)
            self.auto_btn.on_clicked(self._auto_align)
            self.reset_btn.on_clicked(self._reset_alignment)
            self.cancel_btn.on_clicked(self._cancel_alignment)
    
    def _on_press(self, event):
        """Handle mouse press for dragging or beatgrid stretching"""
        # Check which axes are valid based on current step
        valid_axes = []
        if self.current_step in [1, 2]:
            # Single track view - only ax_main is valid
            valid_axes = [self.ax_main]
        else:
            # Dual track view - both axes are valid
            valid_axes = [self.ax_track1, self.ax_track2]
        
        if event.inaxes not in valid_axes or event.button != 1:
            return
        
        # Check if user clicked near a downbeat line for stretching
        clicked_downbeat = self._find_clicked_downbeat(event.xdata, event.inaxes)
        
        if clicked_downbeat is not None:
            # Start beatgrid stretching mode
            self.stretching = True
            self.stretch_anchor_measure = clicked_downbeat
            self.drag_start_x = event.xdata
            
            # Determine which track is being stretched
            if self.current_step == 1:
                self.dragging_track = 1
                self.original_bpm = self.track1.bpm
                self.current_bpm = self.track1.bpm
            elif self.current_step == 2:
                self.dragging_track = 2
                self.original_bpm = self.track2.bpm
                self.current_bpm = self.track2.bpm
            else:
                self.dragging_track = 1 if event.inaxes == self.ax_track1 else 2
                self.original_bpm = self.track1.bpm if self.dragging_track == 1 else self.track2.bpm
                self.current_bpm = self.original_bpm
            
            print(f"Started stretching Track {self.dragging_track} beatgrid from downbeat at {clicked_downbeat:.3f} measures")
            
        else:
            # Regular dragging mode
            self.dragging = True
            self.stretching = False
            
            # Determine which track is being dragged based on current step
            if self.current_step == 1:
                self.dragging_track = 1
            elif self.current_step == 2:
                self.dragging_track = 2
            else:
                self.dragging_track = 1 if event.inaxes == self.ax_track1 else 2
            
            self.drag_start_x = event.xdata
            
            # Store the starting offset for the track being dragged
            if self.dragging_track == 1:
                self.drag_start_offset = self.track1_offset
                current_offset = self.track1_offset
            else:
                self.drag_start_offset = self.track2_offset
                current_offset = self.track2_offset
            
            print(f"Started dragging Track {self.dragging_track} at {event.xdata:.3f} measures (current offset: {current_offset:.3f}s)")
    
    def _on_motion(self, event):
        """Handle mouse motion for dragging or beatgrid stretching"""
        # Check which axes are valid based on current step
        valid_axes = []
        if self.current_step in [1, 2]:
            # Single track view - only ax_main is valid
            valid_axes = [self.ax_main]
        else:
            # Dual track view - both axes are valid
            valid_axes = [self.ax_track1, self.ax_track2]
        
        if not (self.dragging or self.stretching) or event.inaxes not in valid_axes or event.xdata is None:
            return
        
        if self.stretching:
            # Handle beatgrid stretching
            self._handle_beatgrid_stretching(event.xdata)
        else:
            # Handle regular dragging
            # Calculate offset change based on mouse movement (in measures)
            mouse_delta_measures = event.xdata - self.drag_start_x
            
            # Convert measure delta to time delta
            avg_bpm = (self.track1.bpm + self.track2.bpm) / 2.0
            beats_per_second = avg_bpm / 60.0
            measures_per_second = beats_per_second / self.beats_per_measure
            mouse_delta_time = mouse_delta_measures / measures_per_second
            
            new_offset = self.drag_start_offset + mouse_delta_time
            
            # Limit offset to reasonable range (±2 measures worth of time)
            max_measures = 2.0
            max_offset_time = max_measures / measures_per_second
            new_offset = np.clip(new_offset, -max_offset_time, max_offset_time)
            
            # Apply offset to the track being dragged
            if self.dragging_track == 1:
                if abs(new_offset - self.track1_offset) > 0.001:  # Only update if significant change
                    self.track1_offset = new_offset
                    self._draw_beatgrids()
            else:
                if abs(new_offset - self.track2_offset) > 0.001:  # Only update if significant change
                    self.track2_offset = new_offset
                    self._draw_beatgrids()
    
    def _on_release(self, event):
        """Handle mouse release to stop dragging or stretching"""
        if self.dragging:
            final_offset = self.track1_offset if self.dragging_track == 1 else self.track2_offset
            print(f"Stopped dragging Track {self.dragging_track}. Final offset: {final_offset:.3f}s")
            self.dragging = False
        elif self.stretching:
            print(f"Stopped stretching Track {self.dragging_track}. Final BPM: {self.current_bpm:.1f}")
            self.stretching = False
            self.stretch_anchor_measure = None
        
        self.dragging_track = None
    
    def _auto_align(self, event):
        """Automatically find best alignment using actual detected beats"""
        print("Auto-aligning beats (optimizing both tracks)...")
        
        # Calculate window parameters  
        # Adjust window start to align first downbeats at measure 0
        track1_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
        track2_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
        
        # Try different offset combinations to find optimal alignment
        best_track1_offset = 0
        best_track2_offset = 0
        best_score = float('inf')
        
        # Search in small increments
        search_range = min(0.5, self.display_duration * 0.25)  # Search within reasonable range
        search_step = 0.010  # 10ms increments for dual optimization
        
        print(f"  Searching {int(2*search_range/search_step)**2} offset combinations...")
        
        for track1_offset in np.arange(-search_range, search_range, search_step):
            # Get track1 beats with this test offset
            track1_beats_test = self._get_beats_in_window(
                self.track1_all_beats,
                track1_window_start,
                self.display_duration,
                offset=track1_offset
            )
            
            if len(track1_beats_test) == 0:
                continue
                
            for track2_offset in np.arange(-search_range, search_range, search_step):
                # Get track2 beats with this test offset
                track2_beats_test = self._get_beats_in_window(
                    self.track2_all_beats,
                    track2_window_start,
                    self.display_duration,
                    offset=track2_offset
                )
                
                if len(track2_beats_test) == 0:
                    continue
                
                # Calculate alignment score between the two offset tracks
                total_distance = 0
                matches = 0
                
                for t1_beat in track1_beats_test:
                    if 0 <= t1_beat <= self.measures_to_show:
                        valid_t2_beats = track2_beats_test[
                            (track2_beats_test >= 0) & (track2_beats_test <= self.measures_to_show)
                        ]
                        if len(valid_t2_beats) > 0:
                            distances = np.abs(valid_t2_beats - t1_beat)
                            total_distance += np.min(distances)
                            matches += 1
                
                if matches > 0:
                    avg_distance = total_distance / matches
                    if avg_distance < best_score:
                        best_score = avg_distance
                        best_track1_offset = track1_offset
                        best_track2_offset = track2_offset
        
        # Apply the best offsets found
        self.track1_offset = best_track1_offset
        self.track2_offset = best_track2_offset
        self._draw_beatgrids()
        
        print(f"Auto-alignment complete:")
        print(f"  Track 1 offset: {best_track1_offset:.3f}s")
        print(f"  Track 2 offset: {best_track2_offset:.3f}s") 
        print(f"  Alignment score: {best_score*1000:.1f}ms")
    
    def _reset_alignment(self, event):
        """Reset alignment to original position"""
        self.track1_offset = 0.0
        self.track2_offset = 0.0
        self._draw_beatgrids()
        print("Both track alignments reset to original positions")
    
    def _play_section(self, event):
        """Play/pause the audio section currently displayed with optimized playback"""
        try:
            import sounddevice as sd
            import time
            import threading
            
            if self.is_playing:
                # Stop playback
                self._stop_playback()
                return
            
            # Prepare audio data
            if self.current_step == 1:
                self.playback_audio = self.track1_outro_display.copy()
                self.playback_sr = self.track1.sr
                track_name = "Track 1"
            else:
                self.playback_audio = self.track2_intro_display.copy()
                self.playback_sr = self.track2.sr
                track_name = "Track 2"
            
            # Ensure audio is mono and float32 for optimal performance
            if len(self.playback_audio.shape) > 1:
                self.playback_audio = np.mean(self.playback_audio, axis=1)
            self.playback_audio = self.playback_audio.astype(np.float32)
            
            # Normalize audio to prevent clipping/crackling
            max_val = np.max(np.abs(self.playback_audio))
            if max_val > 0:
                self.playback_audio = self.playback_audio * 0.95 / max_val
            
            # Calculate start position
            start_sample = int(self.playback_position * self.playback_sr)
            if start_sample >= len(self.playback_audio):
                self.playback_position = 0.0
                start_sample = 0
            
            # Update UI state
            self.is_playing = True
            self.stop_playback_flag = False
            self.playback_start_time = time.time()
            self.play_btn.label.set_text('Pause')
            self.play_btn.color = 'yellow'
            
            print(f"Playing {track_name} section from {self.playback_position:.1f}s...")
            
            # Start non-blocking playback
            audio_to_play = self.playback_audio[start_sample:]
            sd.play(audio_to_play, samplerate=self.playback_sr)
            
            # Store a reference for stopping if needed
            self.playback_stream = True  # Simple flag to indicate playback is active
            
            # Start visual indicator thread with lower frequency updates
            self.playback_thread = threading.Thread(target=self._update_playback_indicator)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
        except ImportError:
            print("Audio playback not available - install sounddevice: pip install sounddevice")
        except Exception as e:
            print(f"Audio playback failed: {e}")
            self._reset_playback_ui()
    
    def _stop_playback(self):
        """Stop audio playback and clean up"""
        try:
            import sounddevice as sd
            import time
            
            # Save current playback position before stopping
            if self.is_playing and self.playback_start_time:
                current_time = time.time()
                elapsed_since_start = current_time - self.playback_start_time
                start_position_samples = int(self.playback_position * self.playback_sr)
                current_position_samples = start_position_samples + int(elapsed_since_start * self.playback_sr)
                self.playback_position = min(
                    current_position_samples / self.playback_sr,
                    len(self.playback_audio) / self.playback_sr
                )
            
            self.stop_playback_flag = True
            self.is_playing = False
            
            # Stop audio stream
            sd.stop()  # This stops all sounddevice playback
            self.playback_stream = None
            
            # Clean up playback line
            if self.playback_line:
                try:
                    self.playback_line.remove()
                    self.playback_line = None
                except:
                    pass
            
            # Update UI
            self._reset_playback_ui()
            print(f"Playback paused at {self.playback_position:.1f}s")
            
        except Exception as e:
            print(f"Error stopping playback: {e}")
            self._reset_playback_ui()
    
    def _reset_playback_ui(self):
        """Reset playback UI elements"""
        try:
            if hasattr(self, 'play_btn') and self.play_btn is not None:
                self.play_btn.label.set_text('Play Section')
                self.play_btn.color = 'lightgreen'
                self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error resetting UI: {e}")
    
    
    def _next_step(self, event):
        """Advance to the next step in the workflow"""
        if self.current_step == 1:
            # Save Track 1 adjustments and move to Track 2
            self.track1_adjusted_downbeat = self._get_current_downbeat_position(1)
            print(f"Track 1 downbeat position saved: {self.track1_adjusted_downbeat:.3f} measures")
            self.current_step = 2
            self._refresh_ui()
        elif self.current_step == 2:
            # Save Track 2 adjustments and move to final alignment
            self.track2_adjusted_downbeat = self._get_current_downbeat_position(2)
            print(f"Track 2 downbeat position saved: {self.track2_adjusted_downbeat:.3f} measures")
            self.current_step = 3
            self._refresh_ui()
    
    def _get_current_downbeat_position(self, track_num):
        """Get the current position of the first downbeat for the specified track"""
        if track_num == 1:
            # Find the first purple line position (Track 1 downbeat)
            track1_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
            track1_downbeats = self._get_beats_in_window(
                self.track1_all_downbeats,
                track1_window_start,
                self.display_duration,
                offset=self.track1_offset
            )
            return track1_downbeats[0] if len(track1_downbeats) > 0 else 0.0
        else:
            # Find the first green line position (Track 2 downbeat)  
            track2_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
            track2_downbeats = self._get_beats_in_window(
                self.track2_all_downbeats,
                track2_window_start,
                self.display_duration,
                offset=self.track2_offset
            )
            return track2_downbeats[0] if len(track2_downbeats) > 0 else 0.0
    
    def _refresh_ui(self):
        """Refresh the UI for the current workflow step"""
        plt.close(self.fig)
        self._setup_plot()
        self._draw_beatgrids()
        plt.tight_layout()
        plt.show(block=False)
    
    def _confirm_alignment(self, event):
        """Confirm the current alignment"""
        print(f"Confirming alignment:")
        print(f"  Track 1 offset: {self.track1_offset:.3f}s")
        print(f"  Track 2 offset: {self.track2_offset:.3f}s")
        # For now, we'll return the track2 offset for backward compatibility
        # but in the future this could return both offsets
        self._close_with_result(self.track2_offset)
    
    def _cancel_alignment(self, event):
        """Cancel alignment and use original"""
        print("Alignment cancelled, using original positions")
        self._close_with_result(0.0)
    
    def _close_with_result(self, offset: float):
        """Close the GUI and return result via callback"""
        # Stop any ongoing playback properly
        if self.is_playing:
            self._stop_playback()
        
        # Wait a moment for cleanup
        import time
        time.sleep(0.1)
        
        if self.callback_func:
            self.callback_func(offset)
        plt.close(self.fig)
    
    def _find_clicked_downbeat(self, click_x, click_axes):
        """Find if user clicked near a downbeat line for stretching"""
        if self.current_step == 1:
            # Check Track 1 downbeats
            track_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
            downbeats = self._get_beats_in_window(
                self.track1_all_downbeats,
                track_window_start,
                self.display_duration,
                offset=self.track1_offset
            )
        elif self.current_step == 2:
            # Check Track 2 downbeats
            track_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
            downbeats = self._get_beats_in_window(
                self.track2_all_downbeats,
                track_window_start,
                self.display_duration,
                offset=self.track2_offset
            )
        else:
            # Step 3: Check both tracks
            if click_axes == self.ax_track1:
                track_window_start = (self.outro_start - len(self.track1_outro)) / self.sr + self.track1_first_downbeat_offset
                downbeats = self._get_beats_in_window(
                    self.track1_all_downbeats,
                    track_window_start,
                    self.display_duration,
                    offset=self.track1_offset
                )
            else:
                track_window_start = self.track2_start_sample / self.sr + self.track2_first_downbeat_offset
                downbeats = self._get_beats_in_window(
                    self.track2_all_downbeats,
                    track_window_start,
                    self.display_duration,
                    offset=self.track2_offset
                )
        
        # Find the closest downbeat within click tolerance
        click_tolerance = 0.1  # 0.1 measures tolerance
        if len(downbeats) > 0:
            distances = np.abs(downbeats - click_x)
            min_idx = np.argmin(distances)
            if distances[min_idx] < click_tolerance:
                return downbeats[min_idx]
        
        return None
    
    def _handle_beatgrid_stretching(self, current_x):
        """Handle real-time beatgrid stretching based on mouse movement"""
        if self.stretch_anchor_measure is None:
            return
        
        # Calculate how much the user has moved the anchor downbeat
        anchor_delta = current_x - self.stretch_anchor_measure
        
        # Convert anchor delta to BPM change
        # Moving right (positive delta) = faster tempo, moving left = slower tempo
        stretch_factor = 1 + (anchor_delta * 0.1)  # 10% BPM change per measure moved
        
        # Limit stretch factor to reasonable range (0.5x to 2x)
        stretch_factor = np.clip(stretch_factor, 0.5, 2.0)
        
        # Calculate new BPM
        self.current_bpm = self.original_bpm * stretch_factor
        
        # Update the track's effective BPM for display purposes
        if self.dragging_track == 1:
            self.track1.bpm = self.current_bpm
        else:
            self.track2.bpm = self.current_bpm
        
        # Redraw beatgrids with new tempo
        self._draw_beatgrids()
        
        # Update title to show live BPM
        self._update_title_with_live_bpm()
    
    def _update_title_with_live_bpm(self):
        """Update the track title to show live BPM during stretching"""
        if not self.stretching:
            return
        
        if self.current_step == 1:
            title = f'Step 1: Adjust Track 1 Beatgrid - {self.track1.filepath.name} (BPM: {self.current_bpm:.1f}) [STRETCHING]'
            self.ax_main.set_title(title, fontsize=14, fontweight='bold', color='blue')
        elif self.current_step == 2:
            title = f'Step 2: Adjust Track 2 Beatgrid - {self.track2.filepath.name} (BPM: {self.current_bpm:.1f}) [STRETCHING]'
            self.ax_main.set_title(title, fontsize=14, fontweight='bold', color='red')
        else:
            # Step 3: Update appropriate track title
            if self.dragging_track == 1:
                title = f'Step 3: Final Alignment - Track 1: {self.track1.filepath.name} (BPM: {self.current_bpm:.1f}) [STRETCHING]'
                self.ax_track1.set_title(title, fontsize=12, fontweight='bold', color='blue')
            else:
                title = f'Track 2: {self.track2.filepath.name} (BPM: {self.current_bpm:.1f}) [STRETCHING]'
                if hasattr(self, 'ax_track2') and self.ax_track2 is not None:
                    self.ax_track2.set_title(title, fontsize=12, fontweight='bold', color='red')
        
        self.fig.canvas.draw_idle()
    
    def _update_playback_indicator(self):
        """Update the live playback position indicator with optimized performance"""
        import time
        
        last_measure = -1  # Track last drawn measure to avoid unnecessary redraws
        update_count = 0
        
        while self.is_playing and not self.stop_playback_flag:
            try:
                
                # Calculate current playback position
                current_time = time.time()
                elapsed_since_start = current_time - self.playback_start_time
                
                # Update position based on actual elapsed time
                start_position_samples = int(self.playback_position * self.playback_sr)
                current_position_samples = start_position_samples + int(elapsed_since_start * self.playback_sr)
                self.playback_position = current_position_samples / self.playback_sr
                
                # Check if playback finished
                max_duration = len(self.playback_audio) / self.playback_sr
                if self.playback_position >= max_duration or self.stop_playback_flag:
                    # Playback completed naturally
                    self.is_playing = False
                    self.playback_position = 0.0
                    self._reset_playback_ui()
                    break
                
                # Convert playback position to measures
                avg_bpm = self.current_bpm if self.stretching else (
                    self.track1.bpm if self.current_step == 1 else self.track2.bpm
                )
                beats_per_second = avg_bpm / 60.0
                measures_per_second = beats_per_second / self.beats_per_measure
                playback_measure = self.playback_position * measures_per_second
                
                # Only update if position changed significantly (reduces UI blocking)
                measure_rounded = round(playback_measure, 2)  # Round to avoid micro-updates
                
                if (0 <= playback_measure <= self.measures_to_show and 
                    abs(measure_rounded - last_measure) > 0.01):  # Only update if moved > 0.01 measures
                    
                    last_measure = measure_rounded
                    
                    # Use matplotlib's more efficient blitting if available
                    try:
                        # Remove old line efficiently
                        if self.playback_line:
                            self.playback_line.remove()
                            self.playback_line = None
                        
                        # Add new line to appropriate axes
                        axes_to_draw = [self.ax_main] if self.current_step in [1, 2] else [self.ax_track1, self.ax_track2]
                        
                        # Draw on first axis only to reduce load
                        primary_ax = axes_to_draw[0] if axes_to_draw and axes_to_draw[0] is not None else None
                        if primary_ax:
                            self.playback_line = primary_ax.axvline(
                                x=playback_measure, 
                                color='red', 
                                alpha=0.7,  # Slightly more transparent
                                linestyle='-', 
                                linewidth=1.5,  # Slightly thinner
                                zorder=50  # Lower zorder
                            )
                        
                        # Update UI efficiently - only every 3rd frame to reduce load
                        update_count += 1
                        if update_count % 3 == 0:  # Update every 3rd frame (≈3 FPS visual updates)
                            self.fig.canvas.draw_idle()
                    
                    except Exception as e:
                        print(f"Line drawing error: {e}")
                
                # Longer sleep for better performance (5 FPS instead of 10 FPS)
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Playback indicator update failed: {e}")
                break
        
        # Clean up when playback stops
        try:
            if self.playback_line:
                self.playback_line.remove()
                self.playback_line = None
                self.fig.canvas.draw_idle()
        except:
            pass
    
    def show(self, callback_func: Optional[Callable] = None) -> Optional[float]:
        """
        Show the GUI for beatgrid alignment
        
        Args:
            callback_func: Function to call with result (offset in seconds)
        
        Returns:
            Offset in seconds for track2 alignment
        """
        self.callback_func = callback_func
        
        # Show the plot
        plt.tight_layout()
        plt.show(block=True)
        
        return None  # Result is handled via callback


def align_beatgrids_interactive(track1: Track, track2: Track, track1_outro: np.ndarray, track2_intro: np.ndarray,
                               outro_start: int, track2_start_sample: int, transition_duration: float) -> float:
    """
    Interactive function to align beatgrids between two tracks
    
    Args:
        track1: First track (reference)
        track2: Second track (to be aligned)
        track1_outro: Audio segment from track1
        track2_intro: Audio segment from track2  
        outro_start: Start position of track1 outro in original track
        track2_start_sample: Start position of track2 intro in original track
        transition_duration: Duration of transition in seconds
    
    Returns:
        Time offset in seconds to apply to track2 for perfect alignment
    """
    try:
        # Check if GUI is available and try different backends
        import matplotlib
        
        # Try different GUI backends in order of preference
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTKAgg']
        
        gui_available = False
        working_backend = None
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True)
                
                # Test if the backend actually works
                import matplotlib.pyplot as plt
                
                # Try to create a simple figure to test the backend
                test_fig = plt.figure()
                plt.close(test_fig)
                
                gui_available = True
                working_backend = backend
                print(f"  Using {backend} backend for beatgrid alignment GUI")
                break
                
            except Exception as backend_error:
                # This backend doesn't work, try the next one
                continue
        
        if not gui_available:
            print("  No working GUI backend found for beatgrid alignment")
            print("  Install tkinter: sudo apt-get install python3-tk (Ubuntu/Debian)")
            print("  Or install PyQt5: pip install PyQt5")
            print("  Using automatic alignment...")
            return 0.0
        
        aligner = BeatgridAligner(track1, track2, track1_outro, track2_intro, 
                                 outro_start, track2_start_sample, transition_duration)
        
        # Use a result container to capture the selection
        result_container = {'result': 0.0}
        
        def capture_result(result):
            result_container['result'] = result
        
        # Show the aligner
        print(f"\\n🎵 Opening interactive beatgrid aligner")
        print("Drag the orange/green lines to align Track 2 beats with Track 1...")
        
        aligner.show(callback_func=capture_result)
        
        return result_container['result']
        
    except ImportError as e:
        if "tkinter" in str(e).lower() or "_tkinter" in str(e).lower():
            print("  GUI not available: tkinter is not installed")
            print("  To install tkinter:")
            print("    Ubuntu/Debian: sudo apt-get install python3-tk")
            print("    macOS: tkinter should be included with Python")
            print("    Windows: tkinter should be included with Python")
            print("  Alternative: pip install PyQt5")
        else:
            print(f"  GUI not available: {e}")
        print("  Using automatic alignment...")
        return 0.0
        
    except Exception as e:
        print(f"  Could not open beatgrid aligner: {e}")
        print("  Using automatic alignment...")
        return 0.0


if __name__ == "__main__":
    # Test the beatgrid aligner with synthetic data
    import matplotlib
    matplotlib.use('TkAgg')
    
    # Create test audio and tracks - this would normally come from real tracks
    print("This is a test module. Run the main DJ mix generator to use the beatgrid aligner.")