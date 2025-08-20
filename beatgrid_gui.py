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
    """Interactive GUI for aligning beatgrids between two tracks during transitions"""
    
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
        beats_per_measure = 4
        measures_to_show = 4
        beats_per_second = track1.bpm / 60.0
        self.display_duration = (measures_to_show * beats_per_measure) / beats_per_second
        
        # Don't exceed the actual transition duration
        self.display_duration = min(self.display_duration, transition_duration)
        self.sr = track1.sr
        
        # Calculate the number of samples for the display duration
        display_samples = int(self.display_duration * self.sr)
        
        # Truncate audio to display duration
        self.track1_outro_display = track1_outro[:display_samples] if len(track1_outro) > display_samples else track1_outro
        self.track2_intro_display = track2_intro[:display_samples] if len(track2_intro) > display_samples else track2_intro
        
        # Time axis for the display
        self.time_axis = np.linspace(0, self.display_duration, len(self.track1_outro_display))
        
        # Convert beat positions to time within the display window
        self.track1_beats_in_transition = self._get_beats_in_transition(track1, outro_start, display_samples)
        self.track2_beats_in_transition = self._get_beats_in_transition(track2, track2_start_sample, display_samples)
        
        # Track2 alignment offset (user adjustable)
        self.track2_offset = 0.0  # seconds
        self.callback_func = None
        self.dragging = False
        
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
    
    def _setup_plot(self):
        """Setup the matplotlib plot with waveforms and beatgrids"""
        # Create figure with subplots
        self.fig, (self.ax_wave, self.ax_controls) = plt.subplots(
            2, 1, figsize=(16, 10), 
            gridspec_kw={'height_ratios': [5, 1]}
        )
        
        # Plot both waveforms (using display-truncated versions)
        self.ax_wave.plot(self.time_axis, self.track1_outro_display, color='blue', alpha=0.6, linewidth=0.8, label=f'Track 1: {self.track1.filepath.name}')
        self.ax_wave.plot(self.time_axis, self.track2_intro_display, color='red', alpha=0.6, linewidth=0.8, label=f'Track 2: {self.track2.filepath.name}')
        
        self.ax_wave.set_title(f'Interactive Beatgrid Alignment - First 4 Measures ({self.display_duration:.1f}s)', 
                              fontsize=14, fontweight='bold')
        self.ax_wave.set_xlabel('Time (seconds)', fontsize=12)
        self.ax_wave.set_ylabel('Amplitude', fontsize=12)
        self.ax_wave.grid(True, alpha=0.3)
        self.ax_wave.legend(loc='upper right')
        
        # Draw initial beatgrids
        self._draw_beatgrids()
        
        # Instructions
        instruction_text = (
            f"Beatgrid Alignment - First 4 Measures of Transition:\\n"
            f"• Blue waveform: {self.track1.filepath.name} (BPM: {self.track1.bpm:.1f})\\n"
            f"• Red waveform: {self.track2.filepath.name} (BPM: {self.track2.bpm:.1f})\\n"
            f"• Blue lines: Track 1 beats (reference) | Purple lines: Track 1 downbeats\\n"
            f"• Orange lines: Track 2 beats (adjustable) | Green lines: Track 2 downbeats\\n"
            f"• Click ANYWHERE on the waveform and drag left/right to align Track 2\\n"
            f"• Perfect alignment = all orange/green lines align with blue/purple lines"
        )
        
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
        self.fig.canvas.manager.set_window_title('Beatgrid Alignment Tool')
    
    def _draw_beatgrids(self):
        """Draw the beatgrid lines for both tracks"""
        # Clear existing beatgrid lines
        for line in self.ax_wave.lines[2:]:  # Keep only the first 2 waveform lines
            line.remove()
        
        # Get downbeats for both tracks
        track1_downbeats = self._get_downbeats_in_transition(self.track1, self.outro_start, len(self.track1_outro))
        track2_downbeats = self._get_downbeats_in_transition(self.track2, self.track2_start_sample, len(self.track2_intro))
        
        # Apply offset to track2 positions
        track2_beats_offset = self.track2_beats_in_transition + self.track2_offset
        track2_downbeats_offset = track2_downbeats + self.track2_offset
        
        # Draw Track 1 beatgrid (reference - not movable)
        for beat_time in self.track1_beats_in_transition:
            if 0 <= beat_time <= self.display_duration:
                self.ax_wave.axvline(x=beat_time, color='blue', alpha=0.7, linestyle='-', linewidth=2)
        
        for downbeat_time in track1_downbeats:
            if 0 <= downbeat_time <= self.display_duration:
                self.ax_wave.axvline(x=downbeat_time, color='purple', alpha=0.8, linestyle='-', linewidth=3)
        
        # Draw Track 2 beatgrid (adjustable)
        for beat_time in track2_beats_offset:
            if 0 <= beat_time <= self.display_duration:
                self.ax_wave.axvline(x=beat_time, color='orange', alpha=0.7, linestyle='-', linewidth=2)
        
        for downbeat_time in track2_downbeats_offset:
            if 0 <= downbeat_time <= self.display_duration:
                self.ax_wave.axvline(x=downbeat_time, color='green', alpha=0.8, linestyle='-', linewidth=3)
        
        # Add alignment quality indicator
        self._update_alignment_quality()
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def _update_alignment_quality(self):
        """Calculate and display alignment quality"""
        if len(self.track1_beats_in_transition) == 0 or len(self.track2_beats_in_transition) == 0:
            return
        
        # Apply offset to track2 beats
        track2_beats_offset = self.track2_beats_in_transition + self.track2_offset
        
        # Calculate average alignment offset
        total_offset = 0
        matches = 0
        
        for t1_beat in self.track1_beats_in_transition:
            if 0 <= t1_beat <= self.display_duration:
                # Find closest t2 beat
                valid_t2_beats = track2_beats_offset[
                    (track2_beats_offset >= 0) & (track2_beats_offset <= self.display_duration)
                ]
                if len(valid_t2_beats) > 0:
                    distances = np.abs(valid_t2_beats - t1_beat)
                    min_distance = np.min(distances)
                    total_offset += min_distance * 1000  # Convert to ms
                    matches += 1
        
        if matches > 0:
            avg_offset = total_offset / matches
            
            # Update title with alignment quality
            if avg_offset < 5:
                quality = "Excellent"
                color = "green"
            elif avg_offset < 15:
                quality = "Good"  
                color = "orange"
            else:
                quality = "Needs Improvement"
                color = "red"
            
            title = f'Interactive Beatgrid Alignment - {quality} ({avg_offset:.1f}ms avg offset)'
            self.ax_wave.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    def _add_buttons(self):
        """Add control buttons to the interface"""
        # Button positions (left, bottom, width, height)
        confirm_ax = plt.axes([0.75, 0.02, 0.12, 0.06])
        auto_ax = plt.axes([0.62, 0.02, 0.12, 0.06]) 
        reset_ax = plt.axes([0.49, 0.02, 0.12, 0.06])
        cancel_ax = plt.axes([0.36, 0.02, 0.12, 0.06])
        
        # Create buttons
        self.confirm_btn = Button(confirm_ax, 'Confirm', color='lightgreen', hovercolor='green')
        self.auto_btn = Button(auto_ax, 'Auto-Align', color='lightblue', hovercolor='blue')
        self.reset_btn = Button(reset_ax, 'Reset', color='lightyellow', hovercolor='yellow')
        self.cancel_btn = Button(cancel_ax, 'Cancel', color='lightcoral', hovercolor='red')
        
        # Connect button events
        self.confirm_btn.on_clicked(self._confirm_alignment)
        self.auto_btn.on_clicked(self._auto_align)
        self.reset_btn.on_clicked(self._reset_alignment)
        self.cancel_btn.on_clicked(self._cancel_alignment)
    
    def _on_press(self, event):
        """Handle mouse press for dragging - allow clicking anywhere on waveform"""
        if event.inaxes != self.ax_wave or event.button != 1:
            return
        
        # Allow dragging from anywhere on the waveform for easier interaction
        self.dragging = True
        self.drag_start_x = event.xdata
        self.drag_start_offset = self.track2_offset
        print(f"Started dragging at {event.xdata:.3f}s (current offset: {self.track2_offset:.3f}s)")
    
    def _on_motion(self, event):
        """Handle mouse motion for dragging"""
        if not self.dragging or event.inaxes != self.ax_wave or event.xdata is None:
            return
        
        # Calculate offset change based on mouse movement
        mouse_delta = event.xdata - self.drag_start_x
        new_offset = self.drag_start_offset + mouse_delta
        
        # Limit offset to reasonable range
        max_offset = self.display_duration * 0.5
        new_offset = np.clip(new_offset, -max_offset, max_offset)
        
        if abs(new_offset - self.track2_offset) > 0.001:  # Only update if significant change
            self.track2_offset = new_offset
            self._draw_beatgrids()
    
    def _on_release(self, event):
        """Handle mouse release to stop dragging"""
        if self.dragging:
            self.dragging = False
            print(f"Stopped dragging. Final offset: {self.track2_offset:.3f}s")
    
    def _auto_align(self, event):
        """Automatically find best alignment"""
        print("Auto-aligning beats...")
        
        if len(self.track1_beats_in_transition) == 0 or len(self.track2_beats_in_transition) == 0:
            print("Not enough beats for auto-alignment")
            return
        
        # Try different offsets and find the one with minimum total distance
        best_offset = 0
        best_score = float('inf')
        
        # Search in small increments
        search_range = min(1.0, self.display_duration * 0.25)  # Search within reasonable range
        for offset in np.arange(-search_range, search_range, 0.005):  # 5ms increments
            track2_beats_test = self.track2_beats_in_transition + offset
            
            # Calculate alignment score
            total_distance = 0
            matches = 0
            
            for t1_beat in self.track1_beats_in_transition:
                if 0 <= t1_beat <= self.display_duration:
                    valid_t2_beats = track2_beats_test[
                        (track2_beats_test >= 0) & (track2_beats_test <= self.display_duration)
                    ]
                    if len(valid_t2_beats) > 0:
                        distances = np.abs(valid_t2_beats - t1_beat)
                        total_distance += np.min(distances)
                        matches += 1
            
            if matches > 0:
                avg_distance = total_distance / matches
                if avg_distance < best_score:
                    best_score = avg_distance
                    best_offset = offset
        
        self.track2_offset = best_offset
        self._draw_beatgrids()
        print(f"Auto-alignment complete. Offset: {best_offset:.3f}s, Score: {best_score*1000:.1f}ms")
    
    def _reset_alignment(self, event):
        """Reset alignment to original position"""
        self.track2_offset = 0.0
        self._draw_beatgrids()
        print("Alignment reset to original position")
    
    def _confirm_alignment(self, event):
        """Confirm the current alignment"""
        print(f"Confirming alignment with offset: {self.track2_offset:.3f}s")
        self._close_with_result(self.track2_offset)
    
    def _cancel_alignment(self, event):
        """Cancel alignment and use original"""
        print("Alignment cancelled, using original")
        self._close_with_result(0.0)
    
    def _close_with_result(self, offset: float):
        """Close the GUI and return result via callback"""
        if self.callback_func:
            self.callback_func(offset)
        plt.close(self.fig)
    
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