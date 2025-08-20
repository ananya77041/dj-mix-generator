#!/usr/bin/env python3
"""
Visual interface for manual downbeat selection
Displays waveform of first 5 seconds and allows user to click on first downbeat
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import librosa
from typing import Optional, Callable


class DownbeatSelector:
    """Interactive GUI for selecting downbeats and BPM in a track"""
    
    def __init__(self, audio: np.ndarray, sr: int, track_name: str, beats: np.ndarray, 
                 detected_bpm: float, allow_irregular_tempo: bool = False):
        self.audio = audio
        self.sr = sr
        self.track_name = track_name
        self.beats = beats
        self.detected_bpm = detected_bpm
        self.allow_irregular_tempo = allow_irregular_tempo
        
        # Selection state
        self.first_downbeat = None
        self.second_downbeat = None
        self.selection_mode = 'first'  # 'first' or 'second'
        self.calculated_bpm = detected_bpm
        self.callback_func = None
        
        # Only show first 5 seconds
        self.display_duration = 5.0
        self.display_samples = int(self.display_duration * sr)
        self.audio_segment = audio[:self.display_samples] if len(audio) > self.display_samples else audio
        
        # Time axis for display
        self.time_axis = np.linspace(0, len(self.audio_segment) / sr, len(self.audio_segment))
        
        # Convert beats to seconds and filter to display window
        self.beat_times = librosa.frames_to_time(beats, sr=sr)
        self.display_beats = self.beat_times[self.beat_times <= self.display_duration]
        
        # Auto-detect baseline downbeat and measures for visual guidance
        self.auto_first_downbeat, self.auto_measure_markers = self._auto_detect_measures()
        
        # Setup the plot
        self._setup_plot()
    
    def _auto_detect_measures(self):
        """Auto-detect a baseline first downbeat and measure markers for visual guidance"""
        if len(self.display_beats) == 0:
            return None, []
        
        # Simple downbeat detection: use enhanced percussive analysis on display segment
        try:
            # Use HPSS to separate percussive content
            percussive, _ = librosa.effects.hpss(self.audio_segment)
            
            # Calculate onset strength focusing on percussive elements
            onset_envelope = librosa.onset.onset_strength(
                y=percussive, 
                sr=self.sr,
                aggregate=np.median,
                fmax=250,  # Focus on kick drum frequency range
                n_mels=64
            )
            
            # Calculate strength for each beat in display window
            beat_strengths = []
            for beat_time in self.display_beats:
                beat_frame = librosa.time_to_frames(beat_time, sr=self.sr, hop_length=512)
                if beat_frame < len(onset_envelope):
                    # Window around beat
                    window_start = max(0, int(beat_frame - 2))
                    window_end = min(len(onset_envelope), int(beat_frame + 3))
                    strength = np.max(onset_envelope[window_start:window_end])
                    beat_strengths.append(strength)
                else:
                    beat_strengths.append(0)
            
            beat_strengths = np.array(beat_strengths)
            
            # Find the strongest beat in the first measure (assuming 4/4 time)
            beats_per_measure = 4
            if len(beat_strengths) >= beats_per_measure:
                first_measure_end = min(beats_per_measure, len(beat_strengths))
                best_first_beat_idx = np.argmax(beat_strengths[:first_measure_end])
                auto_first_downbeat = self.display_beats[best_first_beat_idx]
                
                # Generate measure markers from this starting point
                measure_markers = []
                beats_per_second = self.detected_bpm / 60.0
                measure_duration = beats_per_measure / beats_per_second
                
                # Add measures starting from the detected downbeat
                current_measure = auto_first_downbeat
                while current_measure <= self.display_duration:
                    measure_markers.append(current_measure)
                    current_measure += measure_duration
                
                # Also add measures before the detected downbeat if possible
                current_measure = auto_first_downbeat - measure_duration
                while current_measure >= 0:
                    measure_markers.insert(0, current_measure)
                    current_measure -= measure_duration
                
                return auto_first_downbeat, measure_markers
            else:
                # Not enough beats for a full measure, just use first beat
                return self.display_beats[0] if len(self.display_beats) > 0 else None, []
                
        except Exception as e:
            print(f"Warning: Auto-detection failed: {e}")
            # Fallback: use first beat and regular intervals
            if len(self.display_beats) > 0:
                beats_per_measure = 4
                beats_per_second = self.detected_bpm / 60.0
                measure_duration = beats_per_measure / beats_per_second
                
                auto_first_downbeat = self.display_beats[0]
                measure_markers = []
                current_measure = auto_first_downbeat
                while current_measure <= self.display_duration:
                    measure_markers.append(current_measure)
                    current_measure += measure_duration
                
                return auto_first_downbeat, measure_markers
            else:
                return None, []
    
    def _setup_plot(self):
        """Setup the matplotlib plot with waveform and controls"""
        # Create figure with subplots
        self.fig, (self.ax_wave, self.ax_controls) = plt.subplots(
            2, 1, figsize=(14, 8), 
            gridspec_kw={'height_ratios': [4, 1]}
        )
        
        # Plot waveform
        self.ax_wave.plot(self.time_axis, self.audio_segment, color='steelblue', alpha=0.8, linewidth=0.5)
        self.ax_wave.set_title(f'Select Downbeats & BPM - {self.track_name} (5 seconds)', fontsize=14, fontweight='bold')
        self.ax_wave.set_xlabel('Time (seconds)', fontsize=12)
        self.ax_wave.set_ylabel('Amplitude', fontsize=12)
        self.ax_wave.grid(True, alpha=0.3)
        
        # Add detected beats as vertical lines (softer)
        for beat_time in self.display_beats:
            self.ax_wave.axvline(x=beat_time, color='orange', alpha=0.3, linestyle=':', linewidth=0.8)
        
        # Add auto-detected measure markers (stronger)
        for measure_time in self.auto_measure_markers:
            self.ax_wave.axvline(x=measure_time, color='purple', alpha=0.6, linestyle='-', linewidth=1.5)
        
        # Add auto-detected first downbeat marker (suggestion)
        if self.auto_first_downbeat is not None:
            self.ax_wave.axvline(x=self.auto_first_downbeat, color='blue', alpha=0.7, linestyle='-', linewidth=2)
        
        # Add legend
        legend_elements = []
        if len(self.display_beats) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='orange', alpha=0.3, linestyle=':', linewidth=0.8, label='Detected beats'))
        if len(self.auto_measure_markers) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='purple', alpha=0.6, linestyle='-', linewidth=1.5, label='Auto-detected measures'))
        if self.auto_first_downbeat is not None:
            legend_elements.append(plt.Line2D([0], [0], color='blue', alpha=0.7, linestyle='-', linewidth=2, label='Suggested first downbeat'))
        
        if legend_elements:
            self.ax_wave.legend(handles=legend_elements, loc='upper right')
        
        # Instructions text
        tempo_mode = "irregular" if self.allow_irregular_tempo else "whole numbers"
        instruction_text = (
            f"Instructions (BPM will be quantized to {tempo_mode}):\n"
            f"• Detected BPM: {self.detected_bpm:.1f} - Showing first 5 seconds\n"
            "• Orange dotted lines: detected beats (softer guidance)\n"
            "• Purple lines: auto-detected measures (4-beat groups)\n" 
            "• Blue line: suggested first downbeat (click to use or choose your own)\n"
            "• Step 1: Click on the first downbeat (red line)\n"
            "• Step 2: Click on a later downbeat to set BPM (green line)"
        )
        
        self.ax_controls.text(0.02, 0.95, instruction_text, 
                            transform=self.ax_controls.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        # Add buttons
        self._add_buttons()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Track selection lines
        self.first_downbeat_line = None
        self.second_downbeat_line = None
        self.bpm_text = None
        
        # Set window title
        self.fig.canvas.manager.set_window_title(f'Downbeat Selection - {self.track_name}')
    
    def _add_buttons(self):
        """Add control buttons to the interface"""
        # Button positions (left, bottom, width, height)
        confirm_ax = plt.axes([0.7, 0.02, 0.12, 0.08])
        auto_ax = plt.axes([0.55, 0.02, 0.12, 0.08]) 
        cancel_ax = plt.axes([0.4, 0.02, 0.12, 0.08])
        
        # Create buttons
        self.confirm_btn = Button(confirm_ax, 'Confirm', color='lightgreen', hovercolor='green')
        self.auto_btn = Button(auto_ax, 'Auto-Detect', color='lightblue', hovercolor='blue')
        self.cancel_btn = Button(cancel_ax, 'Cancel', color='lightcoral', hovercolor='red')
        
        # Connect button events
        self.confirm_btn.on_clicked(self._confirm_selection)
        self.auto_btn.on_clicked(self._use_auto_detection)
        self.cancel_btn.on_clicked(self._cancel_selection)
    
    def _on_click(self, event):
        """Handle mouse click on waveform"""
        # Only process clicks on the waveform axes
        if event.inaxes != self.ax_wave:
            return
        
        # Only left mouse button
        if event.button != 1:
            return
        
        # Get clicked time position
        click_time = event.xdata
        if click_time is None or click_time < 0 or click_time > self.display_duration:
            return
        
        # No snapping - use exact click position for maximum precision
        selected_time = click_time
        
        # Handle selection based on current mode
        if self.selection_mode == 'first':
            self.first_downbeat = selected_time
            self.selection_mode = 'second'
            print(f"First downbeat selected at: {self.first_downbeat:.3f}s")
            print("Now click on a later downbeat to set BPM...")
        else:  # selection_mode == 'second'
            if selected_time <= self.first_downbeat:
                print(f"Second downbeat must be after first downbeat ({self.first_downbeat:.3f}s)")
                return
            
            self.second_downbeat = selected_time
            print(f"Second downbeat selected at: {self.second_downbeat:.3f}s")
            
            # Calculate BPM from the two selections
            self._calculate_bpm()
        
        # Update visual selection
        self._update_selection_visual()
    
    def _calculate_bpm(self):
        """Calculate BPM from two downbeat selections and quantize"""
        if self.first_downbeat is None or self.second_downbeat is None:
            return
        
        # Calculate time difference
        time_diff = self.second_downbeat - self.first_downbeat
        
        # Assume the selections are one measure apart (4 beats)
        # User could select 1 measure, 2 measures, etc. apart
        # We'll calculate for 1 measure and let user adjust if needed
        beats_per_measure = 4
        raw_bpm = (beats_per_measure * 60.0) / time_diff
        
        # Quantize BPM
        if self.allow_irregular_tempo:
            self.calculated_bpm = round(raw_bpm, 1)  # Round to 1 decimal
        else:
            self.calculated_bpm = round(raw_bpm)  # Round to nearest whole number
        
        print(f"Calculated BPM: {raw_bpm:.2f} → Quantized: {self.calculated_bpm:.1f}")
        
        # Recalculate beat and measure markers based on new tempo
        self._recalculate_markers_for_new_tempo()
    
    def _recalculate_markers_for_new_tempo(self):
        """Recalculate beat and measure markers based on user-selected tempo"""
        if self.first_downbeat is None or self.second_downbeat is None:
            return
        
        # Calculate beat interval from user's tempo selection
        beats_per_second = self.calculated_bpm / 60.0
        beat_interval = 1.0 / beats_per_second
        measure_interval = beat_interval * 4  # 4 beats per measure
        
        # Generate new beat markers based on first downbeat and calculated tempo
        new_beat_times = []
        new_measure_markers = []
        
        # Add beats and measures starting from first downbeat
        current_beat = self.first_downbeat
        beat_count = 0
        while current_beat <= self.display_duration:
            new_beat_times.append(current_beat)
            
            # Every 4th beat is a measure marker
            if beat_count % 4 == 0:
                new_measure_markers.append(current_beat)
            
            current_beat += beat_interval
            beat_count += 1
        
        # Add beats and measures before first downbeat if possible
        current_beat = self.first_downbeat - beat_interval
        beat_count = 1  # Start at 1 to maintain the 4-beat cycle
        while current_beat >= 0:
            new_beat_times.insert(0, current_beat)
            
            # Every 4th beat is a measure marker (counting backwards)
            if beat_count % 4 == 0:
                new_measure_markers.insert(0, current_beat)
            
            current_beat -= beat_interval
            beat_count += 1
        
        # Update the marker arrays
        self.display_beats = np.array(new_beat_times)
        self.auto_measure_markers = new_measure_markers
        
        # Redraw the markers
        self._redraw_markers()
    
    def _redraw_markers(self):
        """Redraw beat and measure markers on the plot"""
        # Clear existing marker lines (but keep waveform and selection lines)
        lines_to_remove = []
        for line in self.ax_wave.lines[:]:
            # Remove orange beat lines and purple measure lines, but keep waveform and selection lines
            if hasattr(line, 'get_color'):
                color = line.get_color()
                if color == 'orange' or color == 'purple' or color == 'blue':
                    lines_to_remove.append(line)
        
        for line in lines_to_remove:
            line.remove()
        
        # Redraw beat markers (softer)
        for beat_time in self.display_beats:
            self.ax_wave.axvline(x=beat_time, color='orange', alpha=0.3, linestyle=':', linewidth=0.8)
        
        # Redraw measure markers (stronger)
        for measure_time in self.auto_measure_markers:
            self.ax_wave.axvline(x=measure_time, color='purple', alpha=0.6, linestyle='-', linewidth=1.5)
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def _update_selection_visual(self):
        """Update the visual indicators for selected downbeats and BPM"""
        # Remove previous selection lines
        if self.first_downbeat_line:
            self.first_downbeat_line.remove()
        if self.second_downbeat_line:
            self.second_downbeat_line.remove()
        if self.bpm_text:
            self.bpm_text.remove()
        
        # Add first downbeat line
        if self.first_downbeat is not None:
            self.first_downbeat_line = self.ax_wave.axvline(
                x=self.first_downbeat, 
                color='red', 
                linewidth=1, 
                alpha=0.9,
                label='First downbeat'
            )
        
        # Add second downbeat line
        if self.second_downbeat is not None:
            self.second_downbeat_line = self.ax_wave.axvline(
                x=self.second_downbeat, 
                color='green', 
                linewidth=1, 
                alpha=0.9,
                label='Second downbeat'
            )
        
        # Add BPM text display
        if self.first_downbeat is not None and self.second_downbeat is not None:
            # Position BPM text between the two selections
            text_x = (self.first_downbeat + self.second_downbeat) / 2
            text_y = max(self.audio_segment) * 0.8  # 80% up from bottom
            
            tempo_type = "irregular" if self.allow_irregular_tempo else "quantized"
            self.bpm_text = self.ax_wave.text(
                text_x, text_y,
                f'BPM: {self.calculated_bpm:.1f}\n({tempo_type})',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                fontsize=12, fontweight='bold'
            )
        
        # Update legend
        if self.first_downbeat is not None or self.second_downbeat is not None:
            self.ax_wave.legend(loc='upper right')
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def _confirm_selection(self, event):
        """Confirm the selected downbeats and BPM"""
        if self.first_downbeat is None:
            print("No first downbeat selected. Please click on the waveform first.")
            return
        
        # If only first downbeat is selected, use detected BPM
        if self.second_downbeat is None:
            print(f"Using first downbeat at: {self.first_downbeat:.3f}s with detected BPM: {self.detected_bpm:.1f}")
            result = {
                'first_downbeat': self.first_downbeat,
                'bpm': self.detected_bpm
            }
        else:
            print(f"Confirmed: First downbeat at {self.first_downbeat:.3f}s, BPM: {self.calculated_bpm:.1f}")
            result = {
                'first_downbeat': self.first_downbeat,
                'bpm': self.calculated_bpm
            }
        
        self._close_with_result(result)
    
    def _use_auto_detection(self, event):
        """Use automatic downbeat detection"""
        print("Using automatic downbeat detection")
        self._close_with_result(None)  # None means use auto-detection
    
    def _cancel_selection(self, event):
        """Cancel the selection process"""
        print("Downbeat selection cancelled")
        self._close_with_result('cancel')
    
    def _close_with_result(self, result):
        """Close the GUI and return result via callback"""
        if self.callback_func:
            self.callback_func(result)
        plt.close(self.fig)
    
    def show(self, callback_func: Optional[Callable] = None) -> Optional[float]:
        """
        Show the GUI for downbeat selection
        
        Args:
            callback_func: Function to call with result (time in seconds, None for auto, or 'cancel')
        
        Returns:
            Selected time in seconds, None for auto-detection, or 'cancel' for cancelled
        """
        self.callback_func = callback_func
        
        # Show the plot
        plt.tight_layout()
        plt.show(block=True)
        
        return None  # Result is handled via callback


def select_first_downbeat(audio: np.ndarray, sr: int, track_name: str, beats: np.ndarray, 
                         detected_bpm: float, allow_irregular_tempo: bool = False) -> Optional[dict]:
    """
    Interactive function to select downbeats and BPM
    
    Args:
        audio: Audio signal
        sr: Sample rate
        track_name: Name of the track for display
        beats: Detected beat positions (in frames)
        detected_bpm: Originally detected BPM
        allow_irregular_tempo: If True, allow non-integer BPM values
    
    Returns:
        Dict with 'first_downbeat' and 'bpm', None for auto-detection, or 'cancel' for cancelled
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
                print(f"  Using {backend} backend for GUI")
                break
                
            except Exception as backend_error:
                # This backend doesn't work, try the next one
                continue
        
        if not gui_available:
            print("  No working GUI backend found. Available backends:")
            available_backends = matplotlib.rcsetup.interactive_bk
            print(f"    Matplotlib supports: {', '.join(available_backends)}")
            print("  Install tkinter: sudo apt-get install python3-tk (Ubuntu/Debian)")
            print("  Or install PyQt5: pip install PyQt5")
            print("  Falling back to automatic detection...")
            return None
        
        selector = DownbeatSelector(audio, sr, track_name, beats, detected_bpm, allow_irregular_tempo)
        
        # Use a result container to capture the selection
        result_container = {'result': None}
        
        def capture_result(result):
            result_container['result'] = result
        
        # Show the selector
        print(f"\n🎵 Opening visual downbeat selector for: {track_name}")
        print("Please click on the waveform where the first downbeat should occur...")
        
        selector.show(callback_func=capture_result)
        
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
        print("  Falling back to automatic detection...")
        return None
        
    except Exception as e:
        print(f"  Could not open visual selector: {e}")
        print("  Falling back to automatic detection...")
        return None


if __name__ == "__main__":
    # Test the downbeat selector with a simple sine wave
    import matplotlib
    matplotlib.use('TkAgg')
    
    # Create test audio (sine wave with beats)
    duration = 10.0
    sr = 44100
    t = np.linspace(0, duration, int(duration * sr))
    
    # Create a simple beat pattern
    beat_freq = 2.0  # 2 Hz = 120 BPM
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Add emphasis on beats
    beat_times = np.arange(0, duration, 1/beat_freq)
    for beat_time in beat_times:
        start_idx = int(beat_time * sr)
        end_idx = int((beat_time + 0.1) * sr)
        if end_idx < len(audio):
            audio[start_idx:end_idx] *= 2  # Make beats louder
    
    # Create fake beat positions
    beats = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
    
    # Test the selector
    result = select_first_downbeat(audio, sr, "Test Track", beats)
    print(f"Final result: {result}")