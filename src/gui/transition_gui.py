#!/usr/bin/env python3
"""
Visual interface for manual transition downbeat selection
Similar to downbeat_gui.py but specialized for transition sections
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import librosa
from typing import Optional, Callable, Tuple


class TransitionDownbeatSelector:
    """Interactive GUI for selecting downbeats in transition segments"""
    
    def __init__(self, track1_audio: np.ndarray, track2_audio: np.ndarray, sr: int, 
                 track1_name: str, track2_name: str, track1_bpm: float, track2_bpm: float,
                 transition_duration: float):
        self.track1_audio = track1_audio
        self.track2_audio = track2_audio
        self.sr = sr
        self.track1_name = track1_name
        self.track2_name = track2_name
        self.track1_bpm = track1_bpm
        self.track2_bpm = track2_bpm
        self.transition_duration = transition_duration
        
        # Selection state
        self.track1_downbeat = None
        self.track2_downbeat = None
        self.current_track = 1  # 1 or 2
        self.callback_func = None
        self.result = None
        
        # Display duration - show enough to see context around transition
        self.display_duration = min(10.0, len(track1_audio) / sr, len(track2_audio) / sr)
        self.display_samples = int(self.display_duration * sr)
        
        # Prepare audio segments for display
        self.track1_segment = track1_audio[:self.display_samples] if len(track1_audio) > self.display_samples else track1_audio
        self.track2_segment = track2_audio[:self.display_samples] if len(track2_audio) > self.display_samples else track2_audio
        
        # Time axis for display
        self.time_axis = np.linspace(0, len(self.track1_segment) / sr, len(self.track1_segment))
        
        # Setup the plot
        self._setup_plot()
    
    def _setup_plot(self):
        """Setup the matplotlib plot with waveforms and controls"""
        # Create figure with subplots for both tracks
        self.fig, (self.ax_track1, self.ax_track2, self.ax_controls) = plt.subplots(
            3, 1, figsize=(14, 10), 
            gridspec_kw={'height_ratios': [3, 3, 1]}
        )
        
        # Plot track1 waveform
        self.ax_track1.plot(self.time_axis, self.track1_segment, color='steelblue', alpha=0.8, linewidth=0.5)
        self.ax_track1.set_title(f'Track 1: {self.track1_name} (BPM: {self.track1_bpm:.1f}) - Select Transition Downbeat', 
                                fontsize=12, fontweight='bold', color='steelblue')
        self.ax_track1.set_ylabel('Amplitude', fontsize=10)
        self.ax_track1.grid(True, alpha=0.3)
        
        # Plot track2 waveform 
        track2_time_axis = np.linspace(0, len(self.track2_segment) / self.sr, len(self.track2_segment))
        self.ax_track2.plot(track2_time_axis, self.track2_segment, color='darkorange', alpha=0.8, linewidth=0.5)
        self.ax_track2.set_title(f'Track 2: {self.track2_name} (BPM: {self.track2_bpm:.1f}) - Select Transition Downbeat', 
                                fontsize=12, fontweight='bold', color='darkorange')
        self.ax_track2.set_xlabel('Time (seconds)', fontsize=10)
        self.ax_track2.set_ylabel('Amplitude', fontsize=10)
        self.ax_track2.grid(True, alpha=0.3)
        
        # Add visual indicators for transition boundaries
        transition_line_style = dict(color='red', linestyle='--', alpha=0.7, linewidth=2)
        if self.transition_duration <= self.display_duration:
            self.ax_track1.axvline(self.transition_duration, label='Transition End', **transition_line_style)
            self.ax_track2.axvline(self.transition_duration, label='Transition End', **transition_line_style)
        
        # Initialize selection markers
        self.track1_marker = None
        self.track2_marker = None
        
        # Connect mouse click events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Add buttons and instructions
        self._add_buttons()
        self._update_instructions()
        
        # Set window title
        self.fig.canvas.manager.set_window_title(f'Transition Downbeat Selection')
    
    def _add_buttons(self):
        """Add control buttons to the interface"""
        # Button positions (left, bottom, width, height)
        confirm_ax = plt.axes([0.7, 0.02, 0.12, 0.06])
        skip_ax = plt.axes([0.55, 0.02, 0.12, 0.06]) 
        cancel_ax = plt.axes([0.4, 0.02, 0.12, 0.06])
        
        # Create buttons
        self.confirm_btn = Button(confirm_ax, 'Confirm', color='lightgreen', hovercolor='green')
        self.skip_btn = Button(skip_ax, 'Skip Track', color='lightyellow', hovercolor='orange')
        self.cancel_btn = Button(cancel_ax, 'Cancel', color='lightcoral', hovercolor='red')
        
        # Connect button events
        self.confirm_btn.on_clicked(self._confirm_selection)
        self.skip_btn.on_clicked(self._skip_track)
        self.cancel_btn.on_clicked(self._cancel_selection)
    
    def _update_instructions(self):
        """Update instruction text based on current state"""
        if self.current_track == 1:
            instruction = f"STEP 1/2: Click on Track 1 waveform to select downbeat for transition alignment\n(Track 1 will be remapped using BPM {self.track1_bpm:.1f})"
            self.ax_track1.set_facecolor('lightcyan')
            self.ax_track2.set_facecolor('white')
        else:
            instruction = f"STEP 2/2: Click on Track 2 waveform to select downbeat for transition alignment\n(Track 2 will be remapped using BPM {self.track2_bpm:.1f})"
            self.ax_track1.set_facecolor('white')
            self.ax_track2.set_facecolor('lightcyan')
        
        # Update control panel with instructions
        self.ax_controls.clear()
        self.ax_controls.text(0.5, 0.7, instruction, ha='center', va='center', 
                             fontsize=11, fontweight='bold', transform=self.ax_controls.transAxes)
        self.ax_controls.text(0.5, 0.3, "Click 'Skip Track' to use automatic detection, or 'Cancel' to abort", 
                             ha='center', va='center', fontsize=10, alpha=0.7, transform=self.ax_controls.transAxes)
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
    
    def _on_click(self, event):
        """Handle mouse click on waveform"""
        # Determine which track was clicked
        if event.inaxes == self.ax_track1 and self.current_track == 1:
            click_time = event.xdata
            if click_time is not None and 0 <= click_time <= self.display_duration:
                self.track1_downbeat = click_time
                print(f"Track 1 downbeat selected at: {self.track1_downbeat:.3f}s")
                self._update_track1_visual()
                self._move_to_next_track()
                
        elif event.inaxes == self.ax_track2 and self.current_track == 2:
            click_time = event.xdata
            if click_time is not None and 0 <= click_time <= self.display_duration:
                self.track2_downbeat = click_time
                print(f"Track 2 downbeat selected at: {self.track2_downbeat:.3f}s")
                self._update_track2_visual()
    
    def _update_track1_visual(self):
        """Update visual selection marker for track 1"""
        if self.track1_marker:
            self.track1_marker.remove()
        
        if self.track1_downbeat is not None:
            self.track1_marker = self.ax_track1.axvline(
                self.track1_downbeat, color='red', linewidth=3, alpha=0.8,
                label=f'Selected Downbeat: {self.track1_downbeat:.3f}s'
            )
            self.ax_track1.legend(loc='upper right')
        
        plt.draw()
    
    def _update_track2_visual(self):
        """Update visual selection marker for track 2"""
        if self.track2_marker:
            self.track2_marker.remove()
        
        if self.track2_downbeat is not None:
            self.track2_marker = self.ax_track2.axvline(
                self.track2_downbeat, color='red', linewidth=3, alpha=0.8,
                label=f'Selected Downbeat: {self.track2_downbeat:.3f}s'
            )
            self.ax_track2.legend(loc='upper right')
        
        plt.draw()
    
    def _move_to_next_track(self):
        """Move to the next track selection"""
        if self.current_track == 1:
            self.current_track = 2
            self._update_instructions()
            plt.draw()
    
    def _confirm_selection(self, event):
        """Confirm the current selections"""
        if self.current_track == 1:
            print("Please select downbeat for Track 1 first")
            return
        
        if self.track1_downbeat is None and self.track2_downbeat is None:
            print("No downbeats selected. Please select at least one downbeat or use 'Skip Track'")
            return
        
        self.result = {
            'track1_downbeat': self.track1_downbeat,
            'track2_downbeat': self.track2_downbeat
        }
        
        print(f"Transition downbeat selection confirmed:")
        print(f"  Track 1: {self.track1_downbeat:.3f}s" if self.track1_downbeat else "  Track 1: Auto-detect")
        print(f"  Track 2: {self.track2_downbeat:.3f}s" if self.track2_downbeat else "  Track 2: Auto-detect")
        
        if self.callback_func:
            self.callback_func(self.result)
        
        plt.close(self.fig)
    
    def _skip_track(self, event):
        """Skip the current track (use automatic detection)"""
        if self.current_track == 1:
            print("Skipping Track 1 downbeat selection (will use automatic detection)")
            self.track1_downbeat = None
            self._move_to_next_track()
        else:
            print("Skipping Track 2 downbeat selection (will use automatic detection)")
            self.track2_downbeat = None
    
    def _cancel_selection(self, event):
        """Cancel the selection process"""
        print("Transition downbeat selection cancelled")
        self.result = 'cancel'
        
        if self.callback_func:
            self.callback_func(self.result)
        
        plt.close(self.fig)
    
    def show(self, callback_func: Optional[Callable] = None) -> Optional[dict]:
        """
        Show the GUI for transition downbeat selection
        
        Args:
            callback_func: Function to call with result
        
        Returns:
            Dict with selected downbeats or 'cancel'
        """
        self.callback_func = callback_func
        
        # Show the plot
        plt.tight_layout()
        plt.show(block=True)
        
        return self.result


def select_transition_downbeats(track1_audio: np.ndarray, track2_audio: np.ndarray, sr: int,
                              track1_name: str, track2_name: str, track1_bpm: float, track2_bpm: float,
                              transition_duration: float) -> Optional[dict]:
    """
    Interactive function to select downbeats for transition alignment
    
    Args:
        track1_audio: Audio segment from track 1 (outro)
        track2_audio: Audio segment from track 2 (intro)
        sr: Sample rate
        track1_name: Name of track 1
        track2_name: Name of track 2
        track1_bpm: BPM of track 1
        track2_bpm: BPM of track 2
        transition_duration: Duration of transition in seconds
    
    Returns:
        Dict with selected downbeats or 'cancel'
    """
    try:
        # Check if GUI is available
        import matplotlib
        
        # Try different GUI backends
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTKAgg']
        
        gui_available = False
        working_backend = None
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                
                # Test backend
                test_fig = plt.figure()
                plt.close(test_fig)
                
                gui_available = True
                working_backend = backend
                print(f"  Using {backend} backend for transition downbeat GUI")
                break
                
            except Exception:
                continue
        
        if not gui_available:
            print("  No working GUI backend found for transition downbeat selection")
            print("  Falling back to automatic detection")
            return None
        
        # Create and show the selector
        result_container = [None]
        
        def handle_result(result):
            result_container[0] = result
        
        selector = TransitionDownbeatSelector(
            track1_audio, track2_audio, sr, track1_name, track2_name,
            track1_bpm, track2_bpm, transition_duration
        )
        
        selector.show(callback_func=handle_result)
        
        return result_container[0]
        
    except Exception as e:
        print(f"  Error in transition downbeat selection: {e}")
        print("  Falling back to automatic detection")
        return None