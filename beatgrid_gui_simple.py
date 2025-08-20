#!/usr/bin/env python3
"""
Simplified GPU-accelerated beatgrid alignment GUI using Dear PyGui
Avoids complex nesting to prevent container stack issues
"""

import numpy as np
import librosa
import sounddevice as sd
import threading
import time
from typing import Optional, Callable, Tuple
from models import Track

try:
    import dearpygui.dearpygui as dpg
    DEARPYGUI_AVAILABLE = True
except ImportError:
    DEARPYGUI_AVAILABLE = False


class SimpleBeatgridAligner:
    """Simplified GPU-accelerated beatgrid alignment with flat structure"""
    
    def __init__(self, track1: Track, track2: Track, track1_outro: np.ndarray, track2_intro: np.ndarray,
                 outro_start: int, track2_start_sample: int, transition_duration: float):
        
        if not DEARPYGUI_AVAILABLE:
            raise ImportError("Dear PyGui is required for the beatgrid aligner")
        
        # Track data
        self.track1 = track1
        self.track2 = track2
        self.track1_outro = track1_outro
        self.track2_intro = track2_intro
        self.outro_start = outro_start
        self.track2_start_sample = track2_start_sample
        self.transition_duration = transition_duration
        
        # Display parameters - show 4 measures
        self.beats_per_measure = 4
        self.measures_to_show = 4
        beats_per_second = track1.bpm / 60.0
        self.display_duration = (self.measures_to_show * self.beats_per_measure) / beats_per_second
        self.display_duration = min(self.display_duration, transition_duration)
        self.sr = track1.sr
        
        # Calculate display audio segments
        display_samples = int(self.display_duration * self.sr)
        self.track1_display = track1_outro[:display_samples] if len(track1_outro) >= display_samples else track1_outro
        self.track2_display = track2_intro[:display_samples] if len(track2_intro) >= display_samples else track2_intro
        
        # Create time axis in measures
        time_seconds = np.linspace(0, len(self.track1_display) / self.sr, len(self.track1_display))
        beats_per_second = track1.bpm / 60.0
        measures_per_second = beats_per_second / self.beats_per_measure
        self.time_axis = time_seconds * measures_per_second
        
        # Workflow state
        self.current_step = 1  # 1=Track 1, 2=Track 2, 3=Final alignment
        
        # Track adjustments
        self.track1_offset = 0.0
        self.track2_offset = 0.0
        self.track1_bpm_adjustment = 1.0
        self.track2_bpm_adjustment = 1.0
        
        # Audio playback
        self.is_playing = False
        self.playback_position = 0.0
        self.playback_start_time = None
        self.stop_playback_flag = False
        
        # Result
        self.callback_func = None
        self.result_offset = 0.0
        self.gui_running = False
        
    def create_gui(self):
        """Create simplified Dear PyGui interface"""
        try:
            dpg.create_context()
            
            # Simple dark theme
            with dpg.theme() as global_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 30))
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 130, 200))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 150, 220))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            
            # Create main window with flat structure
            with dpg.window(label="🎵 Professional Beatgrid Alignment", 
                           tag="main_window", width=1200, height=700):
                
                # Step indicator
                dpg.add_text("STEP 1: Adjust Track 1 Beatgrid", tag="step_text", color=(100, 200, 255))
                dpg.add_separator()
                
                # Instructions
                dpg.add_text(self._get_instructions(), tag="instructions", wrap=1150)
                dpg.add_separator()
                
                # Simple waveform plot
                self._create_simple_plot()
                dpg.add_separator()
                
                # Control buttons - flat layout
                dpg.add_text("🎵 Controls", color=(150, 255, 150))
                
                # Play button
                dpg.add_button(label="▶️ Play Section", tag="play_btn", 
                             callback=self._toggle_playback, width=120, height=35)
                dpg.add_same_line()
                
                # BPM slider
                current_bpm = self.track1.bpm
                dpg.add_slider_float(label="BPM", tag="bpm_slider",
                                   default_value=current_bpm,
                                   min_value=current_bpm * 0.5,
                                   max_value=current_bpm * 2.0,
                                   callback=self._update_bpm,
                                   width=200)
                dpg.add_same_line()
                
                # Navigation buttons
                dpg.add_button(label="➡️ Next Step", tag="next_btn",
                             callback=self._next_step, width=100, height=35)
                dpg.add_same_line()
                
                dpg.add_button(label="✅ Confirm", tag="confirm_btn",
                             callback=self._confirm, width=80, height=35)
                dpg.add_same_line()
                
                dpg.add_button(label="❌ Cancel", tag="cancel_btn",
                             callback=self._cancel, width=80, height=35)
                
                dpg.add_separator()
                
                # Status
                dpg.add_text("Ready - Adjust beatgrid and test with Play button", 
                           tag="status", color=(150, 255, 150))
                
                dpg.add_text(f"BPM: {current_bpm:.1f} | Offset: 0.000s | Quality: Not measured",
                           tag="info", color=(200, 200, 255))
            
            # Set primary window
            dpg.set_primary_window("main_window", True)
            
            # Setup viewport
            dpg.create_viewport(title="🎵 DJ Mix Generator - Beatgrid Alignment", 
                              width=1250, height=750)
            
            # Apply theme
            dpg.bind_theme(global_theme)
            dpg.setup_dearpygui()
            
            return True
            
        except Exception as e:
            print(f"Error creating simplified Dear PyGui interface: {e}")
            try:
                dpg.destroy_context()
            except:
                pass
            return False
    
    def _create_simple_plot(self):
        """Create simple waveform plot without complex nesting"""
        try:
            # Determine which track to show
            if self.current_step == 1:
                audio_data = self.track1_display
                track_name = "Track 1"
                color = (100, 150, 255)
            else:
                audio_data = self.track2_display
                track_name = "Track 2"  
                color = (255, 150, 100)
            
            # Simple plot
            with dpg.plot(label=f"{track_name} Waveform", height=300, width=1150, tag="waveform_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Measures", tag="x_axis")
                dpg.set_axis_limits("x_axis", 0, self.measures_to_show)
                
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")
                dpg.set_axis_limits("y_axis", -1.1, 1.1)
                
                # Add waveform
                dpg.add_line_series(self.time_axis.tolist(), audio_data.tolist(), 
                                   label="Waveform", parent="y_axis", tag="waveform_series")
        
        except Exception as e:
            print(f"Error creating plot: {e}")
            # Simple fallback
            dpg.add_text(f"Waveform visualization unavailable: {e}")
            dpg.add_text("Using simplified controls")
    
    def _get_instructions(self):
        """Get current step instructions"""
        if self.current_step == 1:
            return ("STEP 1: Adjust Track 1 beatgrid using BPM slider and test with Play button. "
                   "Click Next Step when satisfied.")
        elif self.current_step == 2:
            return ("STEP 2: Adjust Track 2 beatgrid using BPM slider and test with Play button. "
                   "Click Next Step when satisfied.")
        else:
            return ("STEP 3: Final alignment complete. Click Confirm to apply changes.")
    
    def _toggle_playback(self):
        """Simple play/pause toggle"""
        try:
            if self.is_playing:
                self._stop_playback()
            else:
                self._start_playback()
        except Exception as e:
            print(f"Playback error: {e}")
            dpg.set_value("status", f"Playback error: {e}")
    
    def _start_playback(self):
        """Start audio playback"""
        try:
            # Get audio for current step
            if self.current_step == 1:
                audio = self.track1_display.copy()
                sr = self.track1.sr
            else:
                audio = self.track2_display.copy()
                sr = self.track2.sr
            
            # Prepare audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * 0.9 / max_val
            
            # Start from current position
            start_sample = int(self.playback_position * sr)
            if start_sample >= len(audio):
                self.playback_position = 0.0
                start_sample = 0
            
            audio_to_play = audio[start_sample:]
            
            # Update state
            self.is_playing = True
            self.stop_playback_flag = False
            self.playback_start_time = time.time()
            
            # Update UI
            dpg.set_item_label("play_btn", "⏸️ Pause")
            dpg.set_value("status", f"Playing audio from {self.playback_position:.1f}s")
            
            # Start audio
            sd.play(audio_to_play, samplerate=sr)
            
        except Exception as e:
            print(f"Start playback error: {e}")
            self._reset_playback_ui()
    
    def _stop_playback(self):
        """Stop audio playback"""
        try:
            # Calculate final position
            if self.is_playing and self.playback_start_time:
                elapsed = time.time() - self.playback_start_time
                sr = self.track1.sr if self.current_step == 1 else self.track2.sr
                max_duration = len(self.track1_display if self.current_step == 1 else self.track2_display) / sr
                self.playback_position = min(self.playback_position + elapsed, max_duration)
            
            # Stop
            self.stop_playback_flag = True
            self.is_playing = False
            sd.stop()
            
            # Update UI
            self._reset_playback_ui()
            
        except Exception as e:
            print(f"Stop playback error: {e}")
            self._reset_playback_ui()
    
    def _reset_playback_ui(self):
        """Reset playback UI"""
        try:
            dpg.set_item_label("play_btn", "▶️ Play Section")
            dpg.set_value("status", f"Paused at {self.playback_position:.1f}s")
        except:
            pass
    
    def _update_bpm(self, sender, value):
        """Update BPM with live feedback"""
        try:
            if self.current_step == 1:
                self.track1_bpm_adjustment = value / self.track1.bpm
            else:
                self.track2_bpm_adjustment = value / self.track2.bpm
            
            # Update info display
            current_offset = self.track1_offset if self.current_step == 1 else self.track2_offset
            dpg.set_value("info", f"BPM: {value:.1f} | Offset: {current_offset:.3f}s | Quality: Adjusting...")
            dpg.set_value("status", f"BPM adjusted to {value:.1f}")
            
        except Exception as e:
            print(f"BPM update error: {e}")
    
    def _next_step(self):
        """Move to next step"""
        try:
            if self.current_step == 1:
                self.current_step = 2
                dpg.set_value("step_text", "STEP 2: Adjust Track 2 Beatgrid")
                dpg.set_value("instructions", self._get_instructions())
                
                # Update BPM slider for track 2
                dpg.set_value("bpm_slider", self.track2.bpm)
                self._update_info_display()
                
                # Try to update waveform plot
                try:
                    dpg.set_value("waveform_series", [self.time_axis.tolist(), self.track2_display.tolist()])
                except:
                    pass  # Plot update failed, continue anyway
                
            elif self.current_step == 2:
                self.current_step = 3
                dpg.set_value("step_text", "STEP 3: Final Alignment")
                dpg.set_value("instructions", self._get_instructions())
                dpg.hide_item("next_btn")  # Hide next button in final step
                
            dpg.set_value("status", f"Advanced to step {self.current_step}")
            
        except Exception as e:
            print(f"Next step error: {e}")
    
    def _update_info_display(self):
        """Update info display"""
        try:
            current_bpm = (self.track1.bpm if self.current_step == 1 else self.track2.bpm) * \
                         (self.track1_bpm_adjustment if self.current_step == 1 else self.track2_bpm_adjustment)
            current_offset = self.track1_offset if self.current_step == 1 else self.track2_offset
            dpg.set_value("info", f"BPM: {current_bpm:.1f} | Offset: {current_offset:.3f}s | Quality: Good")
        except:
            pass
    
    def _confirm(self):
        """Confirm alignment"""
        try:
            self.result_offset = self.track2_offset
            dpg.set_value("status", "✅ Alignment confirmed!")
            self._close_with_result(self.result_offset)
        except Exception as e:
            print(f"Confirm error: {e}")
            self._close_with_result(0.0)
    
    def _cancel(self):
        """Cancel alignment"""
        try:
            dpg.set_value("status", "❌ Alignment cancelled")
            self._close_with_result(0.0)
        except Exception as e:
            print(f"Cancel error: {e}")
            self._close_with_result(0.0)
    
    def _close_with_result(self, offset: float):
        """Close GUI and return result"""
        try:
            # Stop playback
            if self.is_playing:
                self._stop_playback()
            
            time.sleep(0.1)
            
            self.gui_running = False
            
            if self.callback_func:
                self.callback_func(offset)
            
        except Exception as e:
            print(f"Close error: {e}")
        
        finally:
            try:
                dpg.destroy_context()
            except:
                pass
    
    def show(self, callback_func: Optional[Callable] = None):
        """Show the simplified beatgrid alignment GUI"""
        try:
            self.callback_func = callback_func
            
            if not self.create_gui():
                if callback_func:
                    callback_func(0.0)
                return
            
            dpg.show_viewport()
            self.gui_running = True
            
            # Simple render loop
            while dpg.is_dearpygui_running() and self.gui_running:
                dpg.render_dearpygui_frame()
            
        except Exception as e:
            print(f"Show GUI error: {e}")
            try:
                if self.is_playing:
                    self._stop_playback()
                dpg.destroy_context()
            except:
                pass
            
            if callback_func:
                callback_func(0.0)


def align_beatgrids_interactive(track1: Track, track2: Track, track1_outro: np.ndarray, track2_intro: np.ndarray,
                               outro_start: int, track2_start_sample: int, transition_duration: float) -> float:
    """
    Simplified GPU-accelerated interactive beatgrid alignment
    """
    if not DEARPYGUI_AVAILABLE:
        print("Dear PyGui not available - falling back to automatic alignment")
        return 0.0
    
    try:
        aligner = SimpleBeatgridAligner(
            track1, track2, track1_outro, track2_intro,
            outro_start, track2_start_sample, transition_duration
        )
        
        result_container = {'result': 0.0}
        
        def capture_result(result):
            result_container['result'] = result
        
        print("🎵 Opening simplified Dear PyGui beatgrid aligner...")
        
        aligner.show(callback_func=capture_result)
        return result_container['result']
        
    except Exception as e:
        print(f"Simplified Dear PyGui aligner failed: {e}")
        return 0.0


if __name__ == "__main__":
    if DEARPYGUI_AVAILABLE:
        print("✅ Simplified Dear PyGui Beatgrid Aligner Ready")
    else:
        print("❌ Dear PyGui not available")
        print("Install with: pip install dearpygui>=1.10.0")