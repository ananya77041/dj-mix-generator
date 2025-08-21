#!/usr/bin/env python3
"""
Base GUI components and utilities
Common functionality for all GUI components
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Optional, Callable, Tuple, Dict, Any
from abc import ABC, abstractmethod
from core.config import GuiConstants


class BaseGuiComponent(ABC):
    """Abstract base class for GUI components"""
    
    def __init__(self, width: int = GuiConstants.WINDOW_WIDTH, 
                 height: int = GuiConstants.WINDOW_HEIGHT):
        self.width = width
        self.height = height
        self.fig = None
        self.result = None
        self.callback_func = None
        self.buttons = {}
        
    @abstractmethod
    def _setup_plot(self):
        """Setup the main plot - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def show(self, callback_func: Optional[Callable] = None) -> Any:
        """Show the GUI and return result"""
        pass
    
    def _add_button(self, name: str, label: str, position: Tuple[float, float, float, float],
                   callback: Callable, color: str = 'lightblue', hover_color: str = 'blue'):
        """Add a button to the GUI"""
        ax = plt.axes(position)
        button = Button(ax, label, color=color, hovercolor=hover_color)
        button.on_clicked(callback)
        self.buttons[name] = button
        return button
    
    def _create_standard_buttons(self):
        """Create standard OK/Cancel buttons"""
        self._add_button('ok', 'OK', [0.7, 0.02, 0.12, 0.06], 
                        self._on_ok, 'lightgreen', 'green')
        self._add_button('cancel', 'Cancel', [0.4, 0.02, 0.12, 0.06],
                        self._on_cancel, 'lightcoral', 'red')
    
    def _on_ok(self, event):
        """Handle OK button click"""
        if self._validate_selection():
            self._finalize_result()
            if self.callback_func:
                self.callback_func(self.result)
            plt.close(self.fig)
    
    def _on_cancel(self, event):
        """Handle Cancel button click"""
        self.result = 'cancel'
        if self.callback_func:
            self.callback_func(self.result)
        plt.close(self.fig)
    
    @abstractmethod
    def _validate_selection(self) -> bool:
        """Validate current selection - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _finalize_result(self):
        """Finalize the result - must be implemented by subclasses"""
        pass


class AudioWaveformPlotter:
    """Utility class for plotting audio waveforms consistently"""
    
    @staticmethod
    def plot_waveform(ax, audio: np.ndarray, sr: int, color: str = 'blue', 
                     alpha: float = 0.8, linewidth: float = 0.5,
                     label: Optional[str] = None):
        """Plot an audio waveform"""
        time_axis = np.linspace(0, len(audio) / sr, len(audio))
        ax.plot(time_axis, audio, color=color, alpha=alpha, linewidth=linewidth, label=label)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=GuiConstants.GRID_ALPHA)
        
        if label:
            ax.legend()
    
    @staticmethod
    def add_time_markers(ax, positions: list, labels: list = None, 
                        color: str = GuiConstants.SELECTION_COLOR, 
                        linestyle: str = '--', alpha: float = 0.7):
        """Add vertical time markers to plot"""
        for i, pos in enumerate(positions):
            label = labels[i] if labels and i < len(labels) else None
            ax.axvline(pos, color=color, linestyle=linestyle, alpha=alpha, 
                      linewidth=2, label=label)
    
    @staticmethod
    def setup_time_axis(ax, duration: float, xlabel: str = 'Time (seconds)'):
        """Setup time axis with proper formatting"""
        ax.set_xlim(0, duration)
        ax.set_xlabel(xlabel)
        
        # Add time ticks every 10 seconds for long durations
        if duration > 30:
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


class GuiBackendManager:
    """Manages GUI backend selection and availability"""
    
    BACKENDS = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTKAgg']
    
    @staticmethod
    def find_working_backend() -> Optional[str]:
        """Find the first working matplotlib backend"""
        import matplotlib
        
        for backend in GuiBackendManager.BACKENDS:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                
                # Test backend by creating a figure
                test_fig = plt.figure()
                plt.close(test_fig)
                
                return backend
            except Exception:
                continue
        
        return None
    
    @staticmethod
    def setup_gui_environment() -> bool:
        """Setup GUI environment and return success status"""
        backend = GuiBackendManager.find_working_backend()
        if backend:
            print(f"Using {backend} backend for GUI")
            return True
        else:
            print("No working GUI backend found")
            return False


class MultiStepGui(BaseGuiComponent):
    """Base class for multi-step GUI workflows"""
    
    def __init__(self, steps: list, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.current_step = 0
        self.step_results = {}
    
    def _next_step(self):
        """Move to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._update_step_display()
        else:
            self._complete_workflow()
    
    def _previous_step(self):
        """Move to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_step_display()
    
    @abstractmethod
    def _update_step_display(self):
        """Update display for current step"""
        pass
    
    @abstractmethod
    def _complete_workflow(self):
        """Complete the multi-step workflow"""
        pass
    
    def _add_navigation_buttons(self):
        """Add step navigation buttons"""
        # Previous button (only show if not on first step)
        if self.current_step > 0:
            self._add_button('prev', 'Previous', [0.1, 0.02, 0.12, 0.06], 
                           self._on_previous, 'lightyellow', 'orange')
        
        # Next/Finish button
        if self.current_step < len(self.steps) - 1:
            self._add_button('next', 'Next', [0.55, 0.02, 0.12, 0.06], 
                           self._on_next, 'lightblue', 'blue')
        else:
            self._add_button('finish', 'Finish', [0.55, 0.02, 0.12, 0.06], 
                           self._on_finish, 'lightgreen', 'green')
    
    def _on_next(self, event):
        """Handle Next button"""
        if self._validate_current_step():
            self._save_current_step_result()
            self._next_step()
    
    def _on_previous(self, event):
        """Handle Previous button"""
        self._previous_step()
    
    def _on_finish(self, event):
        """Handle Finish button"""
        if self._validate_current_step():
            self._save_current_step_result()
            self._complete_workflow()
    
    @abstractmethod
    def _validate_current_step(self) -> bool:
        """Validate current step"""
        pass
    
    @abstractmethod
    def _save_current_step_result(self):
        """Save result from current step"""
        pass