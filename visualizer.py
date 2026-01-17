from typing import Any, Protocol
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
import numpy as np

from game_of_life import Grid


class DisplayMode(Protocol):
    """Protocol for display mode implementations."""
    def update(self, previous_cells: np.ndarray, current_cells: np.ndarray) -> None: ...
    def get_display(self, current_cells: np.ndarray) -> np.ndarray: ...


class NormalMode:
    """Standard display - just shows current cell state."""

    def update(self, previous_cells: np.ndarray, current_cells: np.ndarray) -> None:
        pass

    def get_display(self, current_cells: np.ndarray) -> np.ndarray:
        return current_cells.astype(np.float32)


class DecayMode:
    """Shows trails of recently dead cells fading out."""

    def __init__(self, grid_shape: tuple[int, int], decay_frames: int = 5) -> None:
        self.decay_frames = decay_frames
        self.decay_grid = np.zeros(grid_shape, dtype=np.float32)
        self.decay_grid.fill(decay_frames + 1)

    def update(self, previous_cells: np.ndarray, current_cells: np.ndarray) -> None:
        just_died = (previous_cells == 1) & (current_cells == 0)
        self.decay_grid[just_died] = 1

        still_dead = (previous_cells == 0) & (current_cells == 0)
        self.decay_grid[still_dead] += 1

        self.decay_grid[current_cells == 1] = 0

    def get_display(self, current_cells: np.ndarray) -> np.ndarray:
        display = np.zeros_like(self.decay_grid)
        display[current_cells == 1] = 1.0
        fading = (current_cells == 0) & (self.decay_grid <= self.decay_frames)
        display[fading] = 0.5 * (1.0 - (self.decay_grid[fading] / self.decay_frames))
        return display


class AgeMode:
    """Shows cell age with brightness increasing over time."""

    def __init__(self, grid_shape: tuple[int, int], max_age_display: int = 50) -> None:
        self.max_age_display = max_age_display
        self.age_grid = np.zeros(grid_shape, dtype=np.float32)

    def update(self, previous_cells: np.ndarray, current_cells: np.ndarray) -> None:
        self.age_grid[current_cells == 1] += 1
        self.age_grid[current_cells == 0] = 0

    def get_display(self, current_cells: np.ndarray) -> np.ndarray:
        display = np.zeros_like(self.age_grid)
        alive = current_cells > 0
        display[alive] = 0.15 + 0.9 * np.minimum(
            self.age_grid[alive] / self.max_age_display, 1.0
        )
        return display


class HistoryTracker:
    """Tracks cumulative cell activity for heatmap display."""

    def __init__(self, grid_shape: tuple[int, int]) -> None:
        self.historical_grid = np.zeros(grid_shape, dtype=np.float32)

    def update(self, current_cells: np.ndarray) -> None:
        self.historical_grid[current_cells == 1] += 1

    def get_display(self) -> np.ndarray:
        return self.historical_grid

    def get_max(self) -> float:
        return float(self.historical_grid.max())

try:
    from sonifier import Sonifier
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


class Visualizer:
    """Draws the grid and animates it with matplotlib."""

    def __init__(
        self,
        grid: Grid,
        cmap: str | ListedColormap = "binary",
        figsize: tuple[float, float] = (8, 8),
        sound_enabled: bool = False,
        base_freq: float = 110.0,
    ) -> None:
        self.grid = grid
        self.cmap = cmap
        self.figsize = figsize
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.img: Any = None
        self.anim: animation.FuncAnimation | None = None
        self.population_history: list[int] = []
        self.generation_history: list[int] = []
        self.pop_line: Any = None
        self.pop_ax: plt.Axes | None = None
        self.stats_text: Any = None
        self._current_mode: str = "normal"
        self._modes: dict[str, DisplayMode] = {}
        self._history_tracker: HistoryTracker | None = None
        self.ax_heatmap: plt.Axes | None = None
        self.heatmap_img: Any = None

        self.buttons: dict[str, Button] = {}

        self._sound_enabled = sound_enabled and SOUND_AVAILABLE
        self._base_freq = base_freq
        self.sonifier: Sonifier | None = None

    def _setup_figure(self) -> None:
        """Get the matplotlib window ready."""

        self.fig = plt.figure(figsize=(self.figsize[0] * 2, self.figsize[1] * 1.2))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1])

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_heatmap = self.fig.add_subplot(gs[0, 1])
        self.pop_ax = self.fig.add_subplot(gs[1, :])

        self.fig.subplots_adjust(bottom=0.15, hspace=0.3)
        self._setup_grid_axes()
        self._setup_heatmap_axes()
        self._setup_population_axes()
        self._setup_buttons()
        self._record_population()
        self._update_title()

        # Initialize sonifier if enabled
        if self._sound_enabled:
            self.sonifier = Sonifier(base_freq=self._base_freq)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event: Any) -> None:
        """Handle figure close event."""
        if self.sonifier:
            self.sonifier.cleanup()

    def _setup_grid_axes(self) -> None:
        """Configure the main grid display."""
        self.ax.set_xticks(np.arange(-0.5, self.grid.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid.height, 1), minor=True)
        self.ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        self.ax.grid(which="minor", color="gray", linestyle="dotted", linewidth=0.5, alpha=0.3)
        self.ax.set_aspect("equal")
        self.img = self.ax.imshow(
            self.grid.cells.astype(np.float32),
            cmap=self.cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        self.stats_text = self.ax.text(
            0.02, 0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            color="white",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )
        self.stats_text.set_text(self._get_stats())

        grid_shape = self.grid.cells.shape
        self._modes = {
            "normal": NormalMode(),
            "decay": DecayMode(grid_shape),
            "age": AgeMode(grid_shape),
        }
        self._history_tracker = HistoryTracker(grid_shape)

    def _setup_population_axes(self) -> None:
        """Configure the population history plot."""
        self.pop_ax.set_xlabel("Generation")
        self.pop_ax.set_ylabel("Population")
        self.pop_ax.set_title("Population Over Time")
        (self.pop_line,) = self.pop_ax.plot([], [], color="black", linewidth=1.5)

    def _setup_heatmap_axes(self) -> None:
        """Configure the 2D life history heatmap."""
        self.ax_heatmap.set_title("Life Heatmap")
        self.ax_heatmap.tick_params(length=0, labelbottom=False, labelleft=False)
        self.heatmap_img = self.ax_heatmap.imshow(
            np.zeros_like(self.grid.cells, dtype=np.float32),
            cmap="binary",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )


    def _setup_buttons(self) -> None:
        """Add display mode toggle buttons."""
        button_width = 0.08
        button_height = 0.04
        button_y = 0.02
        button_spacing = 0.09

        left_start = 0.05
        ax_normal = self.fig.add_axes([left_start, button_y, button_width, button_height])
        ax_decay = self.fig.add_axes([left_start + button_spacing, button_y, button_width, button_height])
        ax_age = self.fig.add_axes([left_start + 2 * button_spacing, button_y, button_width, button_height])

        right_end = 0.95
        ax_YlOrRd = self.fig.add_axes([right_end - button_width, button_y, button_width, button_height])
        ax_binary = self.fig.add_axes([right_end - button_width - button_spacing, button_y, button_width, button_height])

        self.buttons["normal"] = Button(ax_normal, "Normal")
        self.buttons["decay"] = Button(ax_decay, "Trail")
        self.buttons["age"] = Button(ax_age, "Age")
        self.buttons["binary"] = Button(ax_binary, "Binary")
        self.buttons["YlOrRd"] = Button(ax_YlOrRd, "YlOrRd")

        self.buttons["normal"].on_clicked(lambda _: self._set_display_mode("normal"))
        self.buttons["decay"].on_clicked(lambda _: self._set_display_mode("decay"))
        self.buttons["age"].on_clicked(lambda _: self._set_display_mode("age"))
        self.buttons["binary"].on_clicked(lambda _: self._set_cmap("binary"))
        self.buttons["YlOrRd"].on_clicked(lambda _: self._set_cmap("YlOrRd"))

        if self._sound_enabled:
            self._ax_sound = plt.axes([right_end - button_width - 2 * button_spacing, button_y, button_width, button_height])
            self._btn_sound = Button(self._ax_sound, "Sound: On")
            self._btn_sound.on_clicked(self._toggle_sound)

    def _toggle_sound(self, event: Any) -> None:
        """Toggle sound on/off."""
        if self.sonifier:
            enabled = self.sonifier.toggle()
            self._btn_sound.label.set_text(f"Sound: {'On' if enabled else 'Off'}")
            self.fig.canvas.draw_idle()

    def _set_display_mode(self, mode: str) -> None:
        """Switch display mode"""
        self._current_mode = mode

    def _set_cmap(self, colorway: str) -> None:
        """Switch colorway mode"""
        self.img.set_cmap(colorway)
        self.heatmap_img.set_cmap(colorway)


    def _update_title(self) -> None:
        """Show which generation we're on."""
        if self.ax:
            self.ax.set_title(f"Generation {self.grid.generation}")

    def _get_stats(self) -> str:
        """Calculate statistics on the game"""
        alive = int(np.sum(self.grid.cells))
        total = self.grid.width * self.grid.height
        percent = (alive / total) * 100
        return f"Alive: {alive} ({percent:.1f}%)"
    
    def _record_population(self) -> None:
        """Records the generation and population, x/y"""
        self.generation_history.append(self.grid.generation)
        self.population_history.append(int(np.sum(self.grid.cells)))

    def _animate_frame(self, frame: int) -> tuple[Any, ...]:
        """Called each frame - step the sim and redraw."""
        previous_cells = self.grid.cells.copy()
        self.grid.step()
        self._record_population()

        for mode in self._modes.values():
            mode.update(previous_cells, self.grid.cells)
        self._history_tracker.update(self.grid.cells)

        if self.sonifier and self.sonifier.enabled:
            just_born = (previous_cells == 0) & (self.grid.cells == 1)
            just_died = (previous_cells == 1) & (self.grid.cells == 0)

            self.sonifier.update(
                population=int(np.sum(self.grid.cells)),
                max_population=self.grid.width * self.grid.height,
                births=int(np.sum(just_born)),
                deaths=int(np.sum(just_died)),
                interval_ms=self._interval,
            )

        self.img.set_array(self._modes[self._current_mode].get_display(self.grid.cells))

        self._update_title()
        self.pop_line.set_data(self.generation_history, self.population_history)
        self.pop_ax.relim()
        self.pop_ax.autoscale_view()
        self.stats_text.set_text(self._get_stats())

        max_val = self._history_tracker.get_max()
        self.heatmap_img.set_array(self._history_tracker.get_display())
        if max_val > 0:
            self.heatmap_img.set_clim(0, max_val)

        return (self.img, self.pop_line, self.stats_text, self.heatmap_img)

    def animate(
        self,
        generations: int = 200,
        interval: int = 100,
        save_path: str | None = None,
    ) -> animation.FuncAnimation:
        """Run it and watch.

        Pass save_path to dump a gif or mp4.
        Returns the animation object if you're in a notebook.
        """
        self._setup_figure()
        self._interval = interval

        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate_frame,
            frames=generations,
            interval=interval,
            blit=False,
            repeat=False,
        )

        if save_path:
            if save_path.endswith(".gif"):
                self.anim.save(save_path, writer="pillow")
            else:
                self.anim.save(save_path, writer="ffmpeg")

        return self.anim

    def show(self) -> None:
        """Pop up the window."""
        plt.show()

    def snapshot(self) -> None:
        """Just show the current state, no animation."""
        self._setup_figure()
        plt.show()


def quick_show(grid: Grid, generations: int = 200, interval: int = 100) -> None:
    """One-liner to visualize a grid."""
    viz = Visualizer(grid)
    viz.animate(generations=generations, interval=interval)
    viz.show()
