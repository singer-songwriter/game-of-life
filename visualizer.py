from typing import Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np

from game_of_life import Grid


class Visualizer:
    """Draws the grid and animates it with matplotlib."""

    def __init__(
        self,
        grid: Grid,
        cmap: str | ListedColormap = "binary",
        figsize: tuple[float, float] = (8, 8),
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
        # self.decay_frames: int = 5
        # self.decay_grid: np.ndarray | None = None

    def _setup_figure(self) -> None:
        """Get the matplotlib window ready."""
        self.fig, (self.ax, self.pop_ax) = plt.subplots(
            1, 2,
            figsize=(self.figsize[0] + 4, self.figsize[1]),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        self._setup_grid_axes()
        self._setup_population_axes()
        self._record_population()
        self._update_title()

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
        # self.decay_grid = np.zeros_like(self.grid.cells, dtype=np.float32)
        # self.decay_grid[self.grid.cells == 0] = self.decay_frames + 1


    def _setup_population_axes(self) -> None:
        """Configure the population history plot."""
        self.pop_ax.set_xlabel("Generation")
        self.pop_ax.set_ylabel("Population")
        self.pop_ax.set_title("Population Over Time")
        (self.pop_line,) = self.pop_ax.plot([], [], color="green", linewidth=1.5)

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

    # def _update_decay(self, previous_cells: np.ndarray) -> np.ndarray:
    #     just_died = (previous_cells == 1) & (self.grid.cells == 0)
    #     self.decay_grid[just_died] = 1

    #     still_dead = (previous_cells == 0) & (self.grid.cells == 0)
    #     self.decay_grid[still_dead] += 1

    #     self.decay_grid[self.grid.cells == 1] = 0

    #     display = np.zeros_like(self.decay_grid)
    #     display[self.grid.cells == 1] = 1.0
    #     fading = (self.grid.cells == 0) & (self.decay_grid <= self.decay_frames)
    #     display[fading] = 0.3 * (1.0 - (self.decay_grid[fading] / self.decay_frames))
    #     return display

    def _animate_frame(self, frame: int) -> tuple[Any, ...]:
        """Called each frame - step the sim and redraw."""
        previous_cells = self.grid.cells.copy()
        self.grid.step()
        self._record_population()
        # display = self._update_decay(previous_cells)
        # self.img.set_array(display)
        self.img.set_array(self.grid.cells)
        self._update_title()
        self.pop_line.set_data(self.generation_history, self.population_history)
        self.pop_ax.relim()
        self.pop_ax.autoscale_view()
        self.stats_text.set_text(self._get_stats())
        return (self.img, self.pop_line, self.stats_text)

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
