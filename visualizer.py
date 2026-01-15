"""Watch the cells do their thing."""

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

    def _setup_figure(self) -> None:
        """Get the matplotlib window ready."""
        self.fig, (self.ax, self.pop_ax) = plt.subplots(
            1, 2,
            figsize = (self.figsize[0] + 4, self.figsize[1]),
            gridspec_kw = {"width_ratios" : [3, 1]},
        )

        # self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xticks(np.arange(-0.5, self.grid.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid.height, 1), minor=True)
        self.ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        self.ax.grid(which="minor", color="gray", linestyle="dotted", linewidth=0.5, alpha=0.3)
        self.ax.set_aspect("equal")
        self.img = self.ax.imshow(
            self.grid.cells,
            cmap=self.cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )   

        self.pop_ax.set_xlabel("Generation")
        self.pop_ax.set_ylabel("Population")
        self.pop_ax.set_title("Population Over Time")
        (self.pop_line,) = self.pop_ax.plot([], [], color="green", linewidth=1.5)

        self.stats_text = self.ax.text(
            0.02, 0.98,
            "",
            transform = self.ax.transAxes,
            fontsize = 10,
            color = "white",
            verticalalignment = "top",
            bbox = dict(boxstyle = "round", facecolor = "black", alpha = 0.7)
        )
        
        self.stats_text.set_text(self._get_stats())
        self._record_population()
        self._update_title()

    def _update_title(self) -> None:
        """Show which generation we're on."""
        if self.ax:
            self.ax.set_title(f"Generation {self.grid.generation}")

    def _get_stats(self) -> str:
        alive = int(np.sum(self.grid.cells))
        total = self.grid.width * self.grid.height
        percent = (alive / total) * 100
        return f"Alive: {alive} ({percent:.1f}%)"
    
    def _record_population(self) -> None:
        self.generation_history.append(self.grid.generation)
        self.population_history.append(int(np.sum(self.grid.cells)))

    def _animate_frame(self, frame: int) -> tuple[Any, ...]:
        """Called each frame - step the sim and redraw."""
        self.grid.step()
        self._record_population()
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
