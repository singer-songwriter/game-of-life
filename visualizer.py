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

    def _setup_figure(self) -> None:
        """Get the matplotlib window ready."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.img = self.ax.imshow(
            self.grid.cells,
            cmap=self.cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        self._update_title()

    def _update_title(self) -> None:
        """Show which generation we're on."""
        if self.ax:
            self.ax.set_title(f"Generation {self.grid.generation}")

    def _animate_frame(self, frame: int) -> tuple[Any, ...]:
        """Called each frame - step the sim and redraw."""
        self.grid.step()
        self.img.set_array(self.grid.cells)
        self._update_title()
        return (self.img,)

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
            blit=True,
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
