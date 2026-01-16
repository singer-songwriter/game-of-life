from typing import Iterator
import numpy as np
from numpy.typing import NDArray


Position = tuple[int, int]
CellState = int  # 0 = dead, 1 = alive


class Grid:
    """The world where cells live and die."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells: NDArray[np.int8] = np.zeros((height, width), dtype=np.int8)
        self.generation = 0

    def get_neighbors(self, x: int, y: int) -> Iterator[Position]:
        """The 8 cells surrounding this one (minus anything off the edge)."""
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield (nx, ny)

    def count_neighbors(self, x: int, y: int) -> int:
        """How many living neighbors does this cell have?"""
        return sum(self.cells[ny, nx] for nx, ny in self.get_neighbors(x, y))

    def next_state(self, current: CellState, neighbor_count: int) -> CellState:
        """Conway's rules: birth on 3, survive on 2-3, die otherwise."""
        if current == 1:
            return 1 if neighbor_count in (2, 3) else 0
        else:
            return 1 if neighbor_count == 3 else 0

    def step(self) -> None:
        """One tick of the simulation."""
        new_cells = np.zeros_like(self.cells)

        for y in range(self.height):
            for x in range(self.width):
                count = self.count_neighbors(x, y)
                new_cells[y, x] = self.next_state(self.cells[y, x], count)

        self.cells = new_cells
        self.generation += 1

    def set_cell(self, x: int, y: int, state: CellState = 1) -> None:
        """Turn a cell on (or off)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y, x] = state

    def set_pattern(self, pattern: list[Position], offset: Position = (0, 0)) -> None:
        """Drop a pattern onto the grid."""
        ox, oy = offset
        for x, y in pattern:
            self.set_cell(x + ox, y + oy)

    def randomize(self, density: float = 0.3) -> None:
        """Scatter random cells. Density is the chance each cell is alive."""
        self.cells = (np.random.random((self.height, self.width)) < density).astype(np.int8)

    def clear(self) -> None:
        """Kill everything."""
        self.cells.fill(0)
        self.generation = 0

class ToroidalMixin:
    """Mixin for wrapping edge topology."""

    def get_neighbors(self, x: int, y: int) -> Iterator[Position]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                yield (nx, ny)


class ProbabilisticMixin:
    """Mixin for probabilistic rules with uniform certainty."""
    certainty: float = 0.9

    def next_state(self, current: CellState, neighbor_count: int) -> CellState:
        if current == 1:
            intended = 1 if neighbor_count in (2, 3) else 0
        else:
            intended = 1 if neighbor_count == 3 else 0

        if np.random.random() < self.certainty:
            return intended
        else:
            return 1 - intended


class GraduatedMixin:
    """Mixin for graduated probability rules."""

    def next_state(self, current: CellState, neighbor_count: int) -> CellState:
        if current == 1:
            if neighbor_count < 2:
                prob = 0.1
            elif neighbor_count in (2, 3):
                prob = 0.95
            else:
                prob = 0.2
            return 1 if np.random.random() < prob else 0
        else:
            if neighbor_count == 3:
                prob = 0.9
            elif neighbor_count == 2:
                prob = 0.1
            else:
                prob = 0.01
            return 1 if np.random.random() < prob else 0


def create_grid(
    width: int,
    height: int,
    toroidal: bool = False,
    rules: str = "conway",
    certainty: float = 0.9,
) -> Grid:
    """Factory function to create a grid with the specified topology and rules."""
    bases: list[type] = []

    if toroidal:
        bases.append(ToroidalMixin)

    if rules == "probabilistic":
        bases.append(ProbabilisticMixin)
    elif rules == "graduated":
        bases.append(GraduatedMixin)

    bases.append(Grid)

    grid_class = type("DynamicGrid", tuple(bases), {})
    grid = grid_class(width, height)

    if rules == "probabilistic":
        grid.certainty = certainty

    return grid


# Some classic patterns to play with
PATTERNS: dict[str, list[Position]] = {
    "glider": [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],  # moves forever
    "blinker": [(0, 1), (1, 1), (2, 1)],  # oscillates
    "block": [(0, 0), (1, 0), (0, 1), (1, 1)],  # just sits there
    "beacon": [(0, 0), (1, 0), (0, 1), (2, 3), (3, 2), (3, 3)],  # blinks
    "toad": [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],  # oscillates
    "r_pentomino": [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],  # chaos from 5 cells
    "glider_gun": [  # spits out gliders forever
        (24, 0), (22, 1), (24, 1), (12, 2), (13, 2), (20, 2), (21, 2), (34, 2), (35, 2),
        (11, 3), (15, 3), (20, 3), (21, 3), (34, 3), (35, 3), (0, 4), (1, 4), (10, 4),
        (16, 4), (20, 4), (21, 4), (0, 5), (1, 5), (10, 5), (14, 5), (16, 5), (17, 5),
        (22, 5), (24, 5), (10, 6), (16, 6), (24, 6), (11, 7), (15, 7), (12, 8), (13, 8),
    ],
}
