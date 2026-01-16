#!/usr/bin/env python3

import argparse
from game_of_life import Grid, ToroidalGrid, PATTERNS
from visualizer import Visualizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available patterns: {', '.join(PATTERNS.keys())}",
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=50,
        help="Grid size (default: 50)",
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        help="Grid width (overrides --size)",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        help="Grid height (overrides --size)",
    )
    parser.add_argument(
        "-p", "--pattern",
        choices=list(PATTERNS.keys()) + ["random"],
        default="random",
        help="Starting pattern (default: random)",
    )
    parser.add_argument(
        "-d", "--density",
        type=float,
        default=0.3,
        help="Random fill density 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "-g", "--generations",
        type=int,
        default=200,
        help="How many steps to run (default: 200)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=100,
        help="Ms between frames (default: 100)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Save to gif or mp4",
    )
    parser.add_argument(
        "-t", "--toroidal",
        action="store_true",
        help="Use toroidal grid (wrapping edges)",
    )

    args = parser.parse_args()

    width = args.width or args.size
    height = args.height or args.size

    grid = ToroidalGrid(width, height) if args.toroidal else Grid(width, height)

    if args.pattern == "random":
        grid.randomize(args.density)
    else:
        pattern = PATTERNS[args.pattern]
        # put it somewhere visible, not jammed in the corner
        offset_x = width // 4
        offset_y = height // 4
        grid.set_pattern(pattern, (offset_x, offset_y))

    viz = Visualizer(grid)
    viz.animate(generations=args.generations, interval=args.interval, save_path=args.output)
    viz.show()


if __name__ == "__main__":
    main()
