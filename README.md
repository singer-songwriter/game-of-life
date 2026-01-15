# Game of Life
Conway's Game of Life in Python with matplotlib visualizations.

## Usage

```bash
python main.py [options]
```

### Options

- `-s, --size` - Grid size (default: 50)
- `-W, --width` - Grid width (overrides --size)
- `-H, --height` - Grid height (overrides --size)
- `-p, --pattern` - Starting pattern: glider, blinker, block, beacon, toad, r_pentomino, glider_gun, or random (default: random)
- `-d, --density` - Random fill density 0-1 (default: 0.3)
- `-g, --generations` - Steps to run (default: 200)
- `-i, --interval` - Ms between frames (default: 100)
- `-o, --output` - Save to gif or mp4

## Requirements

- numpy
- matplotlib
