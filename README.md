# Game of Life
Conway's Game of Life in Python with matplotlib visualizations.

## Installation

Requires Python 3.6+

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib pygame
```

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
- `-t, --toroidal` - Wrap edges (gliders come back around)
- `-r, --rules` - Rule variant: conway, probabilistic, graduated (default: conway)
- `-c, --certainty` - Certainty for probabilistic rules 0-1 (default: 0.9)
- `--sound` - Enable sonification (requires pygame)
- `--base-freq` - Base frequency for sonification in Hz (default: 110)

### Examples

```bash
# Classic glider on a toroidal grid
python main.py -p glider -t -g 500

# Probabilistic rules with 80% certainty
python main.py -r probabilistic -c 0.8

# Large grid with sound enabled
python main.py -s 100 -d 0.2 --sound

# Save a glider gun animation
python main.py -p glider_gun -s 80 -g 300 -o glider_gun.gif
```

## Requirements

- numpy
- matplotlib
- pygame (optional, for sound)
