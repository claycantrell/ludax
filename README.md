# Ludax

A JAX-accelerated board game engine with a domain-specific language for defining games. Compiles `.ldx` game descriptions into hardware-accelerated environments that run at tens of millions of steps per second on modern GPUs.

This is a fork of [gdrtodd/ludax](https://github.com/gdrtodd/ludax) with significant engine extensions, a Ludii-to-Ludax transpiler, and AI-generated game support.

![Throughput of Ludax environments compared to PGX and Ludii implementations](/assets/throughput_comparison.png)

## What's New in This Fork

### Ludii Transpiler
Transpile games written in [Ludii](https://ludii.games/) syntax (which LLMs know from training data) into JAX-accelerated Ludax environments:

```python
from ludax.ludii_transpiler import transpile
from ludax import LudaxEnvironment

lud = '(game "MyGame" (players 2) (equipment {(board (square 8)) (piece "Marker" Each)}) ...)'
ldx = transpile(lud)
env = LudaxEnvironment(game_str=ldx)
```

- 100% Ludii grammar parsing (1,212/1,212 games)
- 87% end-to-end Ludii-to-JAX compilation on a random 100-game sample
- Handles: forEach Piece, multi-phase games, coordinate placement, hop-over-friendly captures

### Extended Engine
- **Leap patterns** — Knight L-shapes, camel jumps, zebra leaps, custom offsets
- **Hop-over-friendly** — Hop over own pieces to capture at destination (Nei-Pat-Kono)
- **Multi-piece movement** — Different piece types in the same `(or ...)` block
- **Multi-phase games** — Placement phase then movement phase with automatic action space padding
- **Stacking** (in progress) — `(stack N)` for games with multiple pieces per cell
- **3+ piece types** — Chess-like games with 6+ piece types, proper rendering
- **Named regions** — Custom board regions for win conditions and placement constraints
- **Auto-generated rules** — Human-readable rules panel in the GUI

### AI Game Generation
40+ AI-generated games bundled in `src/ludax/games/`, created by Claude writing Ludii syntax and transpiling to Ludax. Games include themes, backstories, and balanced mechanics evaluated by fitness functions measuring balance, completion rate, and decision significance.

## Bundled Games

### Classic Games
| Game | Board | Mechanic | File |
|------|-------|----------|------|
| Tic-Tac-Toe | 3x3 | Line of 3 | [tic_tac_toe.ldx](src/ludax/games/tic_tac_toe.ldx) |
| Connect-Four | 7x6 | Drop to connect 4 | [connect_four.ldx](src/ludax/games/connect_four.ldx) |
| Gomoku | 15x15 | Line of 5 | [gomoku.ldx](src/ludax/games/gomoku.ldx) |
| Hex | 11x11 hex | Connect opposite sides | [hex.ldx](src/ludax/games/hex.ldx) |
| Reversi | 8x8 | Outflank to flip | [reversi.ldx](src/ludax/games/reversi.ldx) |
| English Draughts | 8x8 | Diagonal hops + kings | [english_draughts.ldx](src/ludax/games/english_draughts.ldx) |
| Yavalath | Hex 5 | Line of 4 wins, 3 loses | [yavalath.ldx](src/ludax/games/yavalath.ldx) |
| Pente | 19x19 | Line of 5 or 5 captures | [pente.ldx](src/ludax/games/pente.ldx) |
| Wolf and Sheep | 8x8 | Asymmetric chase | [wolf_and_sheep.ldx](src/ludax/games/wolf_and_sheep.ldx) |

### AI-Generated Games (selected)
| Game | Board | Mechanic | File |
|------|-------|----------|------|
| Plumage Wars | 9x9 hex | Custodial flip + score | [plumage_wars.ldx](src/ludax/games/plumage_wars.ldx) |
| Jaguar's Grace | 9x9 hex | Custodial flip + territory | [jaguars_grace.ldx](src/ludax/games/jaguars_grace.ldx) |
| Ember Court | 7x7 | Line of 4 + flip | [ember_court.ldx](src/ludax/games/ember_court.ldx) |
| Cipher League | 9x9 hex | Connection + capture | [cipher_league.ldx](src/ludax/games/cipher_league.ldx) |
| Iron Skies | 9x9 hex | Score by territory | [iron_skies.ldx](src/ludax/games/iron_skies.ldx) |

## Installation

> [!IMPORTANT]
> Requires Python 3.9+. Install [JAX](https://docs.jax.dev/en/latest/installation.html) first for GPU acceleration.

```bash
# Package install
pip install 'ludax[gui,agents]'

# Development install
git clone https://github.com/claycantrell/ludax.git
cd ludax
pip install -r requirements-dev.txt
```

## Usage

### Basic: Run Games at Scale

```python
import jax
from ludax import LudaxEnvironment
from ludax.games import tic_tac_toe

env = LudaxEnvironment(game_str=tic_tac_toe)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
state = init(keys)
# Run 1024 games in parallel on GPU
```

### Transpile Ludii Games

```python
from ludax.ludii_transpiler import transpile
from ludax import LudaxEnvironment
import jax

# Write in Ludii syntax (which LLMs know)
lud = """(game "Breakthrough"
    (players {(player N) (player S)})
    (equipment {(board (square 8))
        (piece "Pawn" Each (or
            (move Step Forward (to if:(is Empty (to))))
            (move Step (directions {FR FL})
                (to if:(is Enemy (who at:(to))) (apply (remove (to)))))))
        (regions P1 (sites Top)) (regions P2 (sites Bottom))})
    (rules
        (start {(place "Pawn1" (expand (sites Bottom)))
                (place "Pawn2" (expand (sites Top)))})
        (play (forEach Piece))
        (end (if (is In (last To) (sites Mover)) (result Mover Win)))))"""

ldx = transpile(lud)
env = LudaxEnvironment(game_str=ldx)
state = env.init(jax.random.PRNGKey(0))
print(f"Legal moves: {state.legal_action_mask.sum()}")
```

### Interactive GUI

```bash
python examples/02-ludax_gui/interactive.py
# Open http://127.0.0.1:8080
```

Features: clickable board, AI opponents (random/greedy/MCTS), auto-generated rules panel.

### Define New Games

```
(game "MyGame"
    (players 2)
    (equipment
        (board (square 8))
        (pieces ("warrior" both) ("knight" both))
    )
    (rules
        (play (repeat (P1 P2)
            (move (or
                (step "warrior" direction:any)
                (leap "knight" offsets:knight capture:true)))))
        (end
            (if (no_legal_actions) (mover lose)))
    )
)
```

## Game DSL Reference

### Board Types
- `(square N)` — NxN grid, 8 directions
- `(rectangle W H)` — WxH grid, 8 directions
- `(hexagon N)` — Hex board with diameter N (must be odd), 6 directions
- `(hex_rectangle W H)` — Rectangular hex grid, 6 directions

### Movement Types
- `(step "piece" direction:any)` — Move 1 cell in any direction
- `(slide "piece" direction:orthogonal)` — Slide any distance until blocked
- `(hop "piece" direction:any hop_over:opponent capture:true)` — Jump over a piece
- `(leap "piece" offsets:knight capture:true)` — Knight/camel/zebra jumps
- `(or (step ...) (hop ...))` — Combine movement types

### Leap Patterns
- `offsets:knight` — (2,1) L-shape, 8 destinations
- `offsets:camel` — (3,1) jump, 8 destinations
- `offsets:zebra` — (3,2) jump, 8 destinations
- `offsets:((2 1) (-2 1) (1 3))` — Custom offset list

### Effects
- `(capture (custodial "piece" N orientation:orthogonal))` — Custodial capture
- `(flip (custodial "piece" N orientation:any))` — Reversi-style flip
- `(promote "old" "new" (edge forward))` — Promote on reaching edge
- `(extra_turn mover)` — Grant extra turn
- `(set_score mover (count (occupied mover)))` — Update score

### End Conditions
- `(if (line "piece" N) (mover win))` — N in a row
- `(if (no_legal_actions) (mover lose))` — No moves = loss
- `(if (full_board) (draw))` — Board full = draw
- `(if (full_board) (by_score))` — Highest score wins
- `(if (>= (connected "piece" ...) 2) (mover win))` — Connection win

### Stacking (in progress)
```
(equipment
    (board (square 8))
    (pieces ("disc" both))
    (stack 4)  ;; max 4 pieces per cell
)
```

## Architecture

```
.ldx game file
    -> Lark parser (grammar.lark)
    -> GameInfoExtractor (game_info.py)   -> GameInfo + GameState namedtuple
    -> GameRuleParser (game_parser.py)    -> compiled JAX functions
    -> LudaxEnvironment (environment.py)  -> gym-like step/init/observe API
```

All game logic compiles to pure JAX functions at init time. No Python overhead during gameplay. Compatible with `jax.jit`, `jax.vmap`, `jax.lax.scan` for massive parallelism.

### Ludii Transpiler Pipeline

```
.lud Ludii file
    -> Permissive Lark parser (ludii_grammar_permissive.lark)
    -> LudiiTranspiler (ludii_transpiler.py)
    -> .ldx Ludax text
    -> Standard Ludax pipeline above
```

## Reinforcement Learning

Train AlphaZero-style agents on any Ludax game:

```bash
python examples/03-alpha-zero/train.py
```

Uses the PGX AlphaZero implementation with Ludax environments. Pretrained agents for bundled games available via `pip install ludax-agents`.

## Acknowledgments

- Original Ludax engine by [Grant Todd](https://github.com/gdrtodd/ludax)
- Inspired by [Ludii](https://ludii.games/) and [PGX](https://github.com/sotetsuk/pgx)
- AI game generation powered by [Claude](https://www.anthropic.com/claude) (Anthropic)
