# Ludax

A JAX-accelerated board game engine with a domain-specific language for defining games. Compiles `.ldx` game descriptions into hardware-accelerated environments that run at hundreds of thousands of steps per second on CPU and tens of millions on GPU.

This is a fork of [gdrtodd/ludax](https://github.com/gdrtodd/ludax) with a unified action model, N-player support, a Ludii transpiler, and AI-generated game support.

![Throughput of Ludax environments compared to PGX and Ludii implementations](/assets/throughput_comparison.png)

## Architecture

Every game compiles down to two pure JAX functions — `legal_action_mask(state)` and `apply_action(state, action)` — with zero Python overhead at runtime. Compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan` for massive parallelism.

```
.ldx game file
    -> Lark parser (grammar.lark)
    -> GameInfoExtractor (game_info.py)   -> GameInfo + GameState namedtuple
    -> GameRuleParser (game_parser.py)    -> compiled JAX functions
    -> LudaxEnvironment (environment.py)  -> gym-like step/init/observe API
```

### Unified Action Model

All movement uses a single `(source, destination)` action space. Step, slide, hop, and leap only differ in which destinations are reachable — they all produce the same `(board_size, board_size)` mask. No special-case combiners. Adding a new movement type is ~15 lines: a reachability function plus an optional side effect.

Placement-only games (Tic-Tac-Toe, Hex, Reversi) use a compact `(board_size,)` action space.

### N-Player Support

All game state arrays are parameterized by `num_players`. The engine supports 2+ players without code changes — only needs grammar tokens for P3/P4 references. No hard-coded `% 2` anywhere.

### Ludii Transpiler

Transpile games written in [Ludii](https://ludii.games/) syntax (which LLMs know from training data) into JAX environments:

```python
from ludax.ludii_transpiler import transpile
from ludax import LudaxEnvironment

lud = '(game "MyGame" (players 2) (equipment {(board (square 8)) (piece "Marker" Each)}) ...)'
ldx = transpile(lud)
env = LudaxEnvironment(game_str=ldx)
```

- 100% Ludii grammar parsing (1,212/1,212 games)
- 87% end-to-end Ludii-to-JAX compilation on a random 100-game sample
- Handles: forEach Piece, multi-phase games, coordinate placement, hop-over-friendly captures, chess-like multi-piece games

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

### Run Games at Scale

```python
import jax
import jax.numpy as jnp
from ludax import LudaxEnvironment
from ludax.games import tic_tac_toe

env = LudaxEnvironment(game_str=tic_tac_toe)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
state = init(keys)
# 1024 games in parallel — ~760K steps/sec on CPU
```

### Define Games

```
(game "KnightBattle"
    (players 2)
    (equipment
        (board (square 8))
        (pieces ("pawn" both) ("knight" both))
    )
    (rules
        (start
            (place "pawn" P1 ((row 1)))
            (place "pawn" P2 ((row 6)))
            (place "knight" P1 (1 6))
            (place "knight" P2 (57 62))
        )
        (play (repeat (P1 P2)
            (move (or
                (step "pawn" direction:any)
                (leap "knight" offsets:knight capture:true)
            ))
        ))
        (end
            (if (captured_all opponent) (mover win))
            (if (no_legal_actions) (mover lose))
        )
    )
)
```

### Interactive GUI

```bash
python examples/02-ludax_gui/interactive.py
# Open http://127.0.0.1:8080
```

Clickable board, AI opponents (random/greedy/MCTS), auto-generated rules panel.

## Game DSL Reference

### Boards
| Syntax | Description |
|--------|-------------|
| `(square N)` | NxN grid, 8 directions |
| `(rectangle W H)` | WxH grid, 8 directions |
| `(hexagon N)` | Hex board, diameter N (odd), 6 directions |
| `(hex_rectangle W H)` | Rectangular hex grid, 6 directions |

### Movement
| Syntax | Description |
|--------|-------------|
| `(step "piece" direction:any)` | Move 1 cell |
| `(step "piece" direction:any distance:2)` | Move exactly 2 cells |
| `(slide "piece" direction:orthogonal)` | Slide until blocked |
| `(hop "piece" direction:any hop_over:opponent capture:true)` | Jump over enemy, capture |
| `(hop "piece" direction:any hop_over:mover capture:true)` | Jump over own piece, capture at destination |
| `(leap "piece" offsets:knight capture:true)` | Knight L-shape jump |
| `(leap "piece" offsets:camel)` | (3,1) jump |
| `(leap "piece" offsets:((2 1) (-2 1))` | Custom offsets |
| `(or (step ...) (hop ...))` | Combine movement types |

### Effects
| Syntax | Description |
|--------|-------------|
| `(capture (custodial "piece" N orientation:orthogonal))` | Custodial capture |
| `(capture_to_hand (custodial "piece" N))` | Capture to hand (Shogi) |
| `(flip (custodial "piece" N orientation:any))` | Reversi-style flip |
| `(promote "old" "new" (edge forward))` | Promote at edge |
| `(extra_turn mover same_piece:true)` | Chain captures |
| `(swap (adjacent (prev_move mover)))` | Swap with neighbor |
| `(set_score mover (count (occupied mover)))` | Update score |

### End Conditions
| Syntax | Description |
|--------|-------------|
| `(if (line "piece" N) (mover win))` | N in a row |
| `(if (captured_all opponent) (mover win))` | Eliminate all enemy pieces |
| `(if (no_legal_actions) (mover lose))` | No moves = loss |
| `(if (full_board) (by_score))` | Highest score wins |
| `(if (>= (connected "piece" ...) 2) (mover win))` | Connection win |

### Turn Structure
| Syntax | Description |
|--------|-------------|
| `(repeat (P1 P2) ...)` | Alternating turns |
| `(once_through (P1 P2) ...)` | Each player acts once, then next phase |
| `(actions_per_turn 2)` | 2 actions per turn |
| `(force_pass)` | Must pass when no legal moves |

### Equipment Options
| Syntax | Description |
|--------|-------------|
| `(stack 4)` | Max 4 pieces per cell |
| `(hand 10)` | Reserve hand, max 10 pieces |
| `(drop "piece" (destination (empty)))` | Place from hand |

## Bundled Games

### Classic
| Game | File |
|------|------|
| Tic-Tac-Toe | [tic_tac_toe.ldx](src/ludax/games/tic_tac_toe.ldx) |
| Connect-Four | [connect_four.ldx](src/ludax/games/connect_four.ldx) |
| Gomoku | [gomoku.ldx](src/ludax/games/gomoku.ldx) |
| Hex | [hex.ldx](src/ludax/games/hex.ldx) |
| Reversi | [reversi.ldx](src/ludax/games/reversi.ldx) |
| English Draughts | [english_draughts.ldx](src/ludax/games/english_draughts.ldx) |
| Yavalath | [yavalath.ldx](src/ludax/games/yavalath.ldx) |
| Pente | [pente.ldx](src/ludax/games/pente.ldx) |
| Wolf and Sheep | [wolf_and_sheep.ldx](src/ludax/games/wolf_and_sheep.ldx) |

### AI-Generated (selected)
| Game | File |
|------|------|
| Plumage Wars | [plumage_wars.ldx](src/ludax/games/plumage_wars.ldx) |
| Jaguar's Grace | [jaguars_grace.ldx](src/ludax/games/jaguars_grace.ldx) |
| Ember Court | [ember_court.ldx](src/ludax/games/ember_court.ldx) |
| Cipher League | [cipher_league.ldx](src/ludax/games/cipher_league.ldx) |
| Iron Skies | [iron_skies.ldx](src/ludax/games/iron_skies.ldx) |

40+ AI-generated games total in `src/ludax/games/`.

## Reinforcement Learning

```bash
python examples/03-alpha-zero/train.py
```

Uses PGX AlphaZero with Ludax environments. Pretrained agents available via `pip install ludax-agents`.

## Acknowledgments

- Original Ludax engine by [Grant Todd](https://github.com/gdrtodd/ludax)
- Inspired by [Ludii](https://ludii.games/) and [PGX](https://github.com/sotetsuk/pgx)
- AI game generation powered by [Claude](https://www.anthropic.com/claude) (Anthropic)
