from importlib.resources import files

def _read_game(game_name: str) -> str:
    with files(__package__).joinpath(f"./{game_name}.ldx").open('r') as f:
        return f.read()

# Package a subset of default game implementations
connect_four = _read_game('connect_four')
connect_six = _read_game('connect_six')
dai_hasami_shogi = _read_game('dai_hasami_shogi')
english_draughts = _read_game('english_draughts')
english_draughts_hex = _read_game('english_draughts_hex')
gridworld = _read_game('gridworld')
hasami_shogi = _read_game('hasami_shogi')
hex = _read_game('hex')
hop_through = _read_game('hop_through')
gomoku = _read_game('gomoku')
pente = _read_game('pente')
reversi = _read_game('reversi')
test = _read_game('test')
tic_tac_toe = _read_game('tic_tac_toe')
yavalath = _read_game('yavalath')
yavalax = _read_game('yavalax')
wolf_and_sheep = _read_game('wolf_and_sheep')

# Auto-discover any additional .ldx files (e.g. generated games)
import os as _os
_games_dir = _os.path.dirname(_os.path.abspath(__file__))
_all_games = []
for _f in sorted(_os.listdir(_games_dir)):
    if _f.endswith('.ldx'):
        _name = _f.replace('.ldx', '')
        if _name not in dir():
            try:
                globals()[_name] = _read_game(_name)
            except Exception:
                continue
        _all_games.append(_name)

__all__ = _all_games
