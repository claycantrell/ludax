from importlib.resources import files

def _read_game(game_name: str) -> str:
    with files(__package__).joinpath(f"./{game_name}.ldx").open('r') as f:
        return f.read()

# Package a subset of default game implementations
complexity_demo = _read_game('complexity_demo')
connect_four = _read_game('connect_four')
connect_six = _read_game('connect_six')
dai_hasami_shogi = _read_game('dai_hasami_shogi')
hex = _read_game('hex')
hop_through = _read_game('hop_through')
gomoku = _read_game('gomoku')
pente = _read_game('pente')
reversi = _read_game('reversi')
tic_tac_toe = _read_game('tic_tac_toe')
yavalath = _read_game('yavalath')
yavalax = _read_game('yavalax')

# List of all games available in the package
__all__ = [
    "complexity_demo",
    "connect_four",
    "connect_six",
    "dai_hasami_shogi",
    "hex",
    "hop_through",
    "gomoku",
    "pente",
    "reversi",
    "tic_tac_toe",
    "yavalath",
    "yavalax"
]
