from importlib.resources import files
with files(__package__).joinpath('grammar.lark').open('r') as f:
    grammar = f.read()

from .environment import LudaxEnvironment

__all__ = [
    'LudaxEnvironment'
]
