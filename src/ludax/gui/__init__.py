try:
    from flask import Flask
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The ludax's optional GUI depends on Flask. Install all dependencies with:\n\n"
        "    pip install ludax[gui]\n"
    ) from exc
app = Flask(__name__)

from . import routes
