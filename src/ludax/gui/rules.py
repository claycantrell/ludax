"""
Generate human-readable rules from a Ludax game string.
Parses the .ldx S-expressions directly — no LLM needed.
"""

import re


def _find_block(game_str: str, keyword: str) -> str:
    """Extract the content of a top-level block like (play ...) or (end ...)."""
    # Match (keyword followed by space or newline, not (keyword... as substring
    pattern = f"({keyword}" + r"[\s)]"
    m = re.search(re.escape(f"({keyword}") + r"[\s\)]", game_str)
    if not m:
        return ""
    idx = m.start()
    depth = 0
    for i in range(idx, len(game_str)):
        if game_str[i] == "(":
            depth += 1
        elif game_str[i] == ")":
            depth -= 1
            if depth == 0:
                return game_str[idx:i + 1]
    return ""


def _extract_pieces(game_str: str) -> list:
    """Extract piece names and ownership."""
    pieces = []
    for m in re.finditer(r'\("(\w+)"\s+(P1|P2|both)\)', game_str):
        pieces.append((m.group(1), m.group(2)))
    return pieces


def _describe_board(game_str: str) -> str:
    m = re.search(r'\(board\s+\((\w+)\s+([^)]+)\)\)', game_str)
    if not m:
        return "Unknown board"
    shape, args = m.group(1), m.group(2).strip()
    names = {"square": "square", "hexagon": "hexagonal", "hex_rectangle": "hex-rectangle", "rectangle": "rectangular"}
    shape_name = names.get(shape, shape)
    return f"{shape_name} board ({args})"


def _describe_movement(play_block: str) -> list:
    rules = []
    if "(place" in play_block:
        rules.append("Place one of your pieces on an empty cell each turn")
    if "(step" in play_block:
        pieces = re.findall(r'\(step\s+"(\w+)"\s+direction:(\w+)', play_block)
        for piece, direction in pieces:
            rules.append(f"Move a {piece} one step in {direction.replace('_', ' ')} direction")
    if "(slide" in play_block:
        pieces = re.findall(r'\(slide\s+"(\w+)"\s+direction:(\w+)', play_block)
        for piece, direction in pieces:
            rules.append(f"Slide a {piece} any distance in {direction.replace('_', ' ')} direction")
    if "(hop" in play_block:
        pieces = re.findall(r'\(hop\s+"(\w+)"\s+direction:(\w+)', play_block)
        for piece, direction in pieces:
            cap = " (capturing the jumped piece)" if "capture:true" in play_block else ""
            rules.append(f"Hop a {piece} over a piece in {direction.replace('_', ' ')} direction{cap}")
    if "(force_pass)" in play_block:
        rules.append("You must pass if you have no legal moves")
    return rules


def _describe_effects(play_block: str) -> list:
    effects = []
    if "(capture" in play_block:
        if "custodial" in play_block:
            m = re.search(r'custodial\s+"(\w+)"\s+(\w+)\s+orientation:(\w+)', play_block)
            orient = m.group(3) if m else "any"
            piece = m.group(1) if m else "pieces"
            dirs = {"orthogonal": "in a straight line (not diagonal)",
                    "diagonal": "diagonally",
                    "any": "in any direction"}
            effects.append(
                f"Custodial capture: if your {piece} sandwiches an opponent's {piece} "
                f"between two of yours {dirs.get(orient, orient)}, the trapped piece is removed")
        else:
            effects.append("Capture opponent pieces")
    if "(flip" in play_block:
        effects.append("Flip: opponent pieces sandwiched between yours change to your color")
    if "(promote" in play_block:
        m = re.search(r'\(promote\s+"(\w+)"\s+"(\w+)"', play_block)
        if m:
            effects.append(f"Promote {m.group(1)} to {m.group(2)} when reaching the far edge")
    if "(extra_turn" in play_block:
        effects.append("Get an extra turn after capturing")
    if "(set_score" in play_block or "(increment_score" in play_block:
        if "count (occupied mover)" in play_block:
            effects.append("Your score = number of your pieces on the board")
        elif "count (occupied opponent)" in play_block:
            effects.append("Score tracks opponent's remaining pieces")
        else:
            effects.append("Score points based on board position")
    return effects


def _describe_end(end_block: str) -> list:
    conditions = []
    # Line wins (skip lines that are loss conditions — handled below)
    for m in re.finditer(r'\(if\s+\(line\s+"(\w+)"\s+(\d+)[^)]*\)\s+\(mover win\)', end_block):
        exact = " exactly" if "exact:true" in end_block[m.start():m.start()+80] else ""
        conditions.append(f"Form{exact} {m.group(2)} {m.group(1)}s in a row to win")
    # Connected
    if "(connected" in end_block:
        conditions.append("Connect your pieces across opposite edges of the board")
    # No legal actions
    if "(no_legal_actions)" in end_block:
        if "(mover win)" in end_block[end_block.find("no_legal_actions"):end_block.find("no_legal_actions")+50]:
            conditions.append("Win if your opponent has no legal moves")
        else:
            conditions.append("Lose if you have no legal moves")
    # Full board
    if "(full_board)" in end_block:
        if "(by_score)" in end_block:
            conditions.append("When the board is full, highest score wins")
        elif "(draw)" in end_block:
            conditions.append("Draw if the board fills up")
    # Captured all
    if "(captured_all" in end_block:
        m = re.search(r'\(captured_all\s+"(\w+)"\)', end_block)
        if m:
            conditions.append(f"Win by capturing all opponent's {m.group(1)} pieces")
    # Score
    if "(by_score)" in end_block and "(full_board)" not in end_block:
        conditions.append("Highest score wins")
    # Lose conditions
    for m in re.finditer(r'\(if\s+\(line\s+"(\w+)"\s+(\d+)\)\s+\(mover lose\)', end_block):
        conditions.append(f"LOSE if you form {m.group(2)} {m.group(1)}s in a row!")
    return conditions


def generate_rules(game_str: str) -> str:
    """Generate human-readable rules HTML from a Ludax game string."""
    lines = []

    # Game name
    m = re.search(r'\(game\s+"([^"]+)"', game_str)
    name = m.group(1) if m else "Unknown Game"

    # Board
    board = _describe_board(game_str)
    lines.append(f"<strong>Board:</strong> {board}")

    # Pieces
    pieces = _extract_pieces(game_str)
    if pieces:
        piece_desc = []
        for pname, owner in pieces:
            if owner == "both":
                piece_desc.append(f"{pname} (both players)")
            else:
                piece_desc.append(f"{pname} ({owner} only)")
        lines.append(f"<strong>Pieces:</strong> {', '.join(piece_desc)}")

    # Turn structure
    if "(repeat (P1 P2)" in game_str:
        lines.append("<strong>Turns:</strong> Players alternate turns (P1 first)")
    elif "(once_through" in game_str:
        lines.append("<strong>Turns:</strong> Each player gets one action per round")

    # Movement / placement
    play_block = _find_block(game_str, "play")
    moves = _describe_movement(play_block)
    if moves:
        lines.append("<strong>On your turn:</strong>")
        for m in moves:
            lines.append(f"&nbsp;&nbsp;- {m}")

    # Effects
    effects = _describe_effects(play_block)
    if effects:
        lines.append("<strong>Effects:</strong>")
        for e in effects:
            lines.append(f"&nbsp;&nbsp;- {e}")

    # Win/loss conditions
    end_block = _find_block(game_str, "end")
    conditions = _describe_end(end_block)
    if conditions:
        lines.append("<strong>How to win:</strong>")
        for c in conditions:
            lines.append(f"&nbsp;&nbsp;- {c}")

    return "<br>".join(lines)
