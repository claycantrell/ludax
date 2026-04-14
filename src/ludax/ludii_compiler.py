"""
Ludii-to-JAX compiler.

Compiles Ludii .lud game descriptions directly to JAX functions,
bypassing the .ldx intermediate format entirely.

Pipeline: .lud text → Ludii parse tree → GameInfo + JAX functions → LudaxEnvironment
"""

import os
import re
import typing
from lark import Lark, Tree, Token

import jax
import jax.numpy as jnp

from .config import (
    BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY, P1, P2,
    ActionTypes, MoveTypes, PlayerAndMoverRefs,
)
from .game_info import GameInfo, RenderingInfo, GameInfoExtractor
from .game_parser import GameRuleParser
from . import utils


_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "ludii_grammar_permissive.lark")
_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        _parser = Lark.open(_GRAMMAR_PATH, start="game", parser="earley", keep_all_tokens=True)
    return _parser


def _sexp_to_str(node) -> str:
    if isinstance(node, Token):
        return str(node)
    if isinstance(node, Tree):
        return " ".join(_sexp_to_str(c) for c in node.children)
    return str(node)


def _find_child(tree: Tree, name: str):
    for c in tree.children:
        if isinstance(c, Tree) and c.data == name:
            return c
    return None


def _find_all(tree: Tree, name: str):
    return [c for c in tree.children if isinstance(c, Tree) and c.data == name]


def _get_text(node) -> str:
    if isinstance(node, Token):
        s = str(node)
        if s in ("(", ")", "{", "}"): return ""
        return s
    if isinstance(node, Tree):
        return " ".join(p for p in (_get_text(c) for c in node.children) if p).strip()
    return str(node)


class LudiiCompiler:
    """Compile a Ludii .lud file directly to JAX functions.

    Parses the Ludii tree, builds GameInfo, then generates the same
    info_dict that GameRuleParser produces — without going through .ldx.
    """

    def __init__(self):
        self.pieces = []          # [(name, owner)]
        self.piece_movements = {} # {name: raw_text}
        self.board_shape = ""
        self.board_ldx = ""
        self.has_set_forward = False
        self.is_mancala = False
        self.mancala_rows = 0
        self.mancala_cols = 0
        self.regions = []

    def compile(self, lud_text: str):
        """Compile Ludii text to (GameInfo, RenderingInfo, game_rules dict).

        Returns the same tuple that LudaxEnvironment expects.
        """
        parser = _get_parser()
        tree = parser.parse(lud_text)
        full_text = _get_text(tree)

        # 1. Extract game structure from Ludii tree
        self._extract_players(tree)
        self._extract_equipment(tree)

        # Detect mancala from sow+track keywords
        if not self.is_mancala and "sow" in full_text.lower() and "track" in full_text.lower():
            self.is_mancala = True
            self._setup_mancala_board()

        self._merge_symmetric_pieces()

        # 2. Build .ldx text from extracted semantics (reuse transpiler logic)
        #    This is the bridge — we generate minimal .ldx and compile it.
        #    Future: replace this with direct tree-to-JAX compilation.
        ldx = self._build_ldx(tree)
        if ldx is None:
            raise ValueError("Failed to compile Ludii game")

        return ldx

    def _extract_players(self, tree):
        players = _find_child(tree, "players")
        if players:
            content = _get_text(players)
            if "player N" in content or "player S" in content:
                self.has_set_forward = True

    def _extract_equipment(self, tree):
        equip = _find_child(tree, "equipment")
        if not equip:
            return

        for item in _find_all(equip, "equip_item"):
            etype = _find_child(item, "equip_type")
            if not etype:
                continue
            type_str = ""
            for c in etype.children:
                if isinstance(c, Token):
                    type_str = str(c)
                    break

            content = _find_child(item, "equip_content")
            content_str = _get_text(content) if content else ""

            if type_str == "board":
                self._parse_board(content_str)
            elif type_str in ("mancalaBoard", "surakartaBoard"):
                self.is_mancala = True
                self._parse_mancala_board(content_str)
            elif type_str == "piece":
                self._parse_piece(content_str)

    def _parse_board(self, content: str):
        """Parse board — reuses transpiler logic."""
        tokens = content.replace("(", " ").replace(")", " ").split()
        if not tokens:
            return

        shape = tokens[0].lower()

        if shape == "diamond":
            if len(tokens) > 1:
                n = tokens[1]
                self.board_ldx = f"hex_rectangle {n} {n}"
                self.board_shape = "hex_rectangle"
            return

        if shape in ("square", "rectangle", "hex", "hexagon", "tri"):
            args = [t for t in tokens[1:] if t.isdigit()]
            if shape in ("hex", "hexagon", "tri"):
                if len(args) >= 3:
                    row_widths = [int(a) for a in args]
                    self.board_ldx = f"hex_rectangle {max(row_widths)} {len(row_widths)}"
                    self.board_shape = "hex_rectangle"
                elif args:
                    size = int(args[0])
                    if size % 2 == 0:
                        size += 1
                    self.board_ldx = f"hexagon {size}"
                    self.board_shape = "hexagon"
                return
            if shape == "rectangle" and len(args) >= 2:
                rows, cols = max(int(args[0]), 1), max(int(args[1]), 2)
                self.board_ldx = f"rectangle {cols} {rows}"
                self.board_shape = "rectangle"
                return
            if args:
                self.board_ldx = f"square {max(int(args[0]), 3)}"
                self.board_shape = "square"
                return

        # Fallback: find largest recognizable shape
        import re as _re
        for shape_name in ["square", "rectangle", "hex", "hexagon"]:
            matches = _re.findall(rf'\b{shape_name}\s+(\d+)(?:\s+(\d+))?', content, _re.IGNORECASE)
            if matches:
                best = max(matches, key=lambda m: int(m[0]))
                n1 = max(int(best[0]), 3)
                if shape_name in ("hex", "hexagon"):
                    if n1 % 2 == 0: n1 += 1
                    self.board_ldx = f"hexagon {n1}"
                    self.board_shape = "hexagon"
                elif shape_name == "rectangle" and best[1]:
                    n2 = max(int(best[1]), 3)
                    self.board_ldx = f"rectangle {max(n2, 2)} {max(n1, 2)}"
                    self.board_shape = "rectangle"
                else:
                    self.board_ldx = f"square {n1}"
                    self.board_shape = "square"
                return

        # Last resort
        if any(kw in content.lower() for kw in ["merge", "add", "remove", "shift", "scale"]):
            nums = [int(t) for t in tokens if t.isdigit() and 3 <= int(t) <= 20]
            size = max(nums) if nums else 7
            self.board_ldx = f"square {size}"
            self.board_shape = "square"

    def _parse_mancala_board(self, content: str):
        nums = [int(t) for t in content.split() if t.isdigit()]
        if len(nums) >= 2:
            self.mancala_rows, self.mancala_cols = nums[0], nums[1]
            self.board_ldx = f"rectangle {nums[1]} {nums[0]}"
        elif nums:
            self.mancala_rows, self.mancala_cols = 2, nums[0]
            self.board_ldx = f"rectangle {nums[0]} 2"
        else:
            self.mancala_rows, self.mancala_cols = 2, 6
            self.board_ldx = "rectangle 6 2"
        self.board_shape = "rectangle"

    def _setup_mancala_board(self):
        if not self.mancala_cols:
            if "square" in self.board_ldx:
                n = int(self.board_ldx.split()[-1])
                self.mancala_rows, self.mancala_cols = 2, (n * n) // 2
            elif "rectangle" in self.board_ldx:
                parts = self.board_ldx.split()
                self.mancala_cols, self.mancala_rows = int(parts[1]), int(parts[2])
            else:
                self.mancala_rows, self.mancala_cols = 2, 6
            self.board_ldx = f"rectangle {self.mancala_cols} {self.mancala_rows}"
            self.board_shape = "rectangle"

    def _parse_piece(self, content: str):
        tokens = content.strip().split()
        if not tokens:
            return
        name = re.sub(r'\d+$', '', tokens[0].strip('"').lower())
        if not name:
            name = tokens[0].strip('"').lower()

        owner = "both"
        for t in tokens[1:]:
            if t == "Each": owner = "both"; break
            elif t == "P1": owner = "P1"; break
            elif t == "P2": owner = "P2"; break
            elif t in ("Neutral", "Shared"): owner = "both"; break

        if name not in self.piece_movements:
            self.piece_movements[name] = content
        else:
            self.piece_movements[name] += " " + content

        for i, (existing_name, existing_owner) in enumerate(self.pieces):
            if existing_name == name:
                if existing_owner != owner:
                    self.pieces[i] = (name, "both")
                return
        self.pieces.append((name, owner))

    def _merge_symmetric_pieces(self):
        if len(self.pieces) < 2:
            return
        p1 = [(i, n) for i, (n, o) in enumerate(self.pieces) if o == "P1"]
        p2 = [(i, n) for i, (n, o) in enumerate(self.pieces) if o == "P2"]
        if not p1 or not p2:
            return

        def _kws(content):
            return {kw for kw in ["Step", "Hop", "Slide", "Leap", "Forward", "Orthogonal", "Diagonal"]
                    if kw in content}

        merged = set()
        for p1_idx, p1_name in p1:
            for p2_idx, p2_name in p2:
                if p2_idx in merged:
                    continue
                if _kws(self.piece_movements.get(p1_name, "")) == _kws(self.piece_movements.get(p2_name, "")):
                    self.pieces[p1_idx] = (p1_name, "both")
                    merged.add(p2_idx)
                    break
        for idx in sorted(merged, reverse=True):
            self.pieces.pop(idx)

    def _build_ldx(self, tree) -> typing.Optional[str]:
        """Build minimal .ldx from Ludii tree — bridge until full direct compilation."""
        # This reuses the transpiler's output generation but from our extracted state.
        # The key difference: this is a method on LudiiCompiler, not a separate class.

        name = "Unknown"
        for c in tree.children:
            if isinstance(c, Token) and c.type == "ESCAPED_STRING":
                name = str(c).strip('"')
                break

        if not self.pieces:
            self.pieces.append(("token", "both"))

        parts = [f'(game "{name}"']

        if self.has_set_forward:
            parts.append("    (players 2 (set_forward (P1 up) (P2 down)))")
        else:
            parts.append("    (players 2)")

        parts.append("    (equipment")
        parts.append(f"        (board ({self.board_ldx}))")
        pieces_str = " ".join(f'("{p}" {o})' for p, o in self.pieces)
        parts.append(f"        (pieces {pieces_str})")
        parts.append("    )")

        # Rules
        parts.append("    (rules")
        start = self._build_start(tree)
        if start:
            parts.append(f"        {start}")
        play = self._build_play(tree)
        if play:
            parts.append(f"        {play}")
        end = self._build_end(tree)
        if end:
            parts.append(f"        {end}")
        parts.append("    )")
        parts.append(")")

        return "\n".join(parts)

    def _build_start(self, tree) -> str:
        """Extract start positions — delegates to transpiler logic."""
        if self.is_mancala:
            return ""

        rules = _find_child(tree, "rules")
        if not rules:
            return ""
        full_text = _get_text(rules)
        if "start" not in full_text:
            return ""

        piece_name = self.pieces[0][0] if self.pieces else "token"

        # Get board dimensions
        board_h = 8
        if "square" in self.board_ldx:
            board_h = int(self.board_ldx.split()[-1])
        elif "rectangle" in self.board_ldx:
            parts = self.board_ldx.split()
            board_h = int(parts[-1]) if len(parts) >= 3 else int(parts[-2])
        elif "hexagon" in self.board_ldx:
            board_h = int(self.board_ldx.split()[-1])
        board_h = max(board_h, 2)

        board_w = board_h
        if "rectangle" in self.board_ldx:
            parts = self.board_ldx.split()
            board_w = int(parts[1]) if len(parts) >= 3 else board_h

        def _row(r):
            return max(0, min(r, board_h - 1))

        # Player-specific pieces
        p1_piece = piece_name
        p2_piece = piece_name
        has_both = any(o == "both" for _, o in self.pieces)
        if not has_both:
            for pn, po in self.pieces:
                if po == "P1": p1_piece = pn; break
            for pn, po in self.pieces:
                if po == "P2": p2_piece = pn; break

        start_ldx = []

        # Row pattern
        row_matches = re.findall(r'place\s+"([^"]+)"\s+(?:\(?sites )?Row (\d+)\)?', full_text)
        for pname_raw, row_num in row_matches:
            pname = re.sub(r'\d+$', '', pname_raw).lower() or piece_name
            player = "P1" if pname_raw.endswith("1") else "P2" if pname_raw.endswith("2") else "P1"
            row = min(int(row_num), board_h - 1)
            start_ldx.append(f'(place "{pname}" {player} ((row {row})))')

        # Coord pattern for chess-like games
        is_chess_like = len(self.pieces) > 2 and "forEach Piece" in full_text
        if is_chess_like and ("square" in self.board_ldx or "rectangle" in self.board_ldx):
            for m in re.finditer(r'place\s+"([^"]+)"\s+((?:coord:)?"[A-Za-z]\d+"(?:\s+"[A-Za-z]\d+")*)', full_text):
                pname_raw = m.group(1)
                coords_str = m.group(2)
                pname = re.sub(r'\d+$', '', pname_raw).lower() or piece_name
                player = "P1" if pname_raw.endswith("1") else "P2" if pname_raw.endswith("2") else "P1"
                coord_values = re.findall(r'"([A-Za-z]\d+)"', coords_str)
                indices = []
                for c in coord_values:
                    col = ord(c[0].upper()) - ord('A')
                    row = int(c[1:]) - 1
                    idx = row * board_w + col
                    if 0 <= idx < board_h * board_w:
                        indices.append(idx)
                if indices:
                    start_ldx.append(f'(place "{pname}" {player} ({" ".join(str(i) for i in indices)}))')

        # 1-row boards
        is_1row = board_h == 1
        if is_1row and not start_ldx:
            total = board_w
            half = total // 2
            if "expand" in full_text or "sites Left" in full_text or "sites Right" in full_text or "sites Bottom" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ({" ".join(str(i) for i in range(half))}))')
                start_ldx.append(f'(place "{p2_piece}" P2 ({" ".join(str(i) for i in range(total - half, total))}))')

        # Expand Bottom/Top
        if not start_ldx and "expand" in full_text:
            if "sites Bottom" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0) (row {_row(1)})))')
            if "sites Top" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-2)}) (row {_row(board_h-1)})))')

        # Sites Left/Right
        if not start_ldx and ("sites Left" in full_text or "sites Right" in full_text):
            if "sites Left" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            if "sites Right" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')

        # Bottom/Top without expand
        if not start_ldx:
            if "sites Bottom" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            if "sites Top" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')

        # Phase N
        if not start_ldx and "sites Phase" in full_text:
            start_ldx.append(f'(place "{p1_piece}" P1 ((row 0) (row {_row(1)}) (row {_row(2)})))')
            start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-3)}) (row {_row(board_h-2)}) (row {_row(board_h-1)})))')

        if start_ldx:
            has_p1 = any("P1" in s for s in start_ldx)
            has_p2 = any("P2" in s for s in start_ldx)
            if has_p1 and not has_p2:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')
            elif has_p2 and not has_p1:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            return "(start " + " ".join(start_ldx) + ")"

        # Fallback auto-placement for movement games
        has_place_in_start = "place" in full_text.split("play")[0] if "play" in full_text else "place" in full_text
        if has_place_in_start and ("forEach Piece" in full_text or "move Step" in full_text or "move Hop" in full_text):
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                if size <= 4:
                    return f'(start (place "{p1_piece}" P1 ((row 0))) (place "{p2_piece}" P2 ((row {size-1}))))'
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {size-2}) (row {size-1}))))'
            elif "hexagon" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {size-2}) (row {size-1}))))'
            elif "rectangle" in self.board_ldx:
                parts = self.board_ldx.split()
                h = int(parts[-1]) if len(parts) >= 3 else 8
                if h <= 4:
                    return f'(start (place "{p1_piece}" P1 ((row 0))) (place "{p2_piece}" P2 ((row {h-1}))))'
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {h-2}) (row {h-1}))))'
        return ""

    def _build_play(self, tree) -> str:
        """Build play rules from Ludii tree."""
        if self.is_mancala:
            rules = _find_child(tree, "rules")
            full = _get_text(rules) if rules else ""
            seeds = 4
            m = re.search(r'set Count (\d+)', full)
            if m:
                seeds = int(m.group(1))
            return f'(play (repeat (P1 P2) (sow seeds:{seeds} capture_opposite)))'

        rules = _find_child(tree, "rules")
        if not rules:
            return ""
        full_text = _get_text(rules)
        rules_content = _find_child(rules, "rules_content")

        # Check for phases
        if rules_content:
            for item in _find_all(rules_content, "rules_item"):
                phases = _find_child(item, "phases")
                if phases:
                    return self._build_phases(phases)

        # Get play text
        play = None
        if rules_content:
            for item in _find_all(rules_content, "rules_item"):
                p = _find_child(item, "play")
                if p:
                    play = p
                    break
        if not play:
            return ""

        play_text = _get_text(play)
        return self._build_play_from_text(play_text, full_text)

    def _build_play_from_text(self, play_text, full_text) -> str:
        """Convert play section to .ldx play rule."""
        # Get movement text including piece definitions
        mt = play_text
        for content in self.piece_movements.values():
            mt += " " + content
        mt += " " + full_text

        is_foreach = "forEach Piece" in play_text
        has_add = "move Add" in mt
        has_step = "move Step" in mt or ("Step" in mt and "move" in mt)
        has_hop = "move Hop" in mt or ("Hop" in mt and "move" in mt)
        has_slide = "move Slide" in mt or ("Slide" in mt and "move" in mt)
        has_leap = "Leap" in mt

        if is_foreach:
            if has_add and not has_step and not has_hop and not has_slide:
                return self._build_placement(play_text)
            if has_add and (has_step or has_hop):
                piece_name = self.pieces[0][0] if self.pieces else "token"
                return f'(play (repeat (P1 P2) (place "{piece_name}" (destination (empty)))))'
            return self._build_foreach_piece(play_text, mt)
        elif "move Add" in play_text:
            return self._build_placement(play_text)
        elif any(kw in play_text for kw in ["move Remove", "move Select", "move Claim"]):
            piece = self.pieces[0][0] if self.pieces else "token"
            return f'(play (repeat (P1 P2) (place "{piece}" (destination (empty)))))'
        elif "priority" in play_text:
            return self._build_foreach_piece(play_text, mt)
        else:
            piece = self.pieces[0][0] if self.pieces else "token"
            return f'(play (repeat (P1 P2) (place "{piece}" (destination (empty)))))'

    def _build_placement(self, play_text) -> str:
        piece_name = self.pieces[0][0] if self.pieces else "token"
        effects = ""
        if "then" in play_text:
            effects = self._extract_effects(play_text)
        if effects:
            return f'(play (repeat (P1 P2) (place "{piece_name}" (destination (empty)) {effects})))'
        return f'(play (repeat (P1 P2) (place "{piece_name}" (destination (empty)))))'

    def _build_foreach_piece(self, play_text, mt) -> str:
        moves = []
        piece_names = [p for p, _ in self.pieces] if len(self.pieces) > 1 else [self.pieces[0][0]]

        has_step = "move Step" in mt or ("Step" in mt and "move" in mt)
        has_hop = "move Hop" in mt or ("Hop" in mt and "move" in mt)
        has_slide = "move Slide" in mt or ("Slide" in mt and "move" in mt)
        has_leap = "Leap" in mt

        if has_slide and (has_step or has_hop):
            has_slide = False
        elif has_slide and "Slide" not in play_text:
            has_slide = False
            has_step = True

        hop_over_target = "opponent"
        if has_hop and "is Friend" in mt and "between" in mt:
            hop_over_target = "mover"

        for pn in piece_names:
            if has_step:
                moves.append(f'(step "{pn}" direction:any)')
            if has_hop:
                moves.append(f'(hop "{pn}" direction:any hop_over:{hop_over_target} capture:true)')

        if has_slide:
            moves.append(f'(slide "{self.pieces[0][0]}" direction:any)')

        if has_leap and not has_step:
            for pn in piece_names:
                moves.append(f'(step "{pn}" direction:any)')

        if not moves:
            for pn in piece_names:
                moves.append(f'(step "{pn}" direction:any)')

        effects = []
        if "remove" in play_text.lower() and ("between" in play_text or "custodial" in play_text):
            effects.append(f'(capture (custodial "{self.pieces[0][0]}" 1 orientation:orthogonal))')
        if "moveAgain" in play_text:
            effects.append('(if (and (action_was mover hop) (can_move_again hop)) (extra_turn mover same_piece:true))')

        effects_str = " (effects " + " ".join(effects) + ")" if effects else ""

        if len(moves) == 1:
            return f'(play (repeat (P1 P2) (move {moves[0]}{effects_str})))'
        return f'(play (repeat (P1 P2) (move (or {" ".join(moves)}){effects_str})))'

    def _build_phases(self, phases) -> str:
        phase_texts = _get_text(phases)
        phase_blocks = re.findall(r'phase\s+"([^"]+)"(.*?)(?=phase\s+"|$)', phase_texts, re.DOTALL)

        ldx_phases = []
        for _, phase_content in phase_blocks:
            if "move Add" in phase_content:
                piece = self.pieces[0][0] if self.pieces else "token"
                if "nextPhase" in phase_content:
                    ldx_phases.append(f'(once_through (P1 P2) (place "{piece}" (destination (empty))))')
                else:
                    ldx_phases.append(f'(repeat (P1 P2) (place "{piece}" (destination (empty))))')
            elif "forEach Piece" in phase_content:
                mt = phase_content
                for content in self.piece_movements.values():
                    mt += " " + content
                result = self._build_foreach_piece(phase_content, mt)
                if result.startswith("(play "):
                    result = result[6:-1]
                ldx_phases.append(result)
            else:
                mt = phase_content
                for content in self.piece_movements.values():
                    mt += " " + content
                result = self._build_foreach_piece(phase_content, mt)
                if result.startswith("(play "):
                    result = result[6:-1]
                ldx_phases.append(result)

        if not ldx_phases:
            return ""
        return "(play\n            " + "\n            ".join(ldx_phases) + "\n        )"

    def _build_end(self, tree) -> str:
        rules = _find_child(tree, "rules")
        if not rules:
            return ""
        full_text = _get_text(rules)
        conditions = []
        piece = self.pieces[0][0] if self.pieces else "token"

        if "is Line" in full_text:
            for m in re.finditer(r'is Line (\d+)(.*?)result (\w+) (\w+)', full_text):
                n, between, player, outcome = m.group(1), m.group(2), m.group(3), m.group(4)
                exact = " exact:true" if "exact:True" in between or "exact:true" in between else ""
                if outcome == "Loss":
                    conditions.append(f'(if (line "{piece}" {n}{exact}) (mover lose))')
                elif outcome == "Win":
                    conditions.append(f'(if (line "{piece}" {n}{exact}) (mover win))')

        if "is Connected" in full_text:
            if self.has_set_forward:
                conditions.append(f'(if (>= (connected "{piece}" ((edge forward) (edge backward))) 2) (mover win))')
            else:
                conditions.append(f'(if (>= (connected "{piece}" ((edge left) (edge right))) 2) (mover win))')

        if "no Moves" in full_text:
            ctx = full_text[full_text.find("no Moves"):full_text.find("no Moves")+100]
            if "Next" in ctx:
                conditions.append("(if (no_legal_actions) (opponent lose))")
            else:
                conditions.append("(if (no_legal_actions) (mover lose))")

        if "no Pieces" in full_text:
            conditions.append("(if (captured_all opponent) (mover win))")

        if "is In" in full_text and ("sites Mover" in full_text or "sites Top" in full_text):
            if self.has_set_forward:
                conditions.append(f'(if (exists (and (occupied mover) (edge forward))) (mover win))')

        if "is Full" in full_text or "full_board" in full_text.lower():
            conditions.append("(if (full_board) (draw))")

        if not conditions:
            conditions.append("(if (no_legal_actions) (mover lose))")

        return "(end " + " ".join(conditions) + ")"

    def _extract_effects(self, text) -> str:
        effects = []
        if "addScore" in text or "set Score" in text:
            effects.append('(set_score mover (count (occupied mover)))')
            effects.append('(set_score opponent (count (occupied opponent)))')
        return f'(effects {" ".join(effects)})' if effects else ""


def compile_ludii(lud_text: str):
    """Compile Ludii .lud text to (game_info, rendering_info, game_rules).

    This is the main entry point. Returns the same format that
    LudaxEnvironment expects.
    """
    compiler = LudiiCompiler()
    ldx = compiler.compile(lud_text)

    # Use existing .ldx compilation pipeline
    from lark import Lark
    grammar_path = os.path.join(os.path.dirname(__file__), "grammar.lark")
    ldx_parser = Lark.open(grammar_path, start="game")
    ldx_tree = ldx_parser.parse(ldx)

    game_info, rendering_info = GameInfoExtractor()(ldx_tree)
    game_rules = GameRuleParser(game_info).transform(ldx_tree)

    return game_info, rendering_info, game_rules
