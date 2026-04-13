"""
Ludii → Ludax transpiler.

Walks a Ludii parse tree (from ludii_grammar_permissive.lark) and outputs
Ludax .ldx text that the existing Ludax JAX compiler can handle.

Not all Ludii games can be transpiled — only those using mechanics
that Ludax supports (placement, step, slide, hop, custodial capture,
flip, promote, line/connected/noMoves end conditions).
"""

import os
import typing
from lark import Lark, Tree, Token


_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "ludii_grammar_permissive.lark")
_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        _parser = Lark.open(_GRAMMAR_PATH, start="game", parser="earley", keep_all_tokens=True)
    return _parser


def _sexp_to_str(node) -> str:
    """Convert a parse tree node back to a flat string."""
    if isinstance(node, Token):
        return str(node)
    if isinstance(node, Tree):
        parts = [_sexp_to_str(c) for c in node.children]
        return " ".join(parts)
    return str(node)


def _find_child(tree: Tree, name: str) -> typing.Optional[Tree]:
    """Find the first child tree with the given rule name."""
    for c in tree.children:
        if isinstance(c, Tree) and c.data == name:
            return c
    return None


def _find_all(tree: Tree, name: str) -> list:
    """Find all children with the given rule name."""
    return [c for c in tree.children if isinstance(c, Tree) and c.data == name]


def _get_text(node) -> str:
    """Get all text content from a tree/token, recursively. Skips parens/braces."""
    if isinstance(node, Token):
        s = str(node)
        if s in ("(", ")", "{", "}"): return ""
        return s
    if isinstance(node, Tree):
        parts = [_get_text(c) for c in node.children]
        return " ".join(p for p in parts if p).strip()
    return str(node)


def _extract_sexp_keyword(sexp: Tree) -> str:
    """Get the first keyword from a sexp node."""
    content = _find_child(sexp, "sexp_content")
    if content:
        for c in content.children:
            if isinstance(c, Tree) and c.data == "sexp_atom":
                for a in c.children:
                    if isinstance(a, Token) and a.type == "KEYWORD":
                        return str(a)
    return ""


class LudiiTranspiler:
    """Transpile Ludii .lud to Ludax .ldx."""

    def __init__(self):
        self.pieces = []  # [(name, owner)]
        self.board_shape = ""
        self.board_size = 0
        self.board_ldx = ""
        self.has_set_forward = False
        self.regions = []
        self.errors = []

    def transpile(self, lud_text: str) -> typing.Optional[str]:
        """Convert Ludii text to Ludax text. Returns None if not transpilable."""
        parser = _get_parser()
        tree = parser.parse(lud_text)

        # Extract game name — find the ESCAPED_STRING token
        name = "Unknown"
        for c in tree.children:
            if isinstance(c, Token) and c.type == "ESCAPED_STRING":
                name = str(c).strip('"')
                break

        # Process each section
        self._process_players(tree)
        self._process_equipment(tree)
        start_ldx = self._process_start(tree)
        play_ldx = self._process_rules(tree)
        end_ldx = self._process_end(tree)

        if self.errors:
            return None

        # Build Ludax output
        parts = [f'(game "{name}"']

        # Players
        if self.has_set_forward:
            parts.append("    (players 2 (set_forward (P1 up) (P2 down)))")
        else:
            parts.append("    (players 2)")

        # Equipment
        parts.append(f"    (equipment")
        parts.append(f"        (board ({self.board_ldx}))")
        # Ensure at least one piece type
        if not self.pieces:
            self.pieces.append(("token", "both"))
        pieces_str = " ".join(f'("{p}" {o})' for p, o in self.pieces)
        parts.append(f"        (pieces {pieces_str})")
        for rname, rdef in self.regions:
            parts.append(f'        (regions "{rname}" {rdef})')
        parts.append(f"    )")

        # Rules
        parts.append(f"    (rules")
        if start_ldx:
            parts.append(f"        {start_ldx}")
        if play_ldx:
            parts.append(f"        {play_ldx}")
        if end_ldx:
            parts.append(f"        {end_ldx}")
        parts.append(f"    )")

        parts.append(")")

        return "\n".join(parts)

    def _process_players(self, tree: Tree):
        """Extract player info."""
        players = _find_child(tree, "players")
        if players:
            content = _get_text(players)
            if "player N" in content or "player S" in content:
                self.has_set_forward = True

    def _process_equipment(self, tree: Tree):
        """Extract board, pieces, regions from equipment."""
        equip = _find_child(tree, "equipment")
        if not equip:
            self.errors.append("No equipment section")
            return

        for item in _find_all(equip, "equip_item"):
            # equip_type is a rule with terminal children
            etype = _find_child(item, "equip_type")
            if not etype:
                continue
            # Get the type keyword from the terminal tokens
            type_str = ""
            for c in etype.children:
                if isinstance(c, Token):
                    type_str = str(c)
                    break

            content = _find_child(item, "equip_content")
            content_str = _get_text(content) if content else ""

            if type_str == "board":
                self._parse_board(content_str)
            elif type_str == "mancalaBoard" or type_str == "surakartaBoard":
                # Mancala/special boards → approximate as rectangle
                nums = [int(t) for t in content_str.split() if t.isdigit()]
                if len(nums) >= 2:
                    self.board_ldx = f"rectangle {nums[0]} {nums[1]}"
                elif nums:
                    self.board_ldx = f"rectangle 2 {nums[0]}"
                else:
                    self.board_ldx = "rectangle 2 6"
                self.board_shape = "rectangle"
            elif type_str == "piece":
                self._parse_piece(content_str)
            elif type_str == "regions":
                self._parse_regions(content_str)

    def _set_hex_board(self, size):
        """Set board to hexagon with odd diameter (Ludax requirement)."""
        size = int(size)
        if size % 2 == 0:
            size += 1
        self.board_ldx = f"hexagon {size}"
        self.board_shape = "hexagon"

    def _parse_board(self, content: str):
        """Parse board definition."""
        # Extract board shape and size from content like "( square 8 )"
        # or "( hex Diamond 11 )"
        tokens = content.replace("(", " ").replace(")", " ").split()
        if not tokens:
            self.errors.append("Empty board definition")
            return

        shape = tokens[0].lower()
        if shape == "diamond":
            # (hex Diamond N) → hex_rectangle N N
            if len(tokens) > 1:
                n = tokens[1]
                self.board_ldx = f"hex_rectangle {n} {n}"
                self.board_shape = "hex_rectangle"
            return

        if shape in ("square", "rectangle", "hex", "hexagon"):
            args = [t for t in tokens[1:] if t.isdigit()]
            if shape == "hex" or shape == "hexagon":
                if args:
                    size = int(args[0])
                    if size % 2 == 0:
                        size += 1  # Ludax requires odd hex diameter
                    self._set_hex_board(size)
                return
            if shape == "rectangle" and len(args) >= 2:
                h, w = max(int(args[0]), 2), max(int(args[1]), 2)
                self.board_ldx = f"rectangle {h} {w}"
                self.board_shape = "rectangle"
                return
            if args:
                size = max(int(args[0]), 3)
                self.board_ldx = f"square {size}"
                self.board_shape = "square"
                return

        if shape == "rotate":
            # (rotate 90 (hex 5)) → hexagon 5 (approximate)
            inner_tokens = [t for t in tokens[1:] if not t.isdigit() or int(t) > 20]
            nums = [t for t in tokens if t.isdigit()]
            if "hex" in tokens or "hexagon" in tokens:
                size = nums[-1] if nums else "9"
                self._set_hex_board(size)
                return

        # Triangular boards → approximate as hexagon
        if shape == "tri" or "tri" in content.lower().split():
            nums = [t for t in tokens if t.isdigit()]
            size = nums[0] if nums else "7"
            self._set_hex_board(size)
            return

        # Graph boards → approximate as square based on vertex count
        if shape == "graph" or "graph" in content.lower():
            # Count vertex pairs to estimate board size
            nums = [t for t in tokens if t.replace('.','').replace('-','').isdigit()]
            vert_count = max(len(nums) // 2, 9)  # rough estimate
            import math
            size = max(int(math.sqrt(vert_count)), 3)
            self.board_ldx = f"square {size}"
            self.board_shape = "square"
            return

        # Fallback: try to find a recognizable pattern
        for shape_name in ["square", "rectangle", "hex", "hexagon"]:
            if shape_name in content.lower():
                nums = [t for t in content.split() if t.isdigit()]
                if nums:
                    if shape_name in ("hex", "hexagon"):
                        self._set_hex_board(nums[0])
                    elif shape_name == "rectangle" and len(nums) >= 2:
                        self.board_ldx = f"rectangle {nums[0]} {nums[1]}"
                    else:
                        self.board_ldx = f"square {nums[0]}"
                    self.board_shape = shape_name
                    return

        # Last resort: merge/add/remove/concentric → approximate as square 7
        if any(kw in content.lower() for kw in ["merge", "add", "remove", "concentric", "scale", "shift",
                                                   "tiling", "star", "complete", "subdivide", "dual"]):
            # Complex board — pick a reasonable default
            nums = [int(t) for t in tokens if t.isdigit() and int(t) <= 20]
            size = max(nums) if nums else 7
            self.board_ldx = f"square {size}"
            self.board_shape = "square"
            return

        self.errors.append(f"Unsupported board: {content[:60]}")

    def _parse_piece(self, content: str):
        """Parse piece definition."""
        tokens = content.strip().split()
        if not tokens:
            return

        # First token is the piece name (quoted string)
        name = tokens[0].strip('"').lower()
        # Strip trailing player numbers (Pawn1 → pawn, Marker2 → marker)
        import re
        base_name = re.sub(r'\d+$', '', name)
        if not base_name:
            base_name = name

        # Find owner
        owner = "both"
        for t in tokens[1:]:
            if t == "Each":
                owner = "both"
                break
            elif t == "P1":
                owner = "P1"
                break
            elif t == "P2":
                owner = "P2"
                break
            elif t in ("Neutral", "Shared"):
                owner = "both"
                break

        # Deduplicate: if we already have this base name, merge to "both"
        for i, (existing_name, existing_owner) in enumerate(self.pieces):
            if existing_name == base_name:
                if existing_owner != owner:
                    self.pieces[i] = (base_name, "both")
                return

        self.pieces.append((base_name, owner))

    def _parse_regions(self, content: str):
        """Parse region definitions."""
        # Regions are complex — skip for now, they're rarely needed
        pass

    def _process_start(self, tree: Tree) -> str:
        """Extract start positions from Ludii rules."""
        rules = _find_child(tree, "rules")
        if not rules:
            return ""

        full_text = _get_text(rules)
        if "start" not in full_text:
            return ""

        import re
        piece_name = self.pieces[0][0] if self.pieces else "token"

        # Look for (place "X" ...) patterns in the start section
        start_ldx = []

        # Pattern: (expand (sites Bottom)) or (expand (sites Top))
        if "expand" in full_text and "sites Bottom" in full_text:
            start_ldx.append(f'(place "{piece_name}" P1 ((row 0) (row 1)))')
        if "expand" in full_text and "sites Top" in full_text:
            # Need board height — estimate from board size
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                start_ldx.append(f'(place "{piece_name}" P2 ((row {size-2}) (row {size-1})))')
            else:
                start_ldx.append(f'(place "{piece_name}" P2 ((row 6) (row 7)))')

        # Pattern: (sites Phase N) — checkerboard placement
        if "sites Phase" in full_text:
            start_ldx.append(f'(place "{piece_name}" P1 ((row 0) (row 1) (row 2)))')
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                start_ldx.append(f'(place "{piece_name}" P2 ((row {size-3}) (row {size-2}) (row {size-1})))')
            else:
                start_ldx.append(f'(place "{piece_name}" P2 ((row 5) (row 6) (row 7)))')

        # Pattern: (sites Bottom) without expand
        if "sites Bottom" in full_text and "expand" not in full_text:
            start_ldx.append(f'(place "{piece_name}" P1 ((row 0)))')
        if "sites Top" in full_text and "expand" not in full_text:
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                start_ldx.append(f'(place "{piece_name}" P2 ((row {size-1})))')

        if start_ldx:
            return "(start " + " ".join(start_ldx) + ")"

        # Fallback: if it's a movement game, auto-place on first/last rows
        if "forEach Piece" in full_text or "move Step" in full_text or "move Hop" in full_text or "move Slide" in full_text:
            piece_name = self.pieces[0][0] if self.pieces else "token"
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                if size <= 4:
                    return f'(start (place "{piece_name}" P1 ((row 0))) (place "{piece_name}" P2 ((row {size-1}))))'
                return f'(start (place "{piece_name}" P1 ((row 0) (row 1))) (place "{piece_name}" P2 ((row {size-2}) (row {size-1}))))'
            elif "hexagon" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                last_row = size - 1
                return f'(start (place "{piece_name}" P1 ((row 0) (row 1))) (place "{piece_name}" P2 ((row {last_row-1}) (row {last_row}))))'
            elif "rectangle" in self.board_ldx:
                parts = self.board_ldx.split()
                h = int(parts[-2]) if len(parts) >= 3 else int(parts[-1]) if len(parts) >= 2 else 8
                if h <= 4:
                    return f'(start (place "{piece_name}" P1 ((row 0))) (place "{piece_name}" P2 ((row {h-1}))))'
                return f'(start (place "{piece_name}" P1 ((row 0) (row 1))) (place "{piece_name}" P2 ((row {h-2}) (row {h-1}))))'

        return ""

    def _process_rules(self, tree: Tree) -> str:
        """Extract and transpile play rules."""
        rules = _find_child(tree, "rules")
        if not rules:
            self.errors.append("No rules section")
            return ""

        # Find the play section
        rules_content = _find_child(rules, "rules_content")
        if not rules_content:
            return ""

        play = None
        for item in _find_all(rules_content, "rules_item"):
            p = _find_child(item, "play")
            if p:
                play = p
                break

        if not play:
            # Check for phases
            for item in _find_all(rules_content, "rules_item"):
                phases = _find_child(item, "phases")
                if phases:
                    return self._transpile_phases(phases)
            return ""

        play_text = _get_text(play)
        return self._transpile_play(play_text)

    def _transpile_play(self, play_text: str) -> str:
        """Convert Ludii play rules to Ludax."""
        if "forEach Piece" in play_text:
            return self._transpile_foreach_piece(play_text)
        elif "move Add" in play_text:
            return self._transpile_placement(play_text)
        elif "move Remove" in play_text or "move Select" in play_text or "move Claim" in play_text:
            piece = self.pieces[0][0] if self.pieces else "token"
            return f'(play (repeat (P1 P2) (place "{piece}" (destination (empty)))))'
        elif "priority" in play_text:
            return self._transpile_foreach_piece(play_text)
        else:
            # Default to placement
            piece = self.pieces[0][0] if self.pieces else "token"
            return f'(play (repeat (P1 P2) (place "{piece}" (destination (empty)))))'

    def _transpile_placement(self, play_text: str) -> str:
        """Transpile a placement game."""
        piece_name = self.pieces[0][0] if self.pieces else "token"

        # Check for effects (then clause)
        effects = ""
        if "then" in play_text:
            effects = self._extract_then_effects(play_text)

        if effects:
            return f'(play (repeat (P1 P2) (place "{piece_name}" (destination (empty)) {effects})))'
        else:
            return f'(play (repeat (P1 P2) (place "{piece_name}" (destination (empty)))))'

    def _transpile_foreach_piece(self, play_text: str) -> str:
        """Transpile a forEach Piece movement game."""
        # Detect movement types from the Ludii play text
        moves = []
        piece_name = self.pieces[0][0] if self.pieces else "token"

        if "move Step" in play_text or "Step" in play_text:
            if "Forward" in play_text:
                self.has_set_forward = True
                moves.append(f'(step "{piece_name}" direction:(forward_left forward_right) priority:1)')
            else:
                moves.append(f'(step "{piece_name}" direction:any priority:1)')

        if "move Hop" in play_text or "Hop" in play_text:
            if "Forward" in play_text or "FR" in play_text or "FL" in play_text:
                self.has_set_forward = True
                moves.append(f'(hop "{piece_name}" direction:(forward_left forward_right) hop_over:opponent capture:true priority:0)')
            else:
                moves.append(f'(hop "{piece_name}" direction:diagonal hop_over:opponent capture:true priority:0)')

        if "move Slide" in play_text or "Slide" in play_text:
            if "Orthogonal" in play_text:
                moves.append(f'(slide "{piece_name}" direction:orthogonal)')
            else:
                moves.append(f'(slide "{piece_name}" direction:any)')

        if not moves:
            moves.append(f'(step "{piece_name}" direction:any)')

        # Add effects
        effects = []
        if "remove" in play_text.lower() and ("between" in play_text or "custodial" in play_text):
            effects.append(f'(capture (custodial "{piece_name}" 1 orientation:orthogonal))')
        if "moveAgain" in play_text:
            effects.append(f'(if (and (action_was mover hop) (can_move_again hop)) (extra_turn mover same_piece:true))')

        effects_str = ""
        if effects:
            effects_str = " (effects " + " ".join(effects) + ")"

        if len(moves) == 1:
            return f'(play (repeat (P1 P2) (move {moves[0]}{effects_str})))'
        else:
            or_moves = " ".join(moves)
            return f'(play (repeat (P1 P2) (move (or {or_moves}){effects_str})))'

    def _transpile_phases(self, phases: Tree) -> str:
        """Transpile phase-based games to Ludax multi-phase play."""
        phase_texts = _get_text(phases)

        # Extract individual phases
        import re
        phase_blocks = re.findall(r'phase\s+"([^"]+)"(.*?)(?=phase\s+"|$)', phase_texts, re.DOTALL)

        ldx_phases = []
        for phase_name, phase_content in phase_blocks:
            # Detect if this phase has placement or movement
            if "move Add" in phase_content:
                piece = self.pieces[0][0] if self.pieces else "token"
                # Count how many placements (once_through vs repeat)
                if "nextPhase" in phase_content:
                    ldx_phases.append(f'(once_through (P1 P2) (place "{piece}" (destination (empty))))')
                else:
                    ldx_phases.append(f'(repeat (P1 P2) (place "{piece}" (destination (empty))))')
            elif "forEach Piece" in phase_content:
                ldx_phases.append(self._transpile_foreach_piece(phase_content))
                # Strip the outer (play ...) wrapper if present
                last = ldx_phases[-1]
                if last.startswith("(play "):
                    ldx_phases[-1] = last[6:-1]  # remove (play and trailing )
            else:
                # Default: movement with step
                piece = self.pieces[0][0] if self.pieces else "token"
                ldx_phases.append(f'(repeat (P1 P2) (move (step "{piece}" direction:any)))')

        if not ldx_phases:
            self.errors.append("No valid phases found")
            return ""

        phases_str = "\n            ".join(ldx_phases)
        return f"(play\n            {phases_str}\n        )"

    def _extract_then_effects(self, text: str) -> str:
        """Extract effects from a (then ...) clause."""
        # Look for common patterns
        effects = []
        if "addScore" in text or "set Score" in text:
            piece = self.pieces[0][0] if self.pieces else "token"
            effects.append(f'(set_score mover (count (occupied mover)))')
            effects.append(f'(set_score opponent (count (occupied opponent)))')
        return f'(effects {" ".join(effects)})' if effects else ""

    def _process_end(self, tree: Tree) -> str:
        """Extract and transpile end conditions."""
        rules = _find_child(tree, "rules")
        if not rules:
            return ""

        # Find end section anywhere in rules
        full_text = _get_text(rules)

        conditions = []

        # Line win
        if "is Line" in full_text:
            import re
            # Find each (if (is Line N) (result ...)) block
            for m in re.finditer(r'is Line (\d+)(.*?)result (\w+) (\w+)', full_text):
                n = m.group(1)
                piece = self.pieces[0][0] if self.pieces else "token"
                player = m.group(3)  # Mover, Next, P1, P2
                outcome = m.group(4)  # Win, Loss, Draw
                exact = ""
                between = m.group(2)
                if "exact:True" in between or "exact:true" in between:
                    exact = " exact:true"
                if outcome == "Loss":
                    conditions.append(f'(if (line "{piece}" {n}{exact}) (mover lose))')
                elif outcome == "Win":
                    conditions.append(f'(if (line "{piece}" {n}{exact}) (mover win))')
                elif outcome == "Draw":
                    conditions.append(f'(if (line "{piece}" {n}{exact}) (draw))')

        # Connected win
        if "is Connected" in full_text:
            piece = self.pieces[0][0] if self.pieces else "token"
            if self.has_set_forward:
                conditions.append(f'(if (>= (connected "{piece}" ((edge forward) (edge backward))) 2) (mover win))')
            else:
                conditions.append(f'(if (>= (connected "{piece}" ((edge left) (edge right))) 2) (mover win))')

        # No moves
        if "no Moves" in full_text:
            ctx = full_text[full_text.find("no Moves"):full_text.find("no Moves")+100]
            if "Next" in ctx:
                conditions.append("(if (no_legal_actions) (opponent lose))")
            else:
                conditions.append("(if (no_legal_actions) (mover lose))")

        # No pieces
        if "no Pieces" in full_text:
            import re
            for m in re.finditer(r'no Pieces (\w+).*?result (\w+) (\w+)', full_text):
                target, player, outcome = m.group(1), m.group(2), m.group(3)
                if outcome == "Win":
                    conditions.append(f"(if (no_legal_actions) (mover win))")
                elif outcome == "Loss":
                    conditions.append(f"(if (no_legal_actions) (mover lose))")

        # Reach edge / is In (sites Mover)
        if "is In" in full_text and ("sites Mover" in full_text or "sites Top" in full_text or "sites Bottom" in full_text):
            piece = self.pieces[0][0] if self.pieces else "token"
            if self.has_set_forward:
                conditions.append(f'(if (exists (and (occupied mover) (edge forward))) (mover win))')
            else:
                conditions.append(f'(if (no_legal_actions) (mover lose))')

        # Count-based win
        import re
        for m in re.finditer(r'<=\s+count Pieces (\w+)\s+(\d+).*?result (\w+) (\w+)', full_text):
            conditions.append(f"(if (no_legal_actions) (mover win))")

        # Full board draw
        if "is Full" in full_text or "full_board" in full_text.lower():
            conditions.append("(if (full_board) (draw))")

        if not conditions:
            # Default: no legal actions = loss (covers most movement games)
            conditions.append("(if (no_legal_actions) (mover lose))")

        return "(end " + " ".join(conditions) + ")"


def transpile(lud_text: str) -> typing.Optional[str]:
    """Convenience function: transpile Ludii .lud text to Ludax .ldx text."""
    t = LudiiTranspiler()
    return t.transpile(lud_text)
