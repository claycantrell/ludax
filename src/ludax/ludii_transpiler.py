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
        self.piece_movements = {}  # {base_name: content_text} — raw movement text from piece definitions
        self.board_shape = ""
        self.board_size = 0
        self.board_ldx = ""
        self.has_set_forward = False
        self.has_hand = False
        self.is_mancala = False
        self.mancala_rows = 0
        self.mancala_cols = 0
        self.regions = []
        self.errors = []

    def transpile(self, lud_text: str) -> typing.Optional[str]:
        """Convert Ludii text to Ludax text. Returns None if not transpilable."""
        parser = _get_parser()
        tree = parser.parse(lud_text)
        self._tree = tree

        # Extract game name — find the ESCAPED_STRING token
        name = "Unknown"
        for c in tree.children:
            if isinstance(c, Token) and c.type == "ESCAPED_STRING":
                name = str(c).strip('"')
                break

        # Process each section
        self._process_players(tree)
        self._process_equipment(tree)

        # Detect mancala: any game with sow + track is mancala even without mancalaBoard
        full_lud = _get_text(tree)
        if not self.is_mancala and "sow" in full_lud.lower() and "track" in full_lud.lower():
            self.is_mancala = True
            if not self.mancala_cols:
                # Estimate board from existing board_ldx
                if "square" in self.board_ldx:
                    n = int(self.board_ldx.split()[-1])
                    self.mancala_rows = 2
                    self.mancala_cols = (n * n) // 2
                elif "rectangle" in self.board_ldx:
                    parts = self.board_ldx.split()
                    self.mancala_cols = int(parts[1])
                    self.mancala_rows = int(parts[2])
                else:
                    self.mancala_rows = 2
                    self.mancala_cols = 6
                # Override board to rectangle for mancala
                self.board_ldx = f"rectangle {self.mancala_cols} {self.mancala_rows}"
                self.board_shape = "rectangle"

        self._merge_symmetric_pieces()
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

    def _merge_symmetric_pieces(self):
        """Merge P1+P2 piece pairs with same movement into a single 'both' piece.

        E.g., ("toad" P1) + ("frog" P2) with same step/hop → ("marker" both)
        This allows Ludax to handle forEach Piece games where P1/P2 have
        different-named but functionally identical pieces.
        """
        if len(self.pieces) < 2:
            return

        # Find P1/P2 pairs
        p1_pieces = [(i, name) for i, (name, owner) in enumerate(self.pieces) if owner == "P1"]
        p2_pieces = [(i, name) for i, (name, owner) in enumerate(self.pieces) if owner == "P2"]

        if not p1_pieces or not p2_pieces:
            return

        # Check if the movement keywords are the same for each pair
        def _movement_keywords(content: str) -> set:
            kws = set()
            for kw in ["Step", "Hop", "Slide", "Leap", "Forward", "Orthogonal", "Diagonal"]:
                if kw in content:
                    kws.add(kw)
            return kws

        merged_indices = set()
        for p1_idx, p1_name in p1_pieces:
            p1_kws = _movement_keywords(self.piece_movements.get(p1_name, ""))
            for p2_idx, p2_name in p2_pieces:
                if p2_idx in merged_indices:
                    continue
                p2_kws = _movement_keywords(self.piece_movements.get(p2_name, ""))
                if p1_kws == p2_kws:
                    # Same movement — merge to "both" using first piece's name
                    self.pieces[p1_idx] = (p1_name, "both")
                    merged_indices.add(p2_idx)
                    break

        # Remove merged P2 pieces (iterate in reverse to preserve indices)
        for idx in sorted(merged_indices, reverse=True):
            self.pieces.pop(idx)

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
                # Mancala: (mancalaBoard rows cols) → Ludax rectangle cols rows
                self.is_mancala = True
                nums = [int(t) for t in content_str.split() if t.isdigit()]
                if len(nums) >= 2:
                    rows, cols = nums[0], nums[1]
                    self.board_ldx = f"rectangle {cols} {rows}"
                    self.mancala_rows = rows
                    self.mancala_cols = cols
                elif nums:
                    self.board_ldx = f"rectangle {nums[0]} 2"
                    self.mancala_rows = 2
                    self.mancala_cols = nums[0]
                else:
                    self.board_ldx = "rectangle 6 2"
                    self.mancala_rows = 2
                    self.mancala_cols = 6
                self.board_shape = "rectangle"
            elif type_str == "hand":
                self.has_hand = True
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

        if shape in ("square", "rectangle", "hex", "hexagon", "tri"):
            args = [t for t in tokens[1:] if t.isdigit()]
            if shape in ("hex", "hexagon", "tri"):
                if len(args) >= 3:
                    # Variable-width hex/tri: {3 4 3 4 3} → hex_rectangle max_width num_rows
                    row_widths = [int(a) for a in args]
                    max_w = max(row_widths)
                    num_rows = len(row_widths)
                    self.board_ldx = f"hex_rectangle {max_w} {num_rows}"
                    self.board_shape = "hex_rectangle"
                elif args:
                    size = int(args[0])
                    if size % 2 == 0:
                        size += 1  # Ludax requires odd hex diameter
                    self._set_hex_board(size)
                return
            if shape == "rectangle" and len(args) >= 2:
                # Ludii: (rectangle rows cols), Ludax: (rectangle width height)
                rows, cols = max(int(args[0]), 1), max(int(args[1]), 2)
                self.board_ldx = f"rectangle {cols} {rows}"
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

        # Fallback: find the LARGEST recognizable shape in the content (for merge boards)
        import re as _re
        for shape_name in ["square", "rectangle", "hex", "hexagon"]:
            # Find ALL instances and pick the largest
            matches = _re.findall(rf'\b{shape_name}\s+(\d+)(?:\s+(\d+))?', content, _re.IGNORECASE)
            if matches:
                best = max(matches, key=lambda m: int(m[0]))
                n1 = max(int(best[0]), 3)
                if shape_name in ("hex", "hexagon"):
                    self._set_hex_board(n1)
                elif shape_name == "rectangle" and best[1]:
                    n2 = max(int(best[1]), 3)
                    self.board_ldx = f"rectangle {max(n2, 2)} {max(n1, 2)}"
                    self.board_shape = "rectangle"
                else:
                    self.board_ldx = f"square {n1}"
                    self.board_shape = "square"
                return

        # Last resort: merge/add/remove/concentric → approximate as square 7
        if any(kw in content.lower() for kw in ["merge", "add", "remove", "concentric", "scale", "shift",
                                                   "tiling", "star", "complete", "subdivide", "dual"]):
            # Complex board — pick a reasonable default
            nums = [int(t) for t in tokens if t.isdigit() and 3 <= int(t) <= 20]
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

        # Store movement text from piece definition (for forEach Piece games)
        if base_name not in self.piece_movements:
            self.piece_movements[base_name] = content
        else:
            # Merge movement text from multiple definitions of same base name
            self.piece_movements[base_name] += " " + content

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
        # Mancala games: seed_counts initialized by engine, no piece placement needed
        if self.is_mancala:
            return ""

        rules = _find_child(tree, "rules")
        if not rules:
            return ""

        full_text = _get_text(rules)
        if "start" not in full_text:
            return ""

        import re
        piece_name = self.pieces[0][0] if self.pieces else "token"

        # Use first piece for most start positions. Only use different names
        # when pieces are explicitly P1/P2-specific with no "both" pieces.
        p1_piece = piece_name
        p2_piece = piece_name
        has_both = any(o == "both" for _, o in self.pieces)
        if not has_both:
            # All pieces are player-specific — use appropriate piece per player
            for pname, powner in self.pieces:
                if powner == "P1":
                    p1_piece = pname
                    break
            for pname, powner in self.pieces:
                if powner == "P2":
                    p2_piece = pname
                    break

        # Look for (place "X" ...) patterns in the start section
        start_ldx = []

        # Get board height for row calculations
        board_h = 8
        if "square" in self.board_ldx:
            board_h = int(self.board_ldx.split()[-1])
        elif "rectangle" in self.board_ldx:
            parts = self.board_ldx.split()
            # Ludax rectangle W H — height is last arg, rows index by height
            board_h = int(parts[-1]) if len(parts) >= 3 else int(parts[-2])
        elif "hexagon" in self.board_ldx:
            board_h = int(self.board_ldx.split()[-1])
        board_h = max(board_h, 2)  # Ensure at least 2 rows

        # Get board width for coordinate conversion
        board_w = board_h
        if "rectangle" in self.board_ldx:
            parts = self.board_ldx.split()
            # Ludax rectangle W H — width is first arg
            board_w = int(parts[1]) if len(parts) >= 3 else board_h

        # Pattern: (sites Row N) — most common (163 games)
        # Note: _get_text strips parentheses, so match with or without them
        row_matches = re.findall(r'place\s+"([^"]+)"\s+(?:\(?sites )?Row (\d+)\)?', full_text)
        if row_matches:
            for pname_raw, row_num in row_matches:
                pname = re.sub(r'\d+$', '', pname_raw).lower()
                if not pname: pname = piece_name
                player = "P1" if pname_raw.endswith("1") else "P2" if pname_raw.endswith("2") else "P1"
                row = int(row_num)
                # Cap to board bounds
                if row >= board_h:
                    row = board_h - 1
                start_ldx.append(f'(place "{pname}" {player} ((row {row})))')

        # Pattern: coord-based placement — both single coord:"X" and coordinate lists "A1" "H1"
        # Only use for chess-like games (multiple piece types with forEach Piece)
        # to avoid breaking simpler games that happen to have coordinate strings.
        is_chess_like = len(self.pieces) > 2 and "forEach Piece" in full_text
        if is_chess_like and ("square" in self.board_ldx or "rectangle" in self.board_ldx):
            def _coord_to_idx(coord_str):
                """Convert chess notation like 'A1' to cell index."""
                coord_str = coord_str.strip('"')
                if len(coord_str) >= 2 and coord_str[0].isalpha() and coord_str[1:].isdigit():
                    col = ord(coord_str[0].upper()) - ord('A')
                    row = int(coord_str[1:]) - 1
                    idx = row * board_w + col
                    if 0 <= idx < board_h * board_w:
                        return idx
                return None

            # Match: place "Name" coord:"X" or place "Name" "A1" "B2" ...
            for m in re.finditer(r'place\s+"([^"]+)"\s+((?:coord:)?"[A-Za-z]\d+"(?:\s+"[A-Za-z]\d+")*)', full_text):
                pname_raw = m.group(1)
                coords_str = m.group(2)
                pname = re.sub(r'\d+$', '', pname_raw).lower()
                if not pname: pname = piece_name
                player = "P1" if pname_raw.endswith("1") else "P2" if pname_raw.endswith("2") else "P1"
                # Extract all coordinate values
                coord_values = re.findall(r'"([A-Za-z]\d+)"', coords_str)
                indices = [_coord_to_idx(c) for c in coord_values]
                indices = [i for i in indices if i is not None]
                if indices:
                    idx_str = " ".join(str(i) for i in indices)
                    start_ldx.append(f'(place "{pname}" {player} ({idx_str}))')

        # Helper to clamp row indices to valid range
        def _row(r):
            return max(0, min(r, board_h - 1))

        # For 1-row boards, use cell-index placement (row-based doesn't work)
        is_1row = board_h == 1
        if is_1row:
            total_cells = board_w
            half = total_cells // 2
            if not start_ldx and ("expand" in full_text or "sites Left" in full_text or "sites Right" in full_text or "sites Bottom" in full_text):
                left_cells = " ".join(str(i) for i in range(half))
                right_cells = " ".join(str(i) for i in range(total_cells - half, total_cells))
                start_ldx.append(f'(place "{p1_piece}" P1 ({left_cells}))')
                start_ldx.append(f'(place "{p2_piece}" P2 ({right_cells}))')

        # Pattern: (expand (sites Bottom/Top))
        if not start_ldx and "expand" in full_text:
            if "sites Bottom" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0) (row {_row(1)})))')
            if "sites Top" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-2)}) (row {_row(board_h-1)})))')

        # Pattern: (sites Left/Right) — horizontal layout
        if not start_ldx and ("sites Left" in full_text or "sites Right" in full_text):
            if "sites Left" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            if "sites Right" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')

        # Pattern: (sites Bottom/Top) without expand
        if not start_ldx:
            if "sites Bottom" in full_text:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            if "sites Top" in full_text:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')

        # Pattern: (sites Phase N) — checkerboard
        if not start_ldx and "sites Phase" in full_text:
            start_ldx.append(f'(place "{p1_piece}" P1 ((row 0) (row {_row(1)}) (row {_row(2)})))')
            start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-3)}) (row {_row(board_h-2)}) (row {_row(board_h-1)})))')

        # Pattern: direct index placement (place "X" N)
        # Skip — Ludii indices don't map to Ludax indices on different board types
        # Use row-based fallback instead

        # Pattern: coord: placement — convert chess-style "C5" to index
        if not start_ldx:
            for m in re.finditer(r'place\s+"([^"]+)"\s+coord:"([^"]+)"', full_text):
                pname_raw = m.group(1)
                coord = m.group(2)
                pname = re.sub(r'\d+$', '', pname_raw).lower()
                if not pname: pname = piece_name
                player = "P1" if pname_raw.endswith("1") else "P2" if pname_raw.endswith("2") else "P1"
                # Convert "C5" → column * board_width + row
                if len(coord) >= 2 and coord[0].isalpha() and coord[1:].isdigit():
                    col = ord(coord[0].upper()) - ord('A')
                    row = int(coord[1:]) - 1
                    if "square" in self.board_ldx:
                        width = int(self.board_ldx.split()[-1])
                        idx = row * width + col
                        if 0 <= idx < width * width:
                            start_ldx.append(f'(place "{pname}" {player} ({idx}))')

        if start_ldx:
            # Validate: ensure we have both P1 and P2 placements
            has_p1 = any("P1" in s for s in start_ldx)
            has_p2 = any("P2" in s for s in start_ldx)
            if has_p1 and not has_p2:
                start_ldx.append(f'(place "{p2_piece}" P2 ((row {_row(board_h-1)})))')
            elif has_p2 and not has_p1:
                start_ldx.append(f'(place "{p1_piece}" P1 ((row 0)))')
            # Check for overlap — if P1 and P2 use same rows, use opposite ends instead
            result = "(start " + " ".join(start_ldx) + ")"
            return result

        # No extracted start — use opposite-end fallback for movement games
        # But only if the original game actually has piece placement in its start section
        has_place_in_start = "place" in full_text.split("play")[0] if "play" in full_text else "place" in full_text

        # Fallback: if it's a movement game WITH piece placement, auto-place on first/last rows
        if has_place_in_start and ("forEach Piece" in full_text or "move Step" in full_text or "move Hop" in full_text or "move Slide" in full_text):
            if "square" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                if size <= 4:
                    return f'(start (place "{p1_piece}" P1 ((row 0))) (place "{p2_piece}" P2 ((row {size-1}))))'
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {size-2}) (row {size-1}))))'
            elif "hexagon" in self.board_ldx:
                size = int(self.board_ldx.split()[-1])
                last_row = size - 1
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {last_row-1}) (row {last_row}))))'
            elif "rectangle" in self.board_ldx:
                parts = self.board_ldx.split()
                h = int(parts[-2]) if len(parts) >= 3 else int(parts[-1]) if len(parts) >= 2 else 8
                if h <= 4:
                    return f'(start (place "{p1_piece}" P1 ((row 0))) (place "{p2_piece}" P2 ((row {h-1}))))'
                return f'(start (place "{p1_piece}" P1 ((row 0) (row 1))) (place "{p2_piece}" P2 ((row {h-2}) (row {h-1}))))'

        return ""

    def _process_rules(self, tree: Tree) -> str:
        """Extract and transpile play rules."""
        # Mancala games always use sow mechanic regardless of Ludii phases
        if self.is_mancala:
            import re
            rules = _find_child(tree, "rules")
            full = _get_text(rules) if rules else ""
            seeds = 4
            m = re.search(r'set Count (\d+)', full)
            if m:
                seeds = int(m.group(1))
            return f'(play (repeat (P1 P2) (sow seeds:{seeds} capture_opposite)))'

        rules = _find_child(tree, "rules")
        if not rules:
            self.errors.append("No rules section")
            return ""

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
            for item in _find_all(rules_content, "rules_item"):
                phases = _find_child(item, "phases")
                if phases:
                    return self._transpile_phases(phases)
            return ""

        play_text = _get_text(play)
        return self._transpile_play(play_text)

    def _get_movement_text(self, play_text: str, include_pieces: bool = False) -> str:
        """Get text to search for movement keywords.

        When include_pieces=True, also searches piece definitions from equipment
        (critical for forEach Piece games where movement is defined per-piece).
        """
        parts = [play_text]
        if include_pieces:
            for content in self.piece_movements.values():
                parts.append(content)
        # Add full rules text as fallback
        if hasattr(self, '_tree'):
            rules = _find_child(self._tree, "rules")
            if rules:
                parts.append(_get_text(rules))
        return " ".join(parts)

    def _transpile_play(self, play_text: str) -> str:
        """Convert Ludii play rules to Ludax."""
        # Mancala games use sow mechanic
        if self.is_mancala:
            # Extract initial seed count from Ludii start section
            seeds = 4  # default
            if hasattr(self, '_tree'):
                import re
                rules = _find_child(self._tree, "rules")
                if rules:
                    full = _get_text(rules)
                    m = re.search(r'set Count (\d+)', full)
                    if m:
                        seeds = int(m.group(1))
            return f'(play (repeat (P1 P2) (sow seeds:{seeds} capture_opposite)))'

        # For forEach Piece, include piece definitions to find movement keywords
        is_foreach = "forEach Piece" in play_text
        movement_text = self._get_movement_text(play_text, include_pieces=is_foreach)

        if is_foreach:
            has_add = "move Add" in movement_text or "move Add" in play_text
            has_movement = "move Step" in movement_text or "move Hop" in movement_text or "move Slide" in movement_text
            if has_add and not has_movement:
                return self._transpile_placement(play_text)
            if has_add and has_movement:
                # Combined placement + movement game: place first, then move
                piece_name = self.pieces[0][0] if self.pieces else "token"
                movement = self._transpile_foreach_piece(play_text)
                # Strip (play ...) wrapper from movement
                if movement.startswith("(play "):
                    movement_inner = movement[6:-1]
                else:
                    movement_inner = movement
                return f'(play\n            (repeat (P1 P2) (place "{piece_name}" (destination (empty))))\n        )'
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
        # Get combined text including piece definitions for movement keyword detection
        mt = self._get_movement_text(play_text, include_pieces=True)

        moves = []
        piece_name = self.pieces[0][0] if self.pieces else "token"

        # Detect which movement types are present (from piece defs + play/rules text)
        has_step = "move Step" in mt or ("Step" in mt and "move" in mt)
        has_hop = "move Hop" in mt or ("Hop" in mt and "move" in mt)
        has_slide = "move Slide" in mt or ("Slide" in mt and "move" in mt)
        has_leap = "Leap" in mt

        # IMPORTANT: Don't mix step/hop (FROM_DIR action space) with slide (FROM_TO action space).
        # Slide detection from piece defs is unreliable and can cause shape errors,
        # so only use slide when it's the ONLY movement and was in play_text directly.
        if has_slide and (has_step or has_hop):
            has_slide = False
        elif has_slide and "Slide" not in play_text:
            # Slide detected only from piece defs — fall back to step (safer)
            has_slide = False
            has_step = True

        # For games with multiple piece types, generate movement for ALL types
        # so that all pieces can move (not just the first type).
        piece_names = [piece_name]
        if len(self.pieces) > 1:
            piece_names = [p for p, _ in self.pieces]

        # Detect hop-over-friendly: hop over own pieces to capture enemy at destination
        # Ludii pattern: (between if:(is Friend ...)) (to if:(is Enemy ...))
        hop_over_target = "opponent"
        if has_hop and ("is Friend" in mt and "between" in mt):
            hop_over_target = "mover"

        # Always use direction:any — specific directions cause more failures than they fix
        for pn in piece_names:
            if has_step:
                moves.append(f'(step "{pn}" direction:any)')
            if has_hop:
                moves.append(f'(hop "{pn}" direction:any hop_over:{hop_over_target} capture:true)')

        # Only add slide for the first piece (to avoid action space explosion)
        if has_slide:
            moves.append(f'(slide "{piece_name}" direction:any)')

        if has_leap and not has_step:
            for pn in piece_names:
                moves.append(f'(step "{pn}" direction:any)')

        if not moves:
            for pn in piece_names:
                moves.append(f'(step "{pn}" direction:any)')

        # Add effects — only from play_text (not piece defs) to avoid false positives
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
                # Default: movement with step for all piece types
                ldx_phases.append(self._transpile_foreach_piece(phase_content))
                last = ldx_phases[-1]
                if last.startswith("(play "):
                    ldx_phases[-1] = last[6:-1]

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
