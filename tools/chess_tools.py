import json
import os
import re
import shutil
from pathlib import Path

from smolagents import Tool

from tools.llm import ask_vision

LOCAL_STOCKFISH = Path(__file__).parent.parent / "cache" / "bin" / "stockfish"

BOARD_PROMPT = """This image shows a chess board. Every square has a small red number (1-64) in
its top-right corner: 1-8 is the top row of the image from left to right, 9-16 the second row, and so on.
1. Work out the orientation from the coordinate labels (a-h, 1-8) on the board edges if present:
   "white_bottom" if rank 1 is at the bottom of the image, "black_bottom" if rank 1 is at the top.
   If there are no labels, assume white_bottom unless the pieces clearly show otherwise.
2. For EVERY numbered square that has a piece on it, give the piece. Look at the square's own number,
   do not count squares.
Reply with JSON only, in this exact shape:
{"orientation": "white_bottom" or "black_bottom", "squares": {"7": "K", "60": "q", ...}}
Use uppercase letters for white pieces and lowercase for black (K Q R B N P / k q r b n p).
Leave out empty squares. Check every row before answering."""


def label_squares(path: str) -> bytes:
    """Draws the square numbers 1-64 on the board image (assumed to be just the board)."""
    import cv2

    image = cv2.imread(path)
    height, width = image.shape[:2]
    cell_h, cell_w = height / 8, width / 8
    scale = max(cell_w / 160, 0.35)
    for index in range(64):
        row, col = divmod(index, 8)
        x, y = int((col + 1) * cell_w) - int(cell_w * 0.32), int(row * cell_h + cell_h * 0.2)
        cv2.putText(image, str(index + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
    return cv2.imencode(".png", image)[1].tobytes()


def square_name(number: int, orientation: str) -> str:
    row, col = divmod(number - 1, 8)
    if orientation == "black_bottom":
        file, rank = 7 - col, row
    else:
        file, rank = col, 7 - row
    return "abcdefgh"[file] + str(rank + 1)


def stockfish_path() -> str | None:
    path = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish")
    if path:
        return path
    return str(LOCAL_STOCKFISH) if LOCAL_STOCKFISH.exists() else None


class ChessBoardFromImageTool(Tool):
    name = "chess_board_from_image"
    description = (
        "Reads a chess position from an image of a board (the image should show just the board) and "
        "returns its FEN plus a text diagram of the board. "
        "Check the diagram against the image (e.g. with analyze_image) if anything looks wrong, "
        "then pass the FEN to chess_best_move."
    )
    inputs = {
        "path": {"type": "string", "description": "Path of the local image file."},
        "side_to_move": {"type": "string", "description": "'white' or 'black'."},
    }
    output_type = "string"

    def forward(self, path: str, side_to_move: str) -> str:
        import chess

        try:
            reply = ask_vision(BOARD_PROMPT, label_squares(path), "image/png")
            data = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
            orientation = data.get("orientation", "white_bottom")
        except Exception as e:
            return f"Could not read the board: {e}"

        board = chess.Board(None)
        for number, symbol in data.get("squares", {}).items():
            try:
                square = chess.parse_square(square_name(int(number), orientation))
                board.set_piece_at(square, chess.Piece.from_symbol(symbol))
            except ValueError:
                continue
        board.turn = chess.WHITE if side_to_move.lower().startswith("w") else chess.BLACK
        # Castling rights can't be seen in an image; assume none.
        board.castling_rights = 0
        status = "valid" if board.is_valid() else f"INVALID ({board.status()!r}) - the board was probably misread"
        return (
            f"FEN: {board.fen()}\nPosition is {status}.\n"
            f"Orientation read from image: {orientation}\n\n{board.unicode(borders=True)}"
        )


class ChessBestMoveTool(Tool):
    name = "chess_best_move"
    description = (
        "Analyses a chess position (FEN) with the Stockfish engine and returns the best move in "
        "standard algebraic notation (e.g. 'Rd5', 'Qxe7+'), the evaluation and the main line."
    )
    inputs = {
        "fen": {"type": "string", "description": "The position in FEN notation."},
        "think_seconds": {"type": "number", "description": "Engine time (default 5).", "nullable": True},
    }
    output_type = "string"

    def forward(self, fen: str, think_seconds: float | None = None) -> str:
        import chess
        import chess.engine

        engine_path = stockfish_path()
        if not engine_path:
            return "Stockfish is not installed. Set STOCKFISH_PATH or install the 'stockfish' package."
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return f"Invalid FEN: {e}"
        if not board.is_valid():
            return f"Illegal position ({board.status()!r}); re-check the board."

        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(time=think_seconds or 5), multipv=3)
        lines = []
        for rank, entry in enumerate(info, start=1):
            pv = entry.get("pv", [])
            if not pv:
                continue
            score = entry["score"].pov(board.turn)
            score_text = f"mate in {score.mate()}" if score.is_mate() else f"{score.score() / 100:+.2f} pawns"
            line = board.copy().variation_san(pv[:8])
            lines.append(f"{rank}. {board.san(pv[0])}  ({score_text} for the side to move)  line: {line}")
        return "Best moves:\n" + "\n".join(lines)
