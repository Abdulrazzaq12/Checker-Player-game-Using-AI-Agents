import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import random
from copy import deepcopy
from groq import Groq

# ----------------------------
# Game Logic for Checkers
# ----------------------------
class CheckersGame:
    def __init__(self):
        # Create an 8x8 board with '.' for empty squares.
        self.board = [['.' for _ in range(8)] for _ in range(8)]
        # Place pieces on dark squares.
        # Black ('b') pieces on the top three rows; Red ('r') pieces on the bottom three rows.
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 'b'
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 'r'

    def get_valid_moves(self, player):
        """
        Generate valid moves for the given player.
        If capturing moves are available, only those moves are returned.
        
        For our board:
         - Red ('r') pieces (at the bottom) move upward (decreasing row).
         - Black ('b') pieces (at the top) move downward (increasing row).
        """
        moves = []
        enemy = 'r' if player == 'b' else 'b'
        direction = -1 if player == 'r' else 1

        # Check for capturing moves first (jump moves).
        for i in range(8):
            for j in range(8):
                if self.board[i][j].lower() == player:
                    for dcol in [-1, 1]:
                        enemy_row = i + direction
                        enemy_col = j + dcol
                        landing_row = i + 2 * direction
                        landing_col = j + 2 * dcol
                        if (0 <= enemy_row < 8 and 0 <= enemy_col < 8 and
                            0 <= landing_row < 8 and 0 <= landing_col < 8):
                            if (self.board[enemy_row][enemy_col].lower() == enemy and 
                                self.board[landing_row][landing_col] == '.'):
                                moves.append(((i, j), (landing_row, landing_col)))
        if moves:
            return moves

        # If no capturing moves, check for simple slide moves.
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j].lower() == player:
                    for dcol in [-1, 1]:
                        new_i = i + direction
                        new_j = j + dcol
                        if 0 <= new_i < 8 and 0 <= new_j < 8 and self.board[new_i][new_j] == '.':
                            moves.append(((i, j), (new_i, new_j)))
        return moves

    def apply_move(self, move):
        start, end = move
        i, j = start
        new_i, new_j = end
        piece = self.board[i][j]
        self.board[i][j] = '.'
        self.board[new_i][new_j] = piece

        # If it's a capturing move (jump of two squares), remove the jumped enemy piece.
        if abs(new_i - i) == 2:
            capture_row = (i + new_i) // 2
            capture_col = (j + new_j) // 2
            self.board[capture_row][capture_col] = '.'

    def is_game_over(self):
        # Game is over if either player has no valid moves.
        return (not self.get_valid_moves('r')) or (not self.get_valid_moves('b'))

    def get_winner(self):
        if not self.get_valid_moves('r'):
            return 'b'
        elif not self.get_valid_moves('b'):
            return 'r'
        return None

# ----------------------------
# AI Agents
# ----------------------------
def evaluate(game, player):
    """Simple evaluation: difference in piece count."""
    red_count = sum(row.count('r') for row in game.board)
    black_count = sum(row.count('b') for row in game.board)
    return red_count - black_count if player == 'r' else black_count - red_count

def minimax(game, depth, player, maximizing_player):
    if depth == 0 or game.is_game_over():
        return evaluate(game, player), None

    current_player = player if maximizing_player else ('r' if player == 'b' else 'b')
    valid_moves = game.get_valid_moves(current_player)
    if not valid_moves:
        return evaluate(game, player), None

    best_move = None
    if maximizing_player:
        max_eval = -float('inf')
        for move in valid_moves:
            new_game = deepcopy(game)
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, player, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_game = deepcopy(game)
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, player, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move

def ai_minimax_move(game, player, depth=3):
    """Compute AI move using minimax search."""
    _, move = minimax(game, depth, player, True)
    return move

def ai_llm_move(game, player):
    """
    Use the Groq client with model "gemma2-9b-it" to suggest a move.
    The board is rendered as text and sent in a prompt.
    Expected LLM return format: "start_row,start_col -> end_row,end_col"
    """
    board_state = "\n".join(" ".join(row) for row in game.board)
    prompt = (
        f"Below is a checkers board state. The board is 8x8 where each cell is either '.', 'r', or 'b'.\n"
        f"Rows are numbered 0 to 7 from top to bottom, and columns 0 to 7 from left to right.\n\n"
        f"Board:\n{board_state}\n\n"
        f"You are playing as '{player}'. Provide your next move in the format:\n"
        f"start_row,start_col -> end_row,end_col\n"
        f"for a simple diagonal slide move or jump move (if a capture is available)."
    )
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
    #        messages=[{"role": "user", "content": prompt}],
    #        temperature=1,
    #        max_completion_tokens=1024,
    #        top_p=1,
    #        stream=True,
    #        stop=None,
            messages=[],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
        )
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""
        move_text = response_text.strip().splitlines()[0]
        parts = move_text.split("->")
        if len(parts) != 2:
            raise ValueError("Unexpected format from LLM output.")
        start = tuple(int(x.strip()) for x in parts[0].split(","))
        end = tuple(int(x.strip()) for x in parts[1].split(","))
        move = (start, end)
        if move not in game.get_valid_moves(player):
            print("LLM suggested an invalid move; choosing a random valid move instead.")
            move = random.choice(game.get_valid_moves(player))
        return move
    except Exception as e:
        print("LLM move error:", e)
        moves = game.get_valid_moves(player)
        return random.choice(moves) if moves else None

# ----------------------------
# GUI Application
# ----------------------------
class CheckersGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Checkers Game (GUI)")
        self.square_size = 80  # pixels per square
        self.canvas = tk.Canvas(master, width=8*self.square_size, height=8*self.square_size)
        self.canvas.pack()

        # Initialize game state.
        self.game = CheckersGame()
        self.selected_piece = None  # Coordinates of selected piece (if any)

        # Ask the player for configuration.
        self.human_player = simpledialog.askstring("Player Side", "Choose your side: 'r' for red or 'b' for black")
        if self.human_player not in ['r', 'b']:
            messagebox.showinfo("Invalid Choice", "Invalid choice; defaulting to red.")
            self.human_player = 'r'
        self.ai_player = 'b' if self.human_player == 'r' else 'r'

        ai_choice = simpledialog.askstring("AI Agent", "Choose AI agent: enter '1' for Minimax or '2' for Groq LLM")
        if ai_choice == '2':
            self.ai_agent = ai_llm_move
            messagebox.showinfo("AI Agent", "Using Groq LLM-based AI agent.")
        else:
            self.ai_agent = ai_minimax_move
            messagebox.showinfo("AI Agent", "Using Minimax-based AI agent.")

        # Red always starts.
        self.current_player = 'r'
        self.update_board()

        # Bind click event.
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # If the starting turn is AI, trigger its move.
        if self.current_player == self.ai_player:
            self.master.after(500, self.perform_ai_move)

    def update_board(self):
        self.canvas.delete("all")
        # Draw board squares.
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = "#D18B47" if (row + col) % 2 == 1 else "#FFCE9E"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        # Highlight selected piece.
        if self.selected_piece:
            row, col = self.selected_piece
            x1 = col * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3)

        # Draw pieces.
        for row in range(8):
            for col in range(8):
                piece = self.game.board[row][col]
                if piece != '.':
                    x_center = col * self.square_size + self.square_size / 2
                    y_center = row * self.square_size + self.square_size / 2
                    radius = self.square_size * 0.35
                    fill_color = "red" if piece.lower() == 'r' else "black"
                    self.canvas.create_oval(x_center - radius, y_center - radius,
                                            x_center + radius, y_center + radius,
                                            fill=fill_color)

    def on_canvas_click(self, event):
        # Debug: print click coordinates and computed row, col.
        print(f"Canvas clicked at pixel: ({event.x}, {event.y})")
        col = event.x // self.square_size
        row = event.y // self.square_size
        print(f"Interpreted board position: ({row}, {col})")

        # Only allow input on human's turn.
        if self.current_player != self.human_player:
            print("Not human's turn.")
            return

        # If no piece selected, try selecting a human piece.
        if self.selected_piece is None:
            if self.game.board[row][col].lower() == self.human_player:
                print(f"Selected piece at: ({row}, {col})")
                self.selected_piece = (row, col)
                self.update_board()
        else:
            # Try to make a move from the selected piece to the clicked square.
            move = (self.selected_piece, (row, col))
            valid_moves = self.game.get_valid_moves(self.human_player)
            print("Attempting move:", move)
            if move in valid_moves:
                print("Move is valid.")
                self.game.apply_move(move)
                self.selected_piece = None
                self.update_board()
                self.after_human_move()
            else:
                # If the clicked square is another of the human's piece, change selection.
                if self.game.board[row][col].lower() == self.human_player:
                    print("Changing selection to new piece.")
                    self.selected_piece = (row, col)
                    self.update_board()
                else:
                    # Invalid move; clear selection.
                    print("Invalid move. Clearing selection.")
                    self.selected_piece = None
                    self.update_board()

    def after_human_move(self):
        if self.game.is_game_over():
            self.end_game()
            return

        # Switch turn.
        self.current_player = self.ai_player if self.current_player == self.human_player else self.human_player

        # If next turn is AI, perform AI move.
        if self.current_player == self.ai_player:
            self.master.after(500, self.perform_ai_move)

    def perform_ai_move(self):
        def ai_move_thread():
            valid_moves = self.game.get_valid_moves(self.ai_player)
            if not valid_moves:
                return
            if self.ai_agent == ai_minimax_move:
                chosen_move = ai_minimax_move(self.game, self.ai_player)
            else:
                chosen_move = ai_llm_move(self.game, self.ai_player)
            self.master.after(0, self.apply_ai_move, chosen_move)
        threading.Thread(target=ai_move_thread).start()

    def apply_ai_move(self, move):
        if move:
            print("AI move:", move)
            self.game.apply_move(move)
            self.update_board()
        if self.game.is_game_over():
            self.end_game()
        else:
            self.current_player = self.human_player if self.current_player == self.ai_player else self.ai_player

    def end_game(self):
        winner = self.game.get_winner()
        if winner:
            message = "Congratulations, you win!" if winner == self.human_player else "AI wins!"
        else:
            message = "It's a draw!"
        messagebox.showinfo("Game Over", message)
        self.master.quit()

# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = CheckersGUI(root)
    root.mainloop()
