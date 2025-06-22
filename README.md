
# 🧠 Checkers Player Game with AI (Minimax & LLM-Groq)
A Python-based Checkers game with a user-friendly GUI and intelligent AI opponents. Play against either a traditional Minimax-based agent or a powerful Groq LLM-based AI.

### 🎮 Features
 ✅ Interactive GUI built using Tkinter.

### 🤖 Two AI Agent Options:

Minimax: Classic game-tree search algorithm.

Groq LLM: Uses a large language model to suggest intelligent moves.

### 🧠 Smart move generation, capturing, and end-game detection.

### 👤 Choose your side ('r' for Red, 'b' for Black).

### 🎨 Board updates and piece highlighting.

### 🧠 AI Agent Overview
AI Agent	Description
Minimax	Implements depth-limited minimax search with simple evaluation heuristics.
Groq LLM	Calls a large language model via Groq API to suggest a move in real-time.

###🚀 Getting Started
###📦 Requirements
Python 3.x

Install dependencies:
pip install groq

### ▶️ How to Run
Clone the repository:
git clone https://github.com/yourusername/checkers-ai-game.git
cd checkers-ai-game
Run the game:

python "checkers Player game.py"
When prompted:

Choose your side ('r' or 'b')

Choose AI agent:

1 for Minimax

2 for Groq LLM

###🔐 Groq API Key Setup
To use the Groq LLM agent:

Replace this line in the code with your actual API key:

python
Copy
Edit
client = Groq(api_key="your_api_key_here")
Important: Never share or hardcode your API key in public code.

###🧪 Future Improvements
###🎯 Add kinging support for crowned pieces.

###🌐 Online multiplayer mode.

###📊 Track and display move history and scores.

###🗂 Export game replays for training/testing AI.

###🧠 Integrate more LLM models like OpenAI, Claude, etc.

###📁 Project Structure

📦 checkers-ai-game/
 ┣ 📜 checkers Player game.py   ← Main game logic and GUI
 ┗ 📜 README.md                  ← Project overview


### 🤝 Contribution
Feel free to fork, improve, or suggest features via issues or PRs. Collaboration is welcome!

### 📄 License
This project is licensed under the MIT License.


