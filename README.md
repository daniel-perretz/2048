# 2048 Game with AI Agents
This project implements the popular game 2048 along with several AI agents designed to play the game. It serves as a platform for exploring and comparing different AI decision-making algorithms in a complex game environment.
Key Components
Game Engine (game.py, game_state.py)

# Implements the core 2048 game logic
Manages game state, including the board, score, and game mechanics
Handles legal moves, state transitions, and game termination conditions

# AI Agents (multi_agents.py)

Reflex Agent: Makes decisions based on immediate state evaluation

Minimax Agent: Utilizes the Minimax algorithm for decision making

Alpha-Beta Agent: Implements Minimax with alpha-beta pruning for improved efficiency

Expectimax Agent: Uses the Expectimax algorithm to handle probabilistic outcomes

# Evaluation Functions

Includes various heuristics to assess game states
Considers factors such as:

Tile positions and values

Board monotonicity

Available moves

Empty tiles

Overall score



# Game Runner (2048.py)

Serves as the main entry point for running games
Supports different display options (graphical or summary)
Allows configuration of game parameters via command-line arguments

# Display Options (displays.py)

Includes a SummaryDisplay for non-graphical output
Tracks and reports game statistics like scores, highest tiles, and durations

# Features

Flexible game configuration (board size, number of games, choice of AI agent)
Command-line interface for easy parameter adjustment
Support for both human players and AI agents
Comprehensive evaluation functions for advanced game state assessment

