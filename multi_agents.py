import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        successor_game_state = current_game_state.generate_successor(
            action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        evaluation = (
                score+ 1.5 * len(
            successor_game_state.get_agent_legal_actions())

        )

        return evaluation


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        _, best_action = self.minimax(game_state, self.depth, 0)
        return best_action

    def minimax(self, state, depth, agent_index):
        if depth == 0 or not state.get_legal_actions(agent_index):
            return self.evaluation_function(state), None

        if agent_index == 0:
            return self.max_value(state, depth, agent_index)
        else:
            return self.min_value(state, depth, agent_index)

    def max_value(self, state, depth, agent_index):
        max_eval = float('-inf')
        best_action = None
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            eval, _ = self.minimax(successor, depth , 1)
            if eval > max_eval:
                max_eval = eval
                best_action = action
        return max_eval, best_action

    def min_value(self, state, depth, agent_index):
        min_eval = float('inf')
        best_action = None
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            eval, _ = self.minimax(successor, depth - 1, 0)
            if eval < min_eval:
                min_eval = eval
                best_action = action
        return min_eval, best_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        alpha = float('-inf')
        beta = float('inf')
        _, best_action = self.minimax(game_state, self.depth, 0, alpha, beta)
        return best_action

    def minimax(self, state, depth, agent_index, alpha, beta):
        if depth == 0 or not state.get_legal_actions(agent_index):
            return self.evaluation_function(state), None

        if agent_index == 0:
            return self.max_value(state, depth, agent_index, alpha, beta)
        else:
            return self.min_value(state, depth, agent_index, alpha, beta)

    def max_value(self, state, depth, agent_index, alpha, beta):
        max_eval = float('-inf')
        best_action = None

        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            eval, _ = self.minimax(successor, depth , 1, alpha,
                                   beta)
            if eval > max_eval:
                max_eval = eval
                best_action = action
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_action

    def min_value(self, state, depth, agent_index, alpha, beta):
        min_eval = float('inf')
        best_action = None

        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            eval, _ = self.minimax(successor, depth - 1, 0, alpha,
                                   beta)
            if eval < min_eval:
                min_eval = eval
                best_action = action
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        _, best_action = self.expectimax(game_state, self.depth, 0)
        return best_action


    def expectimax(self, state, depth, agent_index):
        if depth == 0 or not state.get_legal_actions(agent_index):
            return self.evaluation_function(state), None

        if agent_index == 0:
            return self.max_value(state, depth, agent_index)
        else:
            return self.exp_value(state, depth, agent_index)


    def max_value(self, state, depth, agent_index):
        max_value = float('-inf')
        best_action = None
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            value, _ = self.expectimax(successor, depth , 1)
            if value > max_value:
                max_value = value
                best_action = action
        return max_value, best_action


    def exp_value(self, state, depth, agent_index):
        expected_value = 0
        legal_actions = state.get_legal_actions(agent_index)
        num_actions = len(legal_actions)
        for action in legal_actions:
            successor = state.generate_successor(agent_index, action)
            value, _ = self.expectimax(successor, depth - 1, 0)
            expected_value += value / num_actions
        return expected_value, None



def check_corners_max_tile(board):
    rows, cols = board.shape
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    max_value = np.max(board)
    for corner in corners:
        if board[corner] == max_value:
            return 10
    return -10


def monotonicity(board):
    row_increasing = 0
    row_decreasing = 0
    col_increasing = 0
    col_decreasing = 0

    for i in range(4):
        increasing = True
        decreasing = True
        for j in range(3):
            if board[i][j] > board[i][j + 1]:
                increasing = False
            if board[i][j] < board[i][j + 1]:
                decreasing = False
        if increasing:
            row_increasing += 1
        if decreasing:
            row_decreasing += 1
    for j in range(4):
        increasing = True
        decreasing = True
        for i in range(3):
            if board[i][j] > board[i + 1][j]:
                increasing = False
            if board[i][j] < board[i + 1][j]:
                decreasing = False
        if increasing:
            col_increasing += 1
        if decreasing:
            col_decreasing += 1

    return row_increasing + row_decreasing + col_increasing + col_decreasing



def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:
    This evaluation function assesses the given game state using a linear
    combination of various heuristic features to provide an overall score.
    The features considered are as follows:

    1. Max Tile in Corners: Checks if the highest tile is in one of the
    four corners of the board. A high score is awarded if the condition is met.
    2. Monotonicity: Evaluates how monotonic the rows and columns are,
    indicating that the tiles are either consistently non-increasing
    or non-decreasing.
    3. Smoothness: Measures the difference between adjacent tiles.
     A smaller difference indicates a smoother board,
     which is generally more desirable.
    4. Empty Tiles: Counts the number of empty tiles on the board,
    as having more empty spaces often provides more opportunities
     for making moves.
    5. Legal Actions: Considers the number of possible moves the player can
     make, encouraging flexibility and the ability to continue gameplay.

    The final evaluation score is computed as a weighted linear combination of the above features:
    - 10 * max_tile_in_corners
    - current score
    - max tile value
    - 20 * number of legal actions
    - 0.5 * monotonicity score
    - 0.5 * number of empty tiles
    - 0.5 * smoothness (negative contribution)

    """
    "*** YOUR CODE HERE ***"

    board = current_game_state.board
    score = current_game_state.score
    max_tile = current_game_state.max_tile

    max_tile_in_corners = check_corners_max_tile(board)
    monotonicity_score = monotonicity(board)
    smoothness = 0
    for i in range(4):
        for j in range(4):
            if j < 3:
                smoothness += abs(board[i][j] - board[i][j + 1])
            if i < 3:
                smoothness += abs(board[i][j] - board[i + 1][j])
    empty_tiles = np.count_nonzero(board == 0)
    evaluation = (
            10 * max_tile_in_corners
            + score
            + max_tile
            + 20 * len(current_game_state.get_agent_legal_actions())
            - 1/2 * smoothness
            + monotonicity_score + 0.5 * empty_tiles)
    return evaluation


# Abbreviation
better = better_evaluation_function
