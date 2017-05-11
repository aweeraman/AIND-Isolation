"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

POSINF = float('inf')
NEGINF = float('-inf')

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_next_legal_moves(game, player, current_location):
    "Returns the list of legal moves for the provided location"
    r, c = current_location
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions 
                   if game.move_is_legal((r + dr, c + dc))]
    return valid_moves


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return POSINF
    elif game.is_loser(player):
        return NEGINF

    points = 0.

    board_center = (int(game.width / 2), int(game.height / 2))

    location = game.get_player_location(player)

    if location == board_center:
        points += 5

        valid_moves = get_next_legal_moves(game, player, location)

        points += 0.5 * len(valid_moves)

    return points 


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return POSINF
    elif game.is_loser(player):
        return NEGINF

    points = 0.

    board_center = (int(game.width / 2), int(game.height / 2))

    location = game.get_player_location(player)
    opponent = game.get_player_location(game.get_opponent(player))

    valid_moves_player = get_next_legal_moves(game, player, location)
    valid_moves_opponent = get_next_legal_moves(game, player, opponent)

    points += len(valid_moves_player) - len(valid_moves_opponent)

    return points 


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return POSINF
    elif game.is_loser(player):
        return NEGINF

    points = 0.

    top = [(0, i) for i in range(game.width)]
    right = [(i, game.width - 1) for i in range(game.height)]
    bottom = [(game.height - 1, i) for i in range(game.width)]
    left = [(i, 0) for i in range(game.height)]
    corners = set(top + right + bottom + left)

    location = game.get_player_location(player)
    opponent = game.get_player_location(game.get_opponent(player))

    valid_moves_player = get_next_legal_moves(game, player, location)
    valid_moves_opponent = get_next_legal_moves(game, player, opponent)

    player_corner_moves = 0.
    opponent_corner_moves = 0.

    if location in corners:
        player_corner_moves += 1

    if opponent in corners:
        opponent_corner_moves += 1

    for move in valid_moves_player:
        if move in corners:
            player_corner_moves += 1

    for move in valid_moves_opponent:
        if move in corners:
            opponent_corner_moves += 1

    player_corner_ratio = 0.
    opponent_corner_ratio = 0.

    if len(valid_moves_player) != 0:
        player_corner_ratio = player_corner_moves / float(len(valid_moves_player))

    if len(valid_moves_opponent) != 0:
        opponent_corner_ratio = opponent_corner_moves / float(len(valid_moves_opponent))

    return player_corner_ratio - opponent_corner_ratio


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, max_player=True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _ , move = self.minimax_helper(game, depth)

        return move

    def minimax_helper(self, game, depth, max_player=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_moves = game.get_legal_moves()

        if not possible_moves:
            return game.utility(self), None

        if depth <= 0:
            return self.score(game, self), None

        best_score = None
        best_move = None

        if max_player:
           best_score = NEGINF
           for move in possible_moves:
                forecast = game.forecast_move(move)
                current_score, _ = self.minimax_helper(forecast, depth - 1, False)
                if current_score > best_score:
                    best_score = current_score
                    best_move = move
        else:
            best_score = POSINF
            for move in possible_moves:
                forecast = game.forecast_move(move)
                current_score, _ = self.minimax_helper(forecast, depth - 1, True)
                if current_score < best_score:
                    best_score = current_score
                    best_move = move

        return best_score, best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = self.search_depth
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _, move = self.alphabeta_helper(game, depth, alpha, beta)

        if move not in game.get_legal_moves():
            return (-1, -1)

        return move

    def alphabeta_helper(self, game, depth, alpha=float("-inf"), beta=float("inf"), max_player=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_moves = game.get_legal_moves()

        if not possible_moves:
            return game.utility(self), None

        if depth <= 0:
            return self.score(game, self), None

        best_score = None
        best_move = None

        if max_player:
           best_score = NEGINF
           for move in possible_moves:
                forecast = game.forecast_move(move)
                current_score, _ = self.alphabeta_helper(forecast, depth - 1, alpha, beta, False)
                alpha = max(alpha, current_score)
                if current_score > best_score:
                    best_score = current_score
                    best_move = move
                if alpha >= beta:
                    break
        else:
            best_score = POSINF
            for move in possible_moves:
                forecast = game.forecast_move(move)
                current_score, _ = self.alphabeta_helper(forecast, depth - 1, alpha, beta, True)
                beta = min(beta, current_score)
                if current_score < best_score:
                    best_score = current_score
                    best_move = move
                if alpha >= beta:
                    break

        return best_score, best_move
