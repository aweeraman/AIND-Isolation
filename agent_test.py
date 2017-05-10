"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

from isolation import *
from game_agent import MinimaxPlayer
from game_agent import AlphaBetaPlayer

from importlib import reload

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        pass

    def test_minimax(self):
        self.player1 = MinimaxPlayer()
        self.player2 = MinimaxPlayer()
        self.game = Board(self.player1, self.player2)
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 5))
        winner, history, outcome = self.game.play()
        self.assertTrue(type(history) == list)
        self.assertTrue(len(history) > 1)

    def test_alphabeta(self):
        self.player1 = AlphaBetaPlayer()
        self.player2 = AlphaBetaPlayer()
        self.game = Board(self.player1, self.player2)
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 5))
        winner, history, outcome = self.game.play()
        self.assertTrue(type(history) == list)
        self.assertTrue(len(history) > 1)

if __name__ == '__main__':
    unittest.main()
