from __future__ import print_function
import sys
import ast
from random import sample, shuffle

import numpy as np
from QBFGraph import Graph

class QBFGame():
    """
    This class specifies the basics of the QBF game.
	It loads and stores all QBF formulas that can be played on.
    Use 1 for player1 and -1 for player2.
	
    """
    def __init__(self, QBFS):
        self.games = [Graph(q) for q in load_qbfs(QBFS)]
        self.idx = list(np.random.randint(0,len(self.games),len(self.games)))

    def getInitialState(self):
        return self.games[self.idx[-1]]
        
    def getNewGame(self):
        if len(self.idx) == 0:
            self.idx = list(np.random.randint(0,len(self.games),len(self.games)))
        return self.games[self.idx.pop()]
        

    def getNextState(self, graph, action):
        assert abs(action) == 1
        return graph.set_variable(action)
        
    def getTestSet(self, num, p_true=0.5):
        n_true = int(p_true * num)
        n_false = num - n_true
        if num > 0:
            games = sample([g for g in self.games if g.TRUE.lower() == "true"],n_true)
            games += sample([g for g in self.games if g.TRUE.lower() == "false"],n_false)
        else:
            games = self.games
        shuffle(games)
        return games

    def gameEnded(self, graph):
        """
        Input:
            graph: current QBF graph

        Returns:
            0 if game has not ended, 1 if E-player won, -1 if A-player won
        """
        return graph.done
        

def load_qbfs(path):
    QBFS = []
    with open(path) as file:
        for line in file.readlines():
            prefix,matrix,sat = line.split()
            prefix = ast.literal_eval(prefix)
            matrix = ast.literal_eval(matrix)
            QBFS.append((prefix,matrix,sat))
    return QBFS
