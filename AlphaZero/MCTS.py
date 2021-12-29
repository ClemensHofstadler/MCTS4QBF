import logging
import math

import numpy as np
from collections import defaultdict

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, Enet, Anet, args):
        self.game = game
        self.Enet = Enet
        self.Anet = Anet
        self.args = args
        self.Qsa = defaultdict(lambda: 0)  # stores Q values for the state action pairs (s,a)
        self.Nsa = defaultdict(lambda: 0)  # stores the visit counts of the state action pairs (s,a)
        self.Ns = {}  # stores the visit counts of the state s
        self.Ps = {}  # stores initial policy (returned by the policy net)

        self.Es = {}  # stores game.getGameEnded ended for state s
    
    def reset(self):
        self.Qsa = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: 0)
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
   
    def getActionProb(self, graph):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        graph.

        Returns:
            probs: a probability vector where the probability of the i-th action is
                   proportional to Nsa[(s,a_i)]
        """
        for i in range(self.args.numMCTSSims+1):
            self.search(graph)
                            
        s = graph
        counts = [self.Nsa[(s, a)] for a in [0,1]]
                
        probs = [x / float(sum(counts)) for x in counts]
        
        return probs

    def search(self, graph):
        """
        This function performs one iteration of MCTS. It is recursively called
        untill a leaf node is found. The action chosen at each node is one that
        has the maximum PUCT.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is an end state, the outcome
		is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.


        Returns:
            v: the value of the leaf state
        """

        s = graph
        
        if s not in self.Es:
            self.Es[s] = self.game.gameEnded(s)
        if self.Es[s] != 0:
            # end state
            return self.Es[s]
        
        cur_player = graph.current_player()

        if s not in self.Ps:
            # leaf node
            if cur_player == 1:
                self.Ps[s], v = self.Enet.predict(s)
            else:
                self.Ps[s], v = self.Anet.predict(s)
            self.Ns[s] = 0
            return v

        # pick the action with the highest PUCT
        u0 = self.Qsa[(s, 0)] + self.args.cpuct * self.Ps[s][0] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, 0)])
        u1 = self.Qsa[(s, 1)] + self.args.cpuct * self.Ps[s][1] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, 1)])
    
        a = int(u0 < u1)
        next_graph = self.game.getNextState(graph, 2*a-1)

        v = self.search(next_graph)

        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + cur_player * v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1
        
        return v
