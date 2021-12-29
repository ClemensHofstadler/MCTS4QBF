import logging
import math
import numpy as np
from collections import defaultdict

class standard_MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, args, mode):
        self.game = game
        self.args = args
        self.Qsa = defaultdict(lambda: 0)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(lambda: 0)  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        
        self.mode = mode

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
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]
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
        till a leaf node is found. The action chosen at each node is one that
        has the maximum UCT.

        Once a leaf node is found, a rollout is done according to the
        rollout strategy. This end result of this game is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        
        Returns:
            v: the value of the leaf state
        """

        s = graph
        n_rollouts = 10
       
        if s not in self.Es:
            self.Es[s] = self.game.gameEnded(s)
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]
        
        cur_player = graph.current_player()

        if s not in self.Ps:
            # leaf node
            if self.mode == 'heuristic':
                s0 = s.set_variable(-1)
                s1 = s.set_variable(1)
                r0 = [s0.rollout() for _ in range(n_rollouts)].count(cur_player)
                r1 = [s1.rollout() for _ in range(n_rollouts)].count(cur_player)
                self.Ps[s] = [(r0+1)/(r0+r1+2),(r1+1)/(r0+r1+2)]
                v =  (r0+r1) / n_rollouts - 1
                cur_player = 1
            if self.mode == 'standard':
                self.Ps[s] = [0.5,0.5]
                v = s.rollout()
            if self.mode == 'hard_rollouts':
                self.Ps[s] = [0.5,0.5]
                v = s.hard_rollout()
    
            self.Ns[s] = 1
            return v
        # pick the action with the highest upper confidence bound
        u0 = self.Qsa[(s, 0)] + self.args.cpuct * self.Ps[s][0] * math.sqrt(math.log(self.Ns[s]) / (1 + self.Nsa[(s, 0)]))
       
        u1 = self.Qsa[(s, 1)] + self.args.cpuct * self.Ps[s][1] * math.sqrt(math.log(self.Ns[s]) / (1 + self.Nsa[(s, 1)]))
    
        if u0 == u1:
            a = np.random.randint(2)
        else:
            a = int(u0 < u1)
        next_graph = self.game.getNextState(graph, 2*a-1)

        v = self.search(next_graph)

        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + cur_player * v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1
        return v
