import logging

from tqdm import tqdm
import numpy as np

from NeuralNet import NeuralNet
from MCTS import MCTS

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, game, net, player, args, eval=False):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
           
        """
        self.game = game
        self.player = player
        self.new = net
        self.opp = loadBest(-1*player, args.checkpoint)
        self.best = loadBest(player, args.checkpoint)
        self.args = args
        self.eval = eval
        
    def playGame(self, graph, mcts):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        while self.game.gameEnded(graph) == 0:
            pi = mcts.getActionProb(graph)
            action = np.argmax(pi)
            # transform action to be +-1
            graph = self.game.getNextState(graph, 2*action-1)
       
        return self.game.gameEnded(graph)
        
    def getAccuracy(self, games, players):
        acc = 0
        for g in tqdm(games, desc="Arena.playGames (1)"):
            gameResult = self.playGame(g,players)
            if (gameResult == 1 and g.TRUE.lower() == "true") or (gameResult == -1 and g.TRUE.lower() == "false"):
                acc += 1

        return acc / len(games)
        

    def playGames(self, num, p_true=0.5):
        """
        Plays num games.

        Returns:
            new_acc: accuracy of new player
            prev_acc: accuracy of best previous player
        """
        games = self.game.getTestSet(num,p_true)
        
        if self.eval:
            if self.player == 1:
                mcts = MCTS(self.game, self.new, self.opp, self.args)
            else:
                mcts = MCTS(self.game, self.opp, self.new, self.args)
            return self.getAccuracy(games,mcts)
        
        else:
            if self.player == 1:
                mcts_prev = MCTS(self.game, self.best, self.opp, self.args)
                mcts_new = MCTS(self.game, self.new, self.opp, self.args)
            else:
                mcts_prev = MCTS(self.game, self.opp, self.best, self.args)
                mcts_new = MCTS(self.game, self.opp, self.new, self.args)
                        
            prev_acc = self.getAccuracy(games,mcts_prev)
            new_acc = self.getAccuracy(games,mcts_new)
            return prev_acc, new_acc
        
def loadBest(player,folder_name):
    net = NeuralNet()
    if player == 1:
        try:
            net.load_checkpoint(folder=folder_name, filename='bestE.pth.tar')
        except:
            net.load_checkpoint(folder=folder_name, filename='tempE.pth.tar')
    else:
        try:
            net.load_checkpoint(folder=folder_name, filename='bestA.pth.tar')
        except:
            net.load_checkpoint(folder=folder_name, filename='tempA.pth.tar')
    return net
