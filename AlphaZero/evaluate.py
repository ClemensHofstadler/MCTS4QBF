import sys
import numpy as np
import time

from QBFGame import QBFGame
from NeuralNet import NeuralNet
from standard_MCTS import standard_MCTS
from MCTS import MCTS

from utils import *
import torch

NUM_ITER = 1
NUM_EXAMPLES = -1
MODE = 'alphazero'

args_pit = dotdict({
    'numMCTSSims': 25,
    'cpuct': 1.41421356,
    'checkpoint': './temp/',
    'num_wins': 1
})

if (len(sys.argv) < 2):
    print("Please provide path to data")
    exit()

game = QBFGame(sys.argv[1])
games = game.getTestSet(NUM_EXAMPLES,p_true=0.5)

print("Mode : ", MODE)
times = []
accs = []
for iter in range(NUM_ITER):
    correct = 0
    incorrect = 0
    false_positive = 0
    false_negative = 0
    
    start = time.time()

    if MODE != 'alphazero':
        mcts = standard_MCTS(game,args_pit,MODE)
    else:
        Enet = NeuralNet()
        Anet = NeuralNet()
        Enet.load_checkpoint(folder=args_pit.checkpoint, filename='bestE.pth.tar')
        Anet.load_checkpoint(folder=args_pit.checkpoint, filename='bestA.pth.tar')
        mcts = MCTS(game,Enet,Anet,args_pit)
    
    for graph in games:
        wonE = wonA = 0
        while wonE < args_pit.num_wins and wonA < args_pit.num_wins:
            mcts.reset()
            actions = []
            g = graph
            while g.done == 0:
                pi = mcts.getActionProb(g)
                action = np.argmax(pi)
                actions.append(action)
                # transform action to be +-1
                g = game.getNextState(g, 2*action - 1)
            if g.done == 1:
                wonE += 1
            else:
                wonA += 1
            
        if (wonE > wonA and graph.TRUE.lower() == "true") or (wonA > wonE and graph.TRUE.lower() == "false"):
            correct += 1
        else:
            incorrect += 1
            if graph.TRUE.lower() == "true":
                false_negative += 1
            else:
                false_positive += 1
    
    time_needed = time.time() -start

    print("Time : ",time_needed,"\n")
    print("Correct : ", str(correct))
    print("False positive : ", false_positive)
    print("False negative : ", false_negative)
    
    times += [time_needed]
    accs += [correct / len(games)]

print("Average time = ", sum(times) / NUM_ITER)
print("Average acc = ", sum(accs) / NUM_ITER)
