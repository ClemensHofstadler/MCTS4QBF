import logging

import coloredlogs

from Coach import Coach
from QBFGame import QBFGame
from NeuralNet import NeuralNet
from utils import *

TRAIN_PATH = './Formulas/QBFS_train.txt'
EVAL_PATH = './Formulas/QBFS_eval.txt'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'epochs': 30,
    'self_play_games': 40,           # Number of complete self-play games to simulate during a new iteration.
    'maxlenOfQueue': 2048,		    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,		         # Number of games moves for MCTS to simulate.
    'arena_games': 20,		         # Number of games to play during arena play to determine if new net will be accepted.
    'eval_games': 50,		      # Number of games to play during evaluation
    'cpuct': 1.41421356,            # exploration constant
    'load_folder_file': './temp/',
    'checkpoint': './temp/',
    'load_model': True,
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading training data...')
    train_data = QBFGame(TRAIN_PATH)
    log.info('Loading evaluation data...')
    eval_data = QBFGame(EVAL_PATH)
    
    log.info('Setting up networks...')
    Enet = NeuralNet()
    Anet = NeuralNet()
    Enet.save_checkpoint(folder=args.checkpoint, filename='bestE.pth.tar')
    Anet.save_checkpoint(folder=args.checkpoint, filename='bestA.pth.tar')

    if args.load_model:
        log.info('Loading checkpoint...')
        Enet.load_checkpoint(args.checkpoint, 'bestE.pth.tar')
        Anet.load_checkpoint(args.checkpoint, 'bestA.pth.tar')
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    coach = Coach(train_data, eval_data, Enet, Anet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    coach.learn()

if __name__ == "__main__":
    main()
