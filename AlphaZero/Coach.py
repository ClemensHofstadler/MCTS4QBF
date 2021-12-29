import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from NeuralNet import NeuralNet

log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, train_data, eval_data, Enet, Anet, args):
        self.train_data = train_data
        self.eval_data = eval_data
        self.Enet = Enet
        self.Anet = Anet
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        graph = self.train_data.getNewGame()
        self.curPlayer = graph.current_player()
        episodeStep = 0
        mcts = MCTS(self.train_data, self.Enet, self.Anet, self.args)  # reset search tree

        while True:
            episodeStep += 1
            
            pi = mcts.getActionProb(graph)
            trainExamples.append([graph, self.curPlayer, pi])

            action = 2*np.random.choice(len(pi), p=pi) - 1
            graph = self.train_data.getNextState(graph, action)

            r = self.train_data.gameEnded(graph)
            if r != 0:
                return [(g, a, r, p) for (g,p,a) in trainExamples]
            
            self.curPlayer = graph.current_player()

    def learn(self):
        """
        Performs epochs iterations with self_play_games episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        
        # evalute models
        log.info('EVALUATING MODELS')
        arena = Arena(self.eval_data,self.Enet,1,self.args,eval=True)
        acc = arena.playGames(self.args.eval_games,p_true=0.5)
        log.info('ACCURACY on EVAL DATA : %.2f' % acc)

        for i in range(1, self.args.epochs + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.self_play_games), desc="Self Play"):
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples()

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new networks, keeping copies of the old ones
            self.Enet.save_checkpoint(folder=self.args.checkpoint, filename='tempE.pth.tar')
            self.Anet.save_checkpoint(folder=self.args.checkpoint, filename='tempA.pth.tar')
           
            # train and pit Anet
            self.Anet.train([(g, a, r) for (g,a,r,p) in trainExamples if p == -1])
            arena = Arena(self.train_data,self.Anet,-1,self.args)
            log.info('PITTING AGAINST PREVIOUS VERSION A')
            prev_acc, new_acc = arena.playGames(self.args.arena_games,p_true=0)
            
            log.info('NEW/PREV ACCURACY : %.2f / %.2f' % (new_acc, prev_acc))
            if new_acc <= prev_acc:
                log.info('REJECTING NEW MODEL')
                try:
                    self.Anet.load_checkpoint(folder=self.args.checkpoint, filename='bestA.pth.tar')
                except:
                    self.Anet.load_checkpoint(folder=self.args.checkpoint, filename='tempA.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.Anet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i,'A'))
                self.Anet.save_checkpoint(folder=self.args.checkpoint, filename='bestA.pth.tar')
                
            # train and pit Enet
            self.Enet.train([(g, a, r) for (g,a,r,p) in trainExamples if p == 1])
            log.info('PITTING AGAINST PREVIOUS VERSION E')
            arena = Arena(self.train_data,self.Enet,1,self.args)
            prev_acc, new_acc = arena.playGames(self.args.arena_games,p_true=1)

            log.info('NEW/PREV ACCURACY : %.2f / %.2f' % (new_acc, prev_acc))
            if new_acc <= prev_acc:
                log.info('REJECTING NEW MODEL')
                try:
                    self.Enet.load_checkpoint(folder=self.args.checkpoint, filename='bestE.pth.tar')
                except:
                    self.Enet.load_checkpoint(folder=self.args.checkpoint, filename='tempE.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.Enet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i,'E'))
                self.Enet.save_checkpoint(folder=self.args.checkpoint, filename='bestE.pth.tar')
                
            # evalute models
            log.info('EVALUATING NEW MODELS')
            arena = Arena(self.eval_data,self.Enet,1,self.args,eval=True)
            acc = arena.playGames(self.args.eval_games,p_true=0.5)
            log.info('ACCURACY on EVAL DATA : %.2f' % acc)
            
            
    def getCheckpointFile(self, iteration, p):
        return 'checkpoint_' + p + '_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder,"checkpoint.pth.tar.examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        examplesFile = os.path.join(self.args.checkpoint,"checkpoint.pth.tar.examples")
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
