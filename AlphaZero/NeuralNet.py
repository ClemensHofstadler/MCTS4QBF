import os
import sys
import time

import numpy as np
from tqdm import tqdm
from utils import *

import torch
import torch.optim as optim

from QBFNNet import QBFNNet as nnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 32,
    'cuda': torch.cuda.is_available(),
    'hidden_dim': 128,
    'T': 10
})

class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below.
    """
    def __init__(self):
        self.nnet = nnet(args)

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (s, pi, v). pi is the MCTS informed policy vector for
                      the given state s, and v is its value.
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                graphs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    target_pis, target_vs = target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi = torch.zeros_like(target_pis)
                out_v = torch.zeros_like(target_vs)
                for i,g in enumerate(graphs):
                    out_pi[i], out_v[i] = self.nnet(g)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), len(graphs))
                v_losses.update(l_v.item(), len(graphs))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


    def predict(self, graph):
        """
        Input:
            graph: current QBF graph

        Returns:
            pi: a policy vector for the current graph - a numpy array of length 2
            v: a float in [-1,1] that gives the value of the current board
        """
        # timing
        start = time.time()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(graph)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
