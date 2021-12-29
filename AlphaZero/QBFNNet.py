import sys

from utils import *
from QBFGraph import Graph

import argparse
import torch
import torch.nn as nn

E2A_type = 0
A2E_type = 1
L2C_type = 2
RE_type = 3

class QBFNNet(nn.Module):
    """
    Implements the gated graph neural network (GGNN) that
    forms policy and value network for the QSAT game.
    """
    
    def __init__(self, args):
    
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.T = args.T
        super(QBFNNet, self).__init__()
        
        # matrix A for each edge type
        self.E2A = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.A2E = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.L2C = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.RE  = nn.Linear(self.hidden_dim,self.hidden_dim)
        
        self.GRU = nn.GRU(self.hidden_dim,self.hidden_dim,num_layers=1)
        
        self.fp = nn.Sequential(
                    nn.Linear(2*self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 2)
                    )
        self.gp = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 2)
                    )
        
        self.fv = nn.Sequential(
                    nn.Linear(2*self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1)
                    )
        self.gv = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1)
                    )


    def forward(self, graph):
        H0 = torch.zeros((graph.n_nodes,self.hidden_dim))
        for i,t in enumerate(graph.node_types):
            H0[i,t::3] = 1
        H = H0
        for _ in range(self.T):
            m = torch.zeros_like(H)
            # propagate messages
            for i,node in enumerate(graph.nodes()):
                neighbors = graph.neighbors(node)
                m[i] = (torch.sum(self.E2A(H[[j for (j,t) in neighbors if t == E2A_type]]),dim=0) +
                        torch.sum(self.A2E(H[[j for (j,t) in neighbors if t == A2E_type]]),dim=0) +
                        torch.sum(self.L2C(H[[j for (j,t) in neighbors if t == L2C_type]]),dim=0) +
                        torch.sum(self.RE(H[[j for (j,t) in neighbors if t == RE_type]]),dim=0))
            # compute next hidden states
            _,H = self.GRU(H.view(1,*H.shape),m.view(1,*m.shape))
            H = H.reshape((graph.n_nodes,self.hidden_dim))
        
        # compute p and V
        H0H = torch.cat((H0,H),dim = -1)
        p = torch.sum(torch.sigmoid(self.fp(H0H)) * self.gp(H),dim=0)
        v = torch.sum(torch.sigmoid(self.fv(H0H)) * self.gv(H),dim=0)
        
        return torch.log_softmax(p, dim=0), torch.tanh(v)

        
        
