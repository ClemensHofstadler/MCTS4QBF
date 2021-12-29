import torch
import numpy as np

from itertools import groupby
from math import copysign


E2A_type = 0
A2E_type = 1
L2C_type = 2
RE_type = 3

class Graph():
    """
    Graph class to represent QBFs in PCNF
    
    prefix contains -1,1 for each variable
    -1...universial quantifier
    1...existential quantifier

    matrix is list of clauses
    clause is list of non-zero integers
    > 0: positive literal
    < 0: negative literal
    """
    
    def __init__(self,QBF):
        """
        Set up initial QBF graph
        """

        if len(QBF) == 2:
            (prefix,matrix) = QBF
            self.TRUE = None
            self.partial_assignment = []
        else:
            (prefix,matrix,self.TRUE) = QBF
      
        # done = 1 <-> E won
        # done = -1 <-> A won
        if len(matrix) == 0:
            self.done = 1
        elif [] in matrix:
            self.done = -1
        else:
            self.done = 0
            
        self.prefix = prefix
        self.matrix = matrix

        l = len(prefix)
        self.n_nodes = 2*l + len(matrix)
        # i <-> node for x_i
        # len(prefix) + i <-> node for !x_i
        # 2*len(prefix) + i <-> node for clause_i
        
        # node types:
        # 0...existential node
        # 1...universial node
        # 2...clause node
        self.node_types = 2*[(q+1)//2 for q in prefix] + [2]*len(matrix)
        self.edges = [[] for i in range(self.n_nodes)]
            
        # make reflexive edges
        self.edges[:l] = [[(i+l,RE_type)] for i in range(l)]
        self.edges[l:2*l] = [[(i,RE_type)] for i in range(l)]
            
        # make E2A and A2E edges
        prev_block = []
        for q,b in groupby(enumerate(prefix),lambda ix: ix[1]):
            block = list(b)
            t = (q+1)//2
            for xi,_ in prev_block:
                for xj,_ in block:
                    self.edges[xi].append((xj,t))
                    self.edges[xj].append((xi,t))
                    self.edges[xi+l].append((xj+l,t))
                    self.edges[xj+l].append((xi+l,t))
            prev_block = block
            
        # make L2C edges
        for i,clause in enumerate(matrix):
            for x in clause:
                if x > 0:
                    self.edges[x-1].append((2*l+i,L2C_type))
                    self.edges[2*l+i].append((x-1,L2C_type))
                if x < 0:
                    self.edges[l-x-1].append((2*l+i,L2C_type))
                    self.edges[2*l+i].append((l-x-1,L2C_type))

    def neighbors(self,node):
        """
        Returns the edges in the form (to,type)
        """
        return self.edges[node]

    def nodes(self):
        return range(self.n_nodes)
        
    def current_player(self):
        return self.prefix[0]
    
    def set_variable(self,v):
        """
        Set first variable to value v
        """
        assert abs(v) == 1
        new_matrix = []
        for clause in self.matrix:
            if v not in clause:
                new_matrix.append([int(copysign(abs(x)-1,x)) for x in clause if x != -v])

        g = Graph((self.prefix[1:],new_matrix))
        return g
    
    def evaluate(self,mode=0):
        if mode == 0:
            return [0.5,0.5],0
            
        if self.current_player == 1:
            if [1] in self.matrix:
                return [0,1],1
            elif [-1] in self.matrix:
                return [1,0],1
            else:
                return [0.5,0.5],0
        else:
            if [1] in self.matrix:
                return [1,0],-1
            elif [-1] in self.matrix:
                return [0,1],-1
            else:
                return [0.5,0.5],0
        
            
    def rollout(self):
        """
        Plays a random rollout.
        """
        matrix = self.matrix
        while matrix != []:
            new_matrix = []
            v = np.random.choice([-1,1])
            for clause in matrix:
                if v not in clause:
                    new_clause = [np.sign(x)*(abs(x)-1) for x in clause if x != -v]
                    if new_clause == []:
                        return False
                    else:
                        new_matrix.append(new_clause)
            matrix = new_matrix
        return True
        
    def get_pure_lits(self,matrix):
        pure_lits = set()
        lits = set(lit for clause in matrix for lit in clause)
        pure_lits = {self.prefix[abs(l)-1] * l for l in lits if -l not in lits}
        return pure_lits
        
    def is_unit_clause(self, clause):
        ext_lits = [clause[i] for i in range(len(clause)) if self.prefix[abs(clause[i])-1] > 0]
        if len(ext_lits) == 0:
            return -1
        elif len(ext_lits) == 1 and abs(ext_lits[0]) == min(abs(v) for v in clause):
            return ext_lits[0]
        else:
            return 0
    
    def set_variable_rollout(self,v,matrix):
        unit_lits = set()
        new_matrix = []
        for clause in matrix:
            if v not in clause:
                new_clause = [x for x in clause if x != -v]
                if new_clause == []:
                    return [],-1,unit_lits
                lit = self.is_unit_clause(new_clause)
                if lit == -1:
                    return [],-1,unit_lits
                elif lit > 0:
                    unit_lits.add(lit)
                else:
                    new_matrix.append(new_clause)
        if new_matrix == []:
            return new_matrix,1,unit_lits
        else:
            return new_matrix,0,unit_lits
        
    def hard_rollout(self):
        """
        Plays a (semi-)random rollout with unit and pure literal elimination.
        """
        matrix = self.matrix
        vars = set(abs(lit) for clause in matrix for lit in clause)
        unit_lits = set()
        forced = set()
        flag = 0
        while flag == 0:
            pure_lits = self.get_pure_lits(matrix)
            forced = forced.union(pure_lits)
            while len(forced) > 0 and flag == 0:
                v = forced.pop()
                if abs(v) in vars:
                    vars.remove(abs(v))
                    matrix,flag,unit_lits = self.set_variable_rollout(v,matrix)
                    forced = forced.union(unit_lits)
                    pure_lits = self.get_pure_lits(matrix)
                    forced = forced.union(pure_lits)
                
            if flag != 0:
                return flag > 0
            
            var = vars.pop()
            matrix,flag,unit_lits = self.set_variable_rollout(var * np.random.choice([-1,1]),matrix)
            
        return flag > 0
            
    def __eq__(self, other):
        if not isinstance(other, Graph):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.prefix == other.prefix and self.matrix == other.matrix
    
    def __hash__(self):
        return hash(str(self.prefix) + str(self.matrix))
        
    def __repr__(self):
        if self.TRUE:
            return str(self.prefix) + " " + str(self.matrix) + " " + self.TRUE.upper()
        else:
            return str(self.prefix) + " " + str(self.matrix)
            

