import numpy as np
from pydepqbf import solve, QDPLL_QTYPE_FORALL, QDPLL_QTYPE_EXISTS, QDPLL_RESULT_SAT, QDPLL_RESULT_UNSAT
import time
from itertools import groupby
from random import *


type = "new"

NUM_VARS = (19,23)
NUM_CLAUSES = (7,11)
n = 10
prop_TRUE = 0.5

def make_TRUE():
    while True:
        num_vars = randint(*NUM_VARS)
        prefix = [1]*(num_vars//2) + [-1]*(num_vars//2)
        if num_vars % 2 == 1:
            prefix += [2*randint(0,1)-1]
        shuffle(prefix)
        matrix = []
        for c in range(randint(*NUM_CLAUSES)):
            len_clause = choices([2,3,4,5],weights=[0.1,0.2,0.3,0.4])[0]
            clause = zip([(2*randint(0,1)-1) for i in range(len_clause)],sample(range(1,num_vars+1),len_clause))
            clause = [s*v for (s,v) in clause]
            matrix.append(clause)
    
        translated_prefix = [(-1*q,(i+1,)) for i,q in enumerate(prefix)]
        if solve(translated_prefix,matrix)[0] == QDPLL_RESULT_SAT:
            return (prefix,matrix,"TRUE")
    
def make_FALSE():
    while True:
        num_vars = randint(*NUM_VARS)
        prefix = [1]*(num_vars//2) + [-1]*(num_vars//2)
        if num_vars % 2 == 1:
            prefix += [2*randint(0,1)-1]
        shuffle(prefix)
        matrix = []
        for c in range(randint(*NUM_CLAUSES)):
            len_clause = choices([2,3,4,5],weights=[0.1,0.2,0.3,0.4])[0]
            clause = zip([(2*randint(0,1)-1) for i in range(len_clause)],sample(range(1,num_vars+1),len_clause))
            clause = [s*v for (s,v) in clause]
            matrix.append(clause)
    
        translated_prefix = ((QDPLL_QTYPE_FORALL, tuple((i+1 for i in range(num_vars) if prefix[i] == -1))), (QDPLL_QTYPE_EXISTS, tuple((i+1 for i in range(num_vars) if prefix[i] == 1))))
        if solve(translated_prefix,matrix)[0] == QDPLL_RESULT_UNSAT:
            return (prefix,matrix,"FALSE")

def write_first_line(f,qbf):
    prefix,matrix,_ = qbf
    f.write("p cnf " + str(len(prefix)) + " " + str(len(matrix)) + "\n")
    
def write_scopes(f,qbf):
    prefix,matrix,_ = qbf
    for q,b in groupby(enumerate(prefix),lambda ix: ix[1]):
        if q < 0:
            f.write("a ")
        else:
            f.write("e ")
        for (v,_) in b:
            f.write(str(v+1) + " ")
        f.write("0\n")
        
def write_clauses(f,qbf):
    prefix,matrix,_ = qbf
    for c in matrix:
        for v in c:
            f.write(str(v) + " ")
        f.write("0\n")

QBFS_true = []
QBFS_false = []

for SAT in range(int(n*prop_TRUE)):
    QBFS_true.append(make_TRUE())
for UNSAT in range(n-int(n*prop_TRUE)):
    QBFS_false.append(make_FALSE())
    
i = 0
for qbf in QBFS_false:
    f = open("./formulas/False/random" + str(i) + ".qdimacs", "w");
    write_first_line(f,qbf)
    write_scopes(f,qbf)
    write_clauses(f,qbf)
    f.close()
    i += 1
    f = open("./formulas/QBFS_" + type + ".txt","a")
    prefix,matrix,value = qbf
    f.write(str(prefix).replace(" ", "") + " " + str(matrix).replace(" ", "") + " " + str(value) + "\n")
    f.close()

i = 0
for qbf in QBFS_true:
    f = open("./formulas/True/random" + str(i) + ".qdimacs", "w");
    write_first_line(f,qbf)
    write_scopes(f,qbf)
    write_clauses(f,qbf)
    f.close()
    i += 1
    f = open("./formulas/QBFS_" + type + ".txt","a")
    prefix,matrix,value = qbf
    f.write(str(prefix).replace(" ", "") + " " + str(matrix).replace(" ", "") + " " + str(value) + "\n")
    f.close()
