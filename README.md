# MCTS4QBF

In my master's thesis I implemented different methods to integrate a Monte Carlo Tree Search (MCTS) into the QBF solving process.
In particular, I adapted an open source Python implementation of the famous AlphaZero framework from [this page](https://github.com/suragnair/alpha-zero-general)
to predict the satisfiability of QBFs.
Additionally, in the C program MCTSsolve I integrated a MCTS as a preprocessing tool before passing a simplified QBF formula to the QBF solver DepQBF.

For further information, I refer to my master's thesis.
