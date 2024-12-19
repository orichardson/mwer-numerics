# mwer-numerics
minimax weighted expected regret simulations


Perhaps surprisingly, Halpern's minimax weighted expected regret (MWER) decision rule
[arXiv](https://arxiv.org/pdf/1302.5681),
a decision rule that interpolates between the conservative minimax expected regret (MER) and subjective expected utility (SEU) appears to be *less* conservative than either decision rule.

The following experiment is run by  `coins.py` in this repository.

 - Assume that there are $N+1$ coins, with biases ranging from 0 to 1. These are our possible theories.
 - For each possible choice of ``true theory'', we compute expected trajectories that result from using each decision rule.  In each case, the utilities and payoffs are given as follows:

  | U | No Action | Action |
  | --- | ------------- | ------------- |
  | tails | 0  | -5  |
  | heads | 0  | +1  |

![Alt text](figs/coin2-example.png?raw=true "Coin 2")
