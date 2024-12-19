# Minimax Weighted Expected Regret: Numerical Simulations

We study Halpern's minimax weighted expected regret (MWER) decision rule
[arXiv](https://arxiv.org/pdf/1302.5681),
a decision rule that interpolates between the conservative minimax expected regret (MER) and subjective expected utility (SEU).


The following experiment is run by  `coins.py` in this repository.

 - Assume that there are $N+1$ coins, with biases ranging from 0 to 1. These are our possible theories.
 - For each possible choice of ``true theory'', we compute expected trajectories that result from using each decision rule.  The precise numbers do not matter, but for concreteness, the utilities and payoffs are given as follows:  

  | U | No Action | Action |
  | --- | ------------- | ------------- |
  | tails | 0  | -5  |
  | heads | 0  | +1  |

 - We update weights on the theories with likelihood updating, as suggested by Halpern, which also coincides with the Bayesian updates of the standard Bayesian setting, e.g., the one used by [Bengio et. al.](https://arxiv.org/abs/2408.05284), 


## Preliminary Reults

Perhaps surprisingly, Halpern's minimax weighted expected regret (MWER) decision rule, which, again, interpolates between minimax expected regret minimization (which is a conservative decision rule) and expected utility maximization, appears to be *less* conservative than either decision rule.

Here are a few results supporting that claim.

![coin 2](figs/coin2-example.png?raw=true "Coin 2")
![coin 7](figs/coin7-example.png?raw=true "Coin 7")
![coin 9](figs/coin9-example.png?raw=true "Coin 9")
