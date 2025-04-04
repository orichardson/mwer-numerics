# Minimax Weighted Expected Regret: Numerical Simulations

We study Halpern's minimax weighted expected regret (MWER) decision rule
[[arXiv](https://arxiv.org/pdf/1302.5681)],
a decision rule that interpolates between the conservative minimax expected regret (MER) and subjective expected utility (SEU).


## An Experiment

The following experiment is run by  `coins.py` in this repository.

 - Assume that there are $N+1$ coins, with biases ranging from 0 to 1. These are our possible theories.
 - For each possible choice of ``true theory'', we compute expected trajectories that result from using each decision rule.  The precise numbers do not matter, but for concreteness, the payoff matrix we use is given below:  

  | Utility | No Action | Action |
  | --- | ------------- | ------------- |
  | tails | 0  | -5  |
  | heads | 0  | +1  |

 - We update weights on the theories with likelihood updating, as suggested by Halpern, which also coincides with the Bayesian updates of the standard Bayesian setting, e.g., the one used by [Bengio et. al.](https://arxiv.org/abs/2408.05284),
 - We average over $M = 5000% trajectories, and consider learning across timesteps $t \in \{0, 1, \ldots, T-1\}$, and the figures below are computed with $T=20$, and $N+1 = 11$ coins.


## Preliminary Reults

**This may well be the result of a buggy implementation.** 
<!-- But if the code is correct, then we have found a surprising result. -->

Lueng and Halpern's minimax weighted expected regret (MWER) decision rule, which, again, interpolates between minimax expected regret minimization (which is a conservative decision rule) and expected utility maximization, appears to be *less* conservative than either decision rule.

here are some emperical results supporting that claim.

Each plot below is for one fixed choice of true theory.
In each case, the $x$-axis is the timestep $t$.  
 - In the upper left corner, one sees how the theory weights evolve over time. Notice that, as expected, they approaching the true theory (on average).
 - In the upper right corner, one can see the expected total utility gained by using each decision rule.  Here is the surprise (or perhaps, the bug): Halperns MWER rule is *less* conservative than expected utility maximization.
 - In the bottom left, one can see the probability of acting according to each decision rule. Acting is, of course, risky. Note that MWER acts before EU does. 

![coin 2](figs/coin2-example.png?raw=true "Coin 2")
![coin 5](figs/coin7-example.png?raw=true "Coin 5")
![coin 7](figs/coin7-example.png?raw=true "Coin 7")
![coin 9](figs/coin9-example.png?raw=true "Coin 9")


To generate similar plots for other parameters, clone this repository and run the file `coin.py`. There is a nice user interface, and one can navigate these plots using the arrow keys :)
