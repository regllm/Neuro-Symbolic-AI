"""
Comments on notation found in the appendix.

Notation
--------
  P :
    A set of players, where P_i is one player.
  h :
    A node (i.e history), h is defined by all information of the current
    situation, including private knowledge known only to one player.
  A(h) :
    Denotes the actions available at a node
  P(h) :
    Denotes either the chance or player it is to act at that node.
  I :
    Imperfect information is represented by information sets (infosets) for
    each player P_i. For any infoset belonging to a player P_i, all nodes h to
    h' in the infoset I are indestinguishable to the player P_i. Moreover,
    every non-terminal node h in belongs to exactly one infoset for each
    player.
  sigma (lowercase) :
    A strategy (i.e a policy). Here sigma(I) is a probability vector over the
    actions for acting player P_i in infoset I.
"""
import numpy as np


def monte_carlo_cfr_with_pruning(t):
    """Conduct External-Sampling Monte Carlo CFR with pruning."""
    for player in players:
        pass


def calculate_strategy(R, I):
    """Caluclates the strategy based on regrets."""
    pass


def update_strategy(h, P):
    """Update the average strategy of P_i"""
    pass


def traverse_monte_carlo_cfr(h, P):
    """"Update the regrets for P_i"""
    pass


def traverse_monte_carlo_cfr_with_pruning(h, P):
    """MCCFR with pruning for very negative regrets."""
    pass
