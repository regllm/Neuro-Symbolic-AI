from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card
from poker_ai.poker.evaluation.eval_card import EvaluationCard


# This chart represents the pre-flop ranges for
# 6-max No-Limit Hold'Em tournament without antes from:
# https://matchpoker.com/learn/strategy-guides/pre-flop-ranges-6-max
CHART_STRING = """
1111111221111
1111122333444
1111123444555
1221123455555
1333113445555
4444412345566
4555541234566
4555555223456
4567765522456
4567766552245
4667776655245
5667777766635
5667777776663
"""
CHART = [
    [int(x) for x in line.strip()]
    for line in CHART_STRING.strip().split("\n")
]


def make_starting_hand_lossy(starting_hand, short_deck) -> int:
    ranks = []
    suits = []
    for card in starting_hand:
        ranks.add(EvaluationCard.get_rank_int(card))
        suits.add(EvaluationCard.get_suit_int(card))
    
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    
    coords = [14 - r for r in ranks]
    # Cards in starting_hand is desc-sorted.
    if suited:
        return CHART[coords[0]][coords[1]]
    else:
        return CHART[coords[1]][coords[0]]
