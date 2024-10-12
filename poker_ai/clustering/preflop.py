from typing import Dict, Tuple

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


def make_starting_hand_lossy(starting_hand, short_deck=None) -> int:
    ranks = []
    suits = []
    for card in starting_hand:
        ranks.append(EvaluationCard.get_rank_int(card))
        suits.append(EvaluationCard.get_suit_int(card))
    
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    
    coords = [12 - r for r in ranks]
    # Cards in starting_hand is desc-sorted.
    if suited:
        return CHART[coords[0]][coords[1]]
    else:
        return CHART[coords[1]][coords[0]]


def compute_preflop_lossy_abstraction(builder) -> Dict[Tuple[int, int], int]:
    """Compute the preflop abstraction dictionary."""
    
    # Getting combos and indexing with lossless abstraction
    preflop_lossy: Dict[Tuple[int, int], int] = {}
    for starting_hand in builder.starting_hands:
        desc_sorted_starting_hand = (max(starting_hand), min(starting_hand))
        preflop_lossy[desc_sorted_starting_hand] = make_starting_hand_lossy(
            desc_sorted_starting_hand, builder
        )
    return preflop_lossy



def make_starting_hand_lossless(starting_hand, short_deck=None) -> int:
    """"""
    ranks = set()
    suits = set()
    for card in starting_hand:
        ranks.add(EvaluationCard.get_rank_int(card) + 2)
        suits.add(EvaluationCard.get_suit_int(card))
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    
    idx = 0
    for v in range(14, 1, -1):
        if all(c_rank == v for c_rank in ranks):
            return idx
        idx += 1
    for v in range(14, 2, -1):
        for v2 in range(v - 1, 1, -1):
            if (v in ranks and v2 in ranks) and suited:
                return idx
            idx += 1
    for v in range(14, 2, -1):
        for v2 in range(v - 1, 1, -1):
            if (v in ranks and v2 in ranks) and suited:
                return idx
            idx += 1


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[int, int], int]:
    """Compute the preflop abstraction dictionary."""
    
    # Getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[int, int], int] = {}
    for starting_hand in builder.starting_hands:
        desc_sorted_starting_hand = (max(starting_hand), min(starting_hand))
        preflop_lossless[desc_sorted_starting_hand] = make_starting_hand_lossless(
            desc_sorted_starting_hand, builder
        )
    return preflop_lossless
