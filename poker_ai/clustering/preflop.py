from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card
from poker_ai.poker.evaluation.eval_card import EvaluationCard


# def make_starting_hand_lossless(starting_hand, short_deck) -> int:
#     """"""
#     ranks = []
#     suits = []
#     for card in starting_hand:
#         ranks.append(card.rank_int)
#         suits.append(card.suit)
#     if len(set(suits)) == 1:
#         suited = True
#     else:
#         suited = False
#     if all(c_rank == 14 for c_rank in ranks):
#         return 0
#     elif all(c_rank == 13 for c_rank in ranks):
#         return 1
#     elif all(c_rank == 12 for c_rank in ranks):
#         return 2
#     elif all(c_rank == 11 for c_rank in ranks):
#         return 3
#     elif all(c_rank == 10 for c_rank in ranks):
#         return 4
#     elif 14 in ranks and 13 in ranks:
#         return 5 if suited else 15
#     elif 14 in ranks and 12 in ranks:
#         return 6 if suited else 16
#     elif 14 in ranks and 11 in ranks:
#         return 7 if suited else 17
#     elif 14 in ranks and 10 in ranks:
#         return 8 if suited else 18
#     elif 13 in ranks and 12 in ranks:
#         return 9 if suited else 19
#     elif 13 in ranks and 11 in ranks:
#         return 10 if suited else 20
#     elif 13 in ranks and 10 in ranks:
#         return 11 if suited else 21
#     elif 12 in ranks and 11 in ranks:
#         return 12 if suited else 22
#     elif 12 in ranks and 10 in ranks:
#         return 13 if suited else 23
#     elif 11 in ranks and 10 in ranks:
#         return 14 if suited else 24


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
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    
    # Getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[int, int], int] = {}
    for starting_hand in builder.starting_hands:
        desc_sorted_starting_hand = (max(starting_hand), min(starting_hand))
        preflop_lossless[desc_sorted_starting_hand] = make_starting_hand_lossless(
            desc_sorted_starting_hand, builder
        )
    return preflop_lossless
