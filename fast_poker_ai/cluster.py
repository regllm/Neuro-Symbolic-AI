from typing import Iterable

import time
import math
from itertools import combinations

import numpy as np

from eval_card import EvaluationCard


def create_deck(low_rank: int, high_rank: int):
    base_suits = "shdc"
    base_ranks = "23456789TJQKA"
    ranks = [
        base_ranks[x - 2]
        for x in range(low_rank, high_rank + 1)
    ]

    rank_count = high_rank - low_rank + 1
    deck = np.zeros(rank_count * 4, dtype=int)

    cursor = 0
    for suit in base_suits:
        for rank in ranks:
            deck[cursor] = EvaluationCard.new(rank + suit)
            cursor += 1

    deck.sort()
    
    return deck


def create_card_combos(deck: Iterable[int], count: int):
    return np.array([c for c in combinations(deck, count)])


def create_info_combos(
    deck: Iterable[int],
    start_combos: Iterable[Iterable[int]],
    public_count: int,
):
    max_count = len(start_combos) * math.comb(len(deck) - len(start_combos[0]), public_count)
    hand_size = len(start_combos[0]) + public_count
    our_cards = np.zeros((max_count, hand_size))
    cursor = 0

    for start_combo in start_combos:
        publics = np.array(
            [
                c for c in combinations(
                    [c for c in deck if c not in start_combo],
                    public_count,
                )
            ]
        )
        for public_combo in publics:
            our_cards[cursor][:2] = start_combo[::-1]
            our_cards[cursor][2:] = public_combo[::-1]
            cursor += 1

    return our_cards


def simulate_river_games(
    deck: Iterable[int],
    combo: Iterable[int],
):
    pass


def simulate_river_hand_strengths(
    deck: Iterable[int],
    river_combos: Iterable[Iterable[int]],
):
    river_combos_size = (
        math.comb(len(deck), 2)
        * math.comb(len(deck) - 2, 5)
    )
    result_width = 3
    result = np.ndarray(
        (river_combos_size, result_width), dtype=np.double,
    )

    for i, combo in enumerate(river_combos):
        result[i] = simulate_river_games(deck, combo)
    return result



def create_combos(low_rank: int, high_rank: int):
    deck = create_deck(low_rank, high_rank)
    start_combos = create_card_combos(deck, 2)

    start_at = time.time()
    flop_combos = create_info_combos(deck, start_combos, 3)
    duration = time.time() - start_at
    
    print(f"Created Flop combos in {duration:.2f} seconds.")

    start_at = time.time()
    turn_combos = create_info_combos(deck, start_combos, 4)
    duration = time.time() - start_at
    
    print(f"Created Turn combos in {duration:.2f} seconds.")

    start_at = time.time()
    river_combos = create_info_combos(deck, start_combos, 5)
    duration = time.time() - start_at
    
    print(f"Created River combos in {duration:.2f} seconds.")


def main():
    create_combos(2, 5)


if __name__ == "__main__":
    main()
