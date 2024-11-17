import logging
from typing import List
from itertools import combinations
import operator

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits
from poker_ai.poker.evaluation.eval_card import EvaluationCard


log = logging.getLogger("poker_ai.clustering.runner")


class CardCombos:
    """This class stores combinations of cards (histories) per street."""

    def __init__(
        self, low_card_rank: int, high_card_rank: int,
    ):
        super().__init__()
        # Sort for caching.
        eval_suits = "shdc"
        eval_ranks: List[str] = [
            EvaluationCard.STR_RANKS[rank - 2]
            for rank in range(low_card_rank, high_card_rank + 1)
        ]
        
        self._cards = np.array(
            [
                EvaluationCard.new(rank_char + suit_char)
                for suit_char in eval_suits
                for rank_char in eval_ranks
            ],
        )

        self.starting_hands = self.get_card_combos(2)

        self.flop = self.create_info_combos(
            self.starting_hands, self.get_card_combos(3)
        )
        log.info("created flop")
        self.turn = self.create_info_combos(
            self.starting_hands, self.get_card_combos(4)
        )
        log.info("created turn")
        self.river = self.create_info_combos(
            self.starting_hands, self.get_card_combos(5)
        )
        log.info("created river")

    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """
        Get the card combinations for a given street.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> np.ndarray
        """
        combos = np.array([c for c in combinations(self._cards, num_cards)])
        # Sort each combo in ascending orders.
        combos.sort(1)
        return combos

    def create_info_combos(
        self, start_combos: np.ndarray, publics: np.ndarray
    ) -> np.ndarray:
        """Combinations of private info(hole cards) and public info (board).

        Uses the logic that a AsKsJs on flop with a 10s on turn is the same
        as AsKs10s on flop and Js on turn. That logic is used within the
        literature as well as the logic where those two are different.

        Parameters
        ----------
        start_combos : np.ndarray
            Starting combination of cards (beginning with hole cards)
        publics : np.ndarray
            Public cards being added
        Returns
        -------
            Combinations of private information (hole cards) and public
            information (board)
        """
        if publics.shape[1] == 3:
            betting_stage = "flop"
        elif publics.shape[1] == 4:
            betting_stage = "turn"
        elif publics.shape[1] == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        
        max_count = len(start_combos) * len(publics)
        hand_size = len(start_combos[0]) + len(publics[0])
        our_cards = np.zeros((max_count, hand_size))
        count = 0

        for start_combo in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            for public_combo in publics:
                if not np.any(np.isin(start_combo, public_combo)):
                    # Combine hand and public cards.
                    our_cards[count][:2] = start_combo[::-1]
                    our_cards[count][2:] = public_combo[::-1]
                    count += 1
        return our_cards[:count]
