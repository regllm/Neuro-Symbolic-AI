import logging
import math
from pathlib import Path
from typing import List
from itertools import combinations

import joblib
import numpy as np
from tqdm import tqdm

from poker_ai.poker.evaluation.eval_card import EvaluationCard


log = logging.getLogger("poker_ai.clustering.runner")


class CardCombos:
    """This class stores combinations of cards (histories) per street."""

    def __init__(
        self, low_card_rank: int, high_card_rank: int, save_dir: str
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
        self._sorted_cards = self._cards.copy()
        self._sorted_cards.sort()

        self.starting_hands = self.get_card_combos(2)

        card_combos_flop_filename = f"card_combos_flop_{low_card_rank}_to_{high_card_rank}.joblib"
        self.card_combos_flop_path: Path = Path(save_dir) / card_combos_flop_filename
        card_combos_turn_filename = f"card_combos_turn_{low_card_rank}_to_{high_card_rank}.joblib"
        self.card_combos_turn_path: Path = Path(save_dir) / card_combos_turn_filename
        card_combos_river_filename = f"card_combos_river_{low_card_rank}_to_{high_card_rank}.joblib"
        self.card_combos_river_path: Path = Path(save_dir) / card_combos_river_filename

        try:
            self.flop = joblib.load(self.card_combos_flop_path)
            log.info("loaded flop")
        except FileNotFoundError:
            self.flop = self.create_info_combos(
                self.starting_hands, 3
            )
            log.info("created flop")
        try:
            self.turn = joblib.load(self.card_combos_turn_path)
            log.info("loaded turn")
        except FileNotFoundError:
            self.turn = self.create_info_combos(
                self.starting_hands, 4
            )
            log.info("created turn")
        try:
            self.river = joblib.load(self.card_combos_river_path)
            log.info("loaded river")
        except FileNotFoundError:
            self.river = self.create_info_combos(
                self.starting_hands, 5
            )
            log.info("created river")

    def get_card_combos(
        self, num_cards: int
    ) -> np.ndarray:
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
        combos = np.array([c for c in combinations(self._sorted_cards, num_cards)])

        return combos

    def create_info_combos(
        self, start_combos: np.ndarray, public_num_cards: int
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
        if public_num_cards == 3:
            betting_stage = "flop"
        elif public_num_cards == 4:
            betting_stage = "turn"
        elif public_num_cards == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        
        max_count = len(start_combos) * math.comb(len(self._cards) - len(start_combos[0]), public_num_cards)
        hand_size = len(start_combos[0]) + public_num_cards
        our_cards = np.zeros((max_count, hand_size))
        cursor = 0

        for start_combo in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            publics = np.array(
                [
                    c for c in combinations(
                        [c for c in self._sorted_cards if c not in start_combo],
                        public_num_cards,
                    )
                ]
            )
            for public_combo in publics:
                our_cards[cursor][:2] = start_combo[::-1]
                our_cards[cursor][2:] = public_combo[::-1]
                cursor += 1

        return our_cards
