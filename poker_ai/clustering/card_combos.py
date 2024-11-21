import logging
import os
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

        card_combos_flop_csv_filename = f"card_combos_flop_csv_{low_card_rank}_to_{high_card_rank}.csv"
        self.card_combos_flop_csv_path: Path = Path(save_dir) / card_combos_flop_csv_filename
        card_combos_turn_csv_filename = f"card_combos_turn_csv_{low_card_rank}_to_{high_card_rank}.csv"
        self.card_combos_turn_csv_path: Path = Path(save_dir) / card_combos_turn_csv_filename
        card_combos_river_csv_filename = f"card_combos_river_csv_{low_card_rank}_to_{high_card_rank}.csv"
        self.card_combos_river_csv_path: Path = Path(save_dir) / card_combos_river_csv_filename

        if os.path.exists(self.card_combos_flop_path) and not os.path.exists(self.card_combos_flop_csv_path):
            flop = joblib.load(self.card_combos_flop_path)
            log.info("converting flop")
            with open(card_combos_flop_csv_filename, "w") as f:
                for row in tqdm(flop, ascii=" >="):
                    f.write(",".join([str(int(x)) for x in row]) + "\n")
            os.remove(self.card_combos_flop_path)
        elif not os.path.exists(self.card_combos_flop_csv_path):
            self.write_info_combos(self.starting_hands, 3, card_combos_flop_csv_filename)
            log.info("created flop")
        else:
            log.info("using pre-written flop")

        if os.path.exists(self.card_combos_turn_path) and not os.path.exists(self.card_combos_turn_csv_path):
            turn = joblib.load(self.card_combos_turn_path)
            log.info("converting turn")
            with open(card_combos_turn_csv_filename, "w") as f:
                for row in tqdm(turn, ascii=" >="):
                    f.write(",".join([str(int(x)) for x in row]) + "\n")
            os.remove(self.card_combos_turn_path)
        elif not os.path.exists(self.card_combos_turn_csv_path):
            self.write_info_combos(self.starting_hands, 3, card_combos_turn_csv_filename)
            log.info("created turn")
        else:
            log.info("using pre-written turn")

        if os.path.exists(self.card_combos_river_path) and not os.path.exists(self.card_combos_river_csv_path):
            river = joblib.load(self.card_combos_river_path)
            log.info("converting river")
            with open(card_combos_river_csv_filename, "w") as f:
                for row in tqdm(river, ascii=" >="):
                    f.write(",".join([str(int(x)) for x in row]) + "\n")
            os.remove(self.card_combos_river_path)
        elif not os.path.exists(self.card_combos_river_csv_path):
            self.write_info_combos(self.starting_hands, 3, card_combos_river_csv_filename)
            log.info("created river")
        else:
            log.info("using pre-written river")

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

    def write_info_combos(
        self, start_combos: np.ndarray, public_num_cards: int, output_path: str
    ):
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

        with open(output_path, "w") as f:
            for start_combo in tqdm(
                start_combos,
                dynamic_ncols=True,
                desc=f"Creating {betting_stage} info combos into a file",
                ascii=" >=",
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
                    f.write(",".join([str(int(x)) for x in start_combo[::-1]]))
                    f.write(",")
                    f.write(",".join([str(int(x)) for x in public_combo[::-1]]))
                    f.write("\n")

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
            ascii=" >=",
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
