from typing import Sequence

from poker_ai.poker.evaluation.eval_card import EvaluationCard

class ComboLookup:
    def __init__(self):
        self.lookup_table = {}

    def _get_small_int(self, eval_card: int):
        rank = EvaluationCard.get_rank_int(eval_card)
        suit = EvaluationCard.get_suit_int(eval_card)
        if suit == 1:
            small_suit = 0
        elif suit == 2:
            small_suit = 1
        elif suit == 4:
            small_suit = 2
        elif suit == 8:
            small_suit = 3
        return rank * 4 + small_suit

    def _get_merged_index(self, index_values: Sequence[int]) -> int:
        index = 0
        for i, value in enumerate(index_values):
            index += (self._get_small_int(value) + 1) * (64 ** i)
        return index

    def __getitem__(self, index: Sequence[int]):
        return self.lookup_table[self._get_merged_index(index)]

    def __setitem__(self, index: Sequence[int], value: any):
        self.lookup_table[self._get_merged_index(index)] = value
