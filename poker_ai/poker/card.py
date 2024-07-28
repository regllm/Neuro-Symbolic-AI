from typing import Dict, List, Set, Union

from poker_ai.poker.evaluation.eval_card import EvaluationCard


def get_all_suits() -> Set[str]:
    """Get set of suits that the card can take on."""
    return {"spades", "diamonds", "clubs", "hearts"}


def get_all_ranks() -> List[str]:
    """Get the list of ranks the card could be."""
    return [
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "jack",
        "queen",
        "king",
        "ace",
    ]


class Card:
    """Card to represent a poker card."""

    def __init__(self, rank: Union[str, int], suit: str):
        """Instanciate the card."""
        if not isinstance(rank, (int, str)):
            raise ValueError(f"rank should be str/int but was: {type(rank)}.")
        elif isinstance(rank, str):
            rank = self._str_to_rank(rank)
        if rank < 2 or rank > 14:
            raise ValueError(
                f"rank should be between 2 and 14 (inclusive) but was {rank}"
            )
        if suit not in get_all_suits():
            raise ValueError(f"suit {suit} must be in {get_all_suits()}")
        self._card_id = self._rank_and_suit_to_card_id(rank, suit)

    def __repr__(self):
        """Pretty printing the object."""
        icon = self._suit_to_icon(self.suit)
        return f"<Card card=[{self.rank} of {self.suit} {icon}]>"

    def __int__(self):
        return self.eval_card

    def __lt__(self, other):
        return self.rank < other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __le__(self, other):
        return self.rank <= other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __gt__(self, other):
        return self.rank > other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __ge__(self, other):
        return self.rank >= other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return int(self) != int(other)

    def __hash__(self):
        return hash(int(self))

    @property
    def eval_card(self) -> EvaluationCard:
        rank, suit = self._card_id_to_rank_and_suit(self._card_id)
        rank_char = self._rank_to_char(rank)
        suit_char = self.suit.lower()[0]
        return EvaluationCard.new(f"{rank_char}{suit_char}")

    @property
    def rank_int(self) -> int:
        """Get the rank as an int"""
        return self._card_id % 16

    @property
    def rank(self) -> str:
        """Get the rank as a string."""
        return self._rank_to_str(self._card_id % 16)

    @property
    def suit(self) -> str:
        """Get the suit."""
        suit_dict = {
            0: "spades",
            1: "diamonds",
            2: "clubs",
            3: "hearts",
        }
        return suit_dict[int(self._card_id // 15)]

    def _rank_and_suit_to_card_id(self, rank, suit):
        suit_dict = {
            "spades": 0,
            "diamonds": 1,
            "clubs": 2,
            "hearts": 3,
        }
        suit_index = suit_dict[suit]
        return suit_index * 16 + rank

    def _card_id_to_rank_and_suit(self, card_id):
        suit_dict = {
            0: "spades",
            1: "diamonds",
            2: "clubs",
            3: "hearts",
        }
        rank = card_id % 16
        suit = suit_dict[int(card_id // 15)]
        return rank, suit

    def _str_to_rank(self, string: str) -> int:
        """Convert the string rank to the integer rank."""
        return {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "jack": 11,
            "queen": 12,
            "king": 13,
            "ace": 14,
            "t": 10,
            "j": 11,
            "q": 12,
            "k": 13,
            "a": 14,
        }[string.lower()]

    def _rank_to_str(self, rank: int) -> str:
        """Convert the integer rank to the string rank."""
        return {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
            11: "jack",
            12: "queen",
            13: "king",
            14: "ace",
        }[rank]

    def _rank_to_char(self, rank: int) -> str:
        """Convert the int rank to char used by the `EvaluationCard` object."""
        return {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "T",
            11: "J",
            12: "Q",
            13: "K",
            14: "A",
        }[rank]

    def _suit_to_icon(self, suit: str) -> str:
        """Icons for pretty printing."""
        return {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}[suit]

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Turn into dict."""
        rank, suit = self._card_id_to_rank_and_suit(self._card_id)
        return dict(rank=rank, suit=suit)

    @staticmethod
    def from_dict(x: Dict[str, Union[int, str]]):
        """From dict turn into class."""
        if set(x) != {"rank", "suit"}:
            raise NotImplementedError(f"Unrecognised dict {x}")
        return Card(rank=x["rank"], suit=x["suit"])
    