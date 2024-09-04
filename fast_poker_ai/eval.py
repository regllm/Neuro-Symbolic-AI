STR_RANKS = "23456789TJQKA"
INT_RANKS = range(13)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

CHAR_RANK_TO_INT_RANK = dict(zip(list(STR_RANKS), INT_RANKS))
CHAR_SUIT_TO_INT_SUIT = {
    "s": 1,  # spades
    "h": 2,  # hearts
    "d": 4,  # diamonds
    "c": 8,  # clubs
}


def new(string):
    """
    Converts EvaluationCard string to binary integer representation of card, inspired by:

    http://www.suffecool.net/poker/evaluator.html
    """

    rank_char = string[0]
    suit_char = string[1]
    rank_int = CHAR_RANK_TO_INT_RANK[rank_char]
    suit_int = CHAR_SUIT_TO_INT_SUIT[suit_char]
    rank_prime = PRIMES[rank_int]

    bitrank = 1 << rank_int << 16
    suit = suit_int << 12
    rank = rank_int << 8

    return bitrank | suit | rank | rank_prime
