from typing import Sequence, List

import socket
import struct
import time
from itertools import combinations

import numpy as np

from poker_ai.clustering.preflop import compute_preflop_lossy_abstraction
from poker_ai.poker.evaluation.eval_card import EvaluationCard


RETRY_DELAY = 3


def create_starting_hands(low_card_rank: int, high_card_rank: int) -> np.ndarray:
    num_cards = 2
    eval_suits = "shdc"
    eval_ranks: List[str] = [
        EvaluationCard.STR_RANKS[rank - 2]
        for rank in range(low_card_rank, high_card_rank + 1)
    ]
    
    cards = np.array(
        [
            EvaluationCard.new(rank_char + suit_char)
            for suit_char in eval_suits
            for rank_char in eval_ranks
        ],
    )
    cards.sort()
    combos = np.array([c for c in combinations(cards, num_cards)])

    return combos


class LightBuilder:
    def __init__(self, low_card_rank: int, high_card_rank: int):
        self.starting_hands = create_starting_hands(low_card_rank, high_card_rank)


class ClusterRequester:
    def __init__(self, client):
        self.client = client
    
    def __getitem__(self, cards: Sequence[int]):
        while True:
            try:
                return self.client.request_cluster(cards)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Failed to request the cluster data. Will wait for {RETRY_DELAY} seconds and retry.")
                time.sleep(RETRY_DELAY)
                self.client.connect()


class LookupClient:
    def __init__(self, server_uri: str, low_card_rank: int, high_card_rank: int):
        host, port = server_uri.strip("lut://").split(":")
        self.host = host
        self.port = int(port)
        
        builder = LightBuilder(low_card_rank, high_card_rank)
        self.preflop = compute_preflop_lossy_abstraction(builder)
        self.requester = ClusterRequester(self)

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.host, self.port)
        self.client_socket.connect(server_address)
    
    def request_cluster(self, cards: Sequence[int]):
        data = struct.pack("".join(["!"] + (["i"] * len(cards)) + ["i"]), *cards, -1)

        self.client_socket.sendall(data)

        response = self.client_socket.recv(4)
        result = struct.unpack("!i", response)[0]

        return result
    
    def __getitem__(self, stage: str):
        if stage in ["flop", "turn", "river"]:
            return self.requester
        elif stage == "pre_flop":
            return self.preflop
        else:
            raise KeyError(f"Unavailble item {stage}.")
