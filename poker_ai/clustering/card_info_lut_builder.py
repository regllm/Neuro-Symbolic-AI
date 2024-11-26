import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures
import multiprocessing
import math

import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction

log = logging.getLogger("poker_ai.clustering.runner")


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    """

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        super().__init__(
            low_card_rank, high_card_rank, save_dir
        )
        card_info_lut_filename = f"card_info_lut_{low_card_rank}_to_{high_card_rank}.joblib"
        centroid_filename = f"centroids_{low_card_rank}_to_{high_card_rank}.joblib"
        self.card_info_lut_path: Path = Path(save_dir) / card_info_lut_filename
        self.centroid_path: Path = Path(save_dir) / centroid_filename

        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        if "pre_flop" not in self.card_info_lut:
            self.card_info_lut["pre_flop"] = compute_preflop_lossless_abstraction(
                builder=self
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
        if "river" not in self.card_info_lut:
            self.load_river()
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters,
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        if "turn" not in self.card_info_lut:
            self.load_turn()
            self.card_info_lut["turn"] = self._compute_turn_clusters(n_turn_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        if "flop" not in self.card_info_lut:
            self.load_flop()
            self.card_info_lut["flop"] = self._compute_flop_clusters(n_flop_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    # def _river_ehs_iter(self):
    #     river_ehs = open(self.ehs_flop_csv_path, "w")
    #     for line in river_ehs:
    #         yield [float(x) for x in line.split(",")]
    #     river_ehs.close()

    def dummy(self, x):
        return np.zeros(3)

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start = time.time()

        ## Original
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     self._river_ehs = list(
        #         tqdm(
        #             executor.map(
        #                 self.process_river_ehs,
        #                 self.river,
        #                 # chunksize=len(self.river) // 160,
        #                 chunksize=1,  # To prevent memory overflow
        #             ),
        #             total=len(self.river),
        #             ascii=" >=",
        #         )
        #     )

        # self._river_ehs = []
        # for e in tqdm(self.river, ascii=" >="):
        #     if len(river_batch) < batch_size:
        #         river_batch.append(e)
        #     self._river_ehs.append(self.process_river_ehs(e))
        
        ## Mini batched
        # batch_size = 960
        # loop_count = len(self.river) // batch_size + 1
        # self._river_ehs = np.zeros(len(self.river))
        # for i in tqdm(range(loop_count), ascii=" >="):
        #     start = i * batch_size
        #     end = min(len(self.river), start + batch_size)
        #     river_batch = self.river[start:end]
        #     with concurrent.futures.ProcessPoolExecutor() as executor:
        #         results = executor.map(
        #             self.process_river_ehs,
        #             river_batch,
        #             max_workers=96,
        #         )
        #         self._river_ehs[start:end] = results
        
        ## File-based
        # if not os.path.exists(self.ehs_flop_csv_path):
        #     river = open(self.card_combos_river_csv_path, "r")
        #     river_ehs = open(self.ehs_flop_csv_path, "w")
        #     river_size = math.comb(len(self._cards), 2) * math.comb(len(self._cards) - 2, 5)
            
        #     with multiprocessing.Pool() as pool:
        #         for ehs in tqdm(
        #             pool.imap(self.process_river_ehs, river, chunksize=960),
        #             ascii=" >=",
        #             total=river_size,
        #         ):
        #             river_ehs.write(",".join([float(x) for x in ehs]))
            
        #     river.close()
        #     river_ehs.close()

        river_size = math.comb(len(self._cards), 2) * math.comb(len(self._cards) - 2, 5)
        try:
            self._river_ehs = joblib.load(self.ehs_river_path)
            log.info("loaded river ehs")
        except FileNotFoundError:
            with tqdm(total=river_size) as pbar:
                self._river_ehs = np.zeros((river_size, 3), dtype=int)
                batch_size = 100
                cursor = 0
                while True:
                    batch_cursor = 0
                    river_batch = np.zeros((batch_size, 7), dtype=int)
                    for _ in range(batch_size):
                        try:
                            river_batch[batch_cursor] = next(self.river)
                            batch_cursor += 1
                        except StopIteration:
                            break
                    if batch_cursor < batch_size:
                        river_batch = river_batch[:batch_cursor]
                    
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        # batch_result = executor.map(
                        #     self.process_river_ehs, river_batch, chunksize=9600
                        # )
                        batch_result = executor.map(
                            self.dummy, river_batch, chunksize=9600
                        )
                        for i, v in batch_result:
                            self._river_ehs[cursor + i] = v
                        cursor += batch_cursor
                        pbar.update(batch_cursor)
                    if batch_cursor < batch_size:
                        break
                joblib.dump(self._river_ehs, self.ehs_river_path)
        
        self.centroids["river"], self._river_clusters = self.cluster(
            num_clusters=n_river_clusters, X=tqdm(self._river_ehs, total=river_size, ascii=" >=")
        )
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._turn_ehs_distributions = list(
                tqdm(
                    executor.map(
                        self.process_turn_ehs_distributions,
                        self.turn,
                        chunksize=len(self.turn) // 160,
                    ),
                    total=len(self.turn),
                    ascii=" >=",
                )
            )
        self.centroids["turn"], self._turn_clusters = self.cluster(
            num_clusters=n_turn_clusters, X=self._turn_ehs_distributions
        )
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_flop_clusters: int):
        """Compute flop clusters and create lookup table."""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._flop_potential_aware_distributions = list(
                tqdm(
                    executor.map(
                        self.process_flop_potential_aware_distributions,
                        self.flop,
                        chunksize=len(self.flop) // 160,
                    ),
                    total=len(self.flop),
                    ascii=" >=",
                )
            )
        self.centroids["flop"], self._flop_clusters = self.cluster(
            num_clusters=n_flop_clusters, X=self._flop_potential_aware_distributions
        )
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    def simulate_get_ehs(self, game: GameUtility,) -> np.ndarray:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand

        Returns
        -------
        ehs : np.ndarray
            [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(self.n_simulations_river):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / self.n_simulations_river
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: np.ndarray,
        the_board: np.ndarray,
        our_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.ndarray
            Array of available cards on the turn
        the_board : np.nearray
            The board as of the turn
        our_hand : np.ndarray
            Cards our hand (Card)

        Returns
        -------
        turn_ehs_distribution : np.ndarray
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """
        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        # sample river cards and run a simulation
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            # get EMD for expected hand strength against each river centroid
            # to which does it belong?
            for idx, river_centroid in enumerate(self.centroids["river"]):
                emd = wasserstein_distance(ehs, river_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # now increment the cluster to which it belongs -
            turn_ehs_distribution[min_idx] += 1 / self.n_simulations_turn
        return turn_ehs_distribution

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Get the expected hand strength for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Expected hand strength
        """
        our_hand = public[:2]
        board = public[2:7]
        # Get expected hand strength
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """
        Get all cards that are available.

        Parameters
        ----------
        cards : np.ndarray
        unavailable_cards : np.array
            Cards that are not available.

        Returns
        -------
            Available cards
        """
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Get the potential aware turn distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware turn distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2],
        )
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Get the potential aware flop distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware flop distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = np.zeros(len(self.centroids["turn"]))
        for j in range(self.n_simulations_flop):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            # getting available cards
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand,
            )
            for idx, turn_centroid in enumerate(self.centroids["turn"]):
                # earth mover distance
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # Now increment the cluster to which it belongs.
            potential_aware_distribution_flop[min_idx] += 1 / self.n_simulations_flop
        return potential_aware_distribution_flop

    @staticmethod
    def cluster(num_clusters: int, X: np.ndarray):
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        # Centers to be used for r - 1 (ie; the previous round)
        centroids = km.cluster_centers_
        return centroids, y_km

    @staticmethod
    def create_card_lookup(clusters: np.ndarray, card_combos: np.ndarray) -> Dict:
        """
        Create lookup table.

        Parameters
        ----------
        clusters : np.ndarray
            Array of cluster ids.
        card_combos : np.ndarray
            The card combos to which the cluster ids belong.

        Returns
        -------
        lossy_lookup : Dict
            Lookup table for finding cluster ids.
        """
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
