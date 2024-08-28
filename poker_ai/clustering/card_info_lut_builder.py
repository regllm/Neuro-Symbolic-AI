import ctypes
import logging
import time
import concurrent.futures
import multiprocessing
import math
from pathlib import Path
from typing import Any, Dict
from multiprocessing.shared_memory import SharedMemory

import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction
from poker_ai.poker.evaluation import Evaluator
from poker_ai.utils.safethread import batch_process

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
        self._evaluator = Evaluator()
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

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start = time.time()
        river_ehs_sm = None
        river_size = math.comb(len(self._cards), 2) * math.comb(len(self._cards) - 2, 5)
        try:
            river_ehs = joblib.load(self.ehs_river_path)
            log.info("loaded river ehs")
        except FileNotFoundError:
            # Allocate shared memory for river_ehs
            river_ehs_bytes = river_size * 3 * 8
            river_ehs_sm = SharedMemory(
                name="river_ehs_mem", create=True, size=river_ehs_bytes
            )
            river_ehs = np.ndarray(
                (river_size, 3), dtype=np.double, buffer=river_ehs_sm.buf
            )

            def process_all(batch, cursor, river_size):
                sm = SharedMemory("river_ehs_mem")
                result = np.ndarray(
                    (river_size, 3), dtype=np.double, buffer=sm.buf
                )
                for i, x in enumerate(batch):
                    result[cursor] = self.process_river_ehs(x)
                sm.close()
            
            worker_count = multiprocessing.cpu_count()
            batch_size = min(10_000, river_size // worker_count)
            cursor = 0
            max_batch_seconds = None
            batch_failed = False

            with tqdm(total=river_size, ascii=" >=") as pbar:
                while True:
                    task_done = False
                    
                    if batch_failed:
                        batches = cached_batches
                    else:
                        batches = []
                        for _ in range(worker_count):
                            batch = []
                            for _ in range(batch_size):
                                try:
                                    batch.append(next(self.river))
                                except StopIteration:
                                    task_done = True
                                    break
                            if len(batch) > 0:
                                batches.append(batch)
                    
                    cached_batches = batches
                    batch_failed = False
                    
                    total_batch_size = 0
                    processes = []

                    start = time.time()
                    for batch in batches:
                        process = multiprocessing.Process(
                            target=process_all, args=(batch, cursor, river_size)
                        )
                        process.start()
                        processes.append(process)
                        total_batch_size += len(batch)
                        cursor += len(batch)
                
                    for process in processes:
                        if max_batch_seconds is None:
                            process.join()
                            continue
                        process.join(timeout=max_batch_seconds)

                        if process.is_alive():
                            # Failed to handle this batch.
                            batch_failed = True
                            cursor -= total_batch_size
                            break

                    if batch_failed:
                        continue

                    end = time.time()
                    if max_batch_seconds is None:
                        duration = end - start
                        max_batch_seconds = int(duration * 3)
                        log.info(f"Completed first batch of river ehs in {duration:.2f} seconds.")
                        log.info(f"Setting batch timeout as {max_batch_seconds} seconds.")
                        
                    pbar.update(total_batch_size)
                    
                    if task_done:
                        break
            joblib.dump(river_ehs, self.ehs_river_path)

        self.centroids["river"], self._river_clusters = self.cluster(
            num_clusters=n_river_clusters, X=river_ehs
        )
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        if river_ehs_sm is not None:
            river_ehs_sm.close()
            river_ehs_sm.unlink()
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start = time.time()

        def batch_process(batch, cursor, result):
            for i, x in enumerate(batch):
                ehs = self.process_turn_ehs_distributions(x)
                result_pos = (cursor + i) * 3
                result[result_pos:result_pos + 3] = ehs[0]
                result[result_pos + 1] = ehs[1]
                result[result_pos + 2] = ehs[2]
        
        batch_process(
            self.turn,
            batch_process,
        )


        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     self._turn_ehs_distributions = list(
        #         tqdm(
        #             executor.map(
        #                 self.process_turn_ehs_distributions,
        #                 self.turn,
        #                 chunksize=len(self.turn) // 160,
        #             ),
        #             total=len(self.turn),
        #             ascii=" >=",
        #         )
        #     )
        self._turn_ehs_distributions = []
        for combo in tqdm(self.turn, ascii=" >="):
            self._turn_ehs_distributions.append(
                self.process_turn_ehs_distributions(combo)
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
        public: np.ndarray,
    ) -> np.ndarray:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.ndarray
            Array of available cards on the turn
        public : np.ndarray
            Cards of our hand (public[:2]) and the board as of the turn (public[2:])
            concatenated.

        Returns
        -------
        turn_ehs_distribution : np.ndarray
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """

        available_cards = np.array([c for c in self._cards if c not in public])
        hand_size = len(public) + 1
        hand_evaluator = self._evaluator.hand_size_map[hand_size]

        prob_unit = 1 / self.n_simulations_turn
        prob_sub_unit = 1 / self.n_simulations_river
        ehs: np.ndarray = np.zeros(3)
        our_hand = np.zeros(len(public) + 1, dtype=int)
        our_hand[:len(public)] = public
        opp_hand = np.zeros(len(public) + 1, dtype=int)
        opp_hand[:len(public)] = public

        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        # Sample river cards and run simulations.
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            non_river_cards = np.array([c for c in available_cards if c != river_card])
            our_hand[-1:] = river_card
            opp_hand[-1:] = river_card
            
            for _ in range(self.n_simulations_river):
                opp_hand[:2] = np.random.choice(non_river_cards, 2, replace=False)

                our_hand_rank = hand_evaluator(our_hand)
                opp_hand_rank = hand_evaluator(opp_hand)
                if our_hand_rank > opp_hand_rank:
                    ehs[0] += prob_sub_unit
                elif our_hand_rank < opp_hand_rank:
                    ehs[1] += prob_sub_unit
                else:
                    ehs[2] += prob_sub_unit
            
            # Get EMD for expected hand strength against each river centroid
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
            turn_ehs_distribution[min_idx] += prob_unit
    
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
        # our_hand = public[:2]
        # board = public[2:7]

        our_hand_rank = self._evaluator._seven(public)
        available_cards = np.array([c for c in self._cards if c not in public])
        
        prob_unit = 1 / self.n_simulations_river
        ehs: np.ndarray = np.zeros(3)
        opp_hand = public.copy()
        for _ in range(self.n_simulations_river):
            opp_hand[:2] = np.random.choice(available_cards, 2, replace=False)
            opp_hand_rank = self._evaluator._seven(opp_hand)
            if our_hand_rank > opp_hand_rank:
                ehs[0] += prob_unit
            elif our_hand_rank < opp_hand_rank:
                ehs[1] += prob_unit
            else:
                ehs[2] += prob_unit
        return ehs

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
        unavailable_cards = set(unavailable_cards)
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
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            public,
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
        extended_public = np.zeros(len(public) + 1)
        extended_public[:-1] = public
        for j in range(self.n_simulations_flop):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            extended_public[-1:] = turn_card
            # getting available cards
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                extended_public,
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
