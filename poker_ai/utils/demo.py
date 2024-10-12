import logging
import random

import numpy as np
import joblib

from poker_ai.clustering.lookup_client import LookupClient
from poker_ai.games.short_deck.state import new_game
from poker_ai.utils.algos import rotate_list

logger = logging.getLogger("poker_ai.utils.demo")


def player_to_str(player, name, hidden=True):
    chunks = []
    turn_char = " "
    if player.is_turn:
        turn_char = "*"
    elif player.is_broke:
        turn_char = "∅"
    elif not player.is_active:
        turn_char = "#"
    chunks.append(f"[{name:^10}]{turn_char}")
    if hidden:
        chunks.append("CARD: [---][---]")
    else:
        chunks.append("CARD: " + "".join([card.to_pretty() for card in player.cards]))
    chunks.append(f"POT: {player.n_bet_chips:>6}")
    chunks.append(f"BANK: {player.n_chips:>6}")
    if player.is_broke:
        chunks.append("<BROKE>")
    else:
        if player.is_small_blind:
            chunks.append("<SMALL BLIND>")
        if player.is_big_blind:
            chunks.append("<BIG BLIND>")
        if player.is_dealer:
            chunks.append("<DEALER>")
        if not player.is_active:
            chunks.append("<FOLDED>")
    return " ".join(chunks)


def player_to_dict(player, name, hidden=True):
    return {
        "name": name,
        "folded": not player.is_active,
        "broke": player.is_broke,
        "is_turn": player.is_turn,
        "cards": None if hidden else [card.to_pair() for card in player.cards],
        "pot": player.n_bet_chips,
        "bank": player.n_chips,
        "is_small_blind": player.is_small_blind,
        "is_big_blind": player.is_big_blind,
        "is_dealer": player.is_dealer,
    }


def state_to_str(state, players, names, client_player_name):
    lines = []
    lines.append("[TABLE] " + "".join([card.to_pretty() for card in state.community_cards]))
    lines.append(f"[POT] {state._table.pot.total}")
    lines.append("----------------")
    for player, name in zip(players, names):
        is_client = player.name == client_player_name
        hidden = not state.is_terminal and not is_client
        lines.append(player_to_str(player, name, hidden=hidden))
    return "\n".join(lines)


def state_to_dict(state, players, names, client_player_name):
    player_dicts = []
    client_player_n_chips = None
    for player, name in zip(players, names):
        is_client = player.name == client_player_name
        hidden = not state.is_terminal and not is_client
        player_dicts.append(player_to_dict(player, name, hidden=hidden))
        if is_client:
            client_player_n_chips = player.n_chips
    actions = get_available_actions(state)
    if client_player_n_chips == 0:
        actions = [action for action in actions if action != "raise"]
    return {
        "publics": [card.to_pair() for card in state.community_cards],
        "players": player_dicts,
        "pot": state._table.pot.total,
        "is_terminal": state.is_terminal,
        "is_waiting": is_waiting(state, client_player_name),
        "actions": actions,
        "text": state_to_str(state, players, names, client_player_name),
    }


def is_waiting(state, client_player_name):
    return state.is_terminal or state.current_player.name == client_player_name


def get_available_actions(state):
    if state.is_terminal:
        return ["new"]
    else:
        return state.legal_actions


def calc_action(state, strategy=None):
    if strategy is None:
        action = random.choice(state.legal_actions)
    else:
        default_strategy = {action: 1 / len(state.legal_actions) for action in state.legal_actions}
        this_state_strategy = strategy.get(state.info_set, default_strategy)
        # Normalizing the strategy.
        total = sum(this_state_strategy.values())
        this_state_strategy = {
            k: v / total for k, v in this_state_strategy.items()
        }
        actions = list(this_state_strategy.keys())
        probabilties = list(this_state_strategy.values())
        action = np.random.choice(actions, p=probabilties)
    return action


def load_strategy(strategy_path):
    strategy_dict = joblib.load(strategy_path)
    return strategy_dict['strategy']


def load_lut(lut_path: str, low_card_rank: int = 2, high_card_rank: int = 14):
    if lut_path.startswith("lut://"):
        card_info_lut = LookupClient(
            lut_path,
            low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
        )
        card_info_lut.connect()
        return card_info_lut
    else:
        filename = f"card_info_lut_{low_card_rank}_to_{high_card_rank}.joblib"
        return joblib.load(lut_path + "/" + filename)


class PokerDemo:
    def __init__(
        self,
        n_players=6,
        low_card_rank=2,
        high_card_rank=14,
        lut=None,
        strategy=None,
        include_dumb_players=False,
    ):
        # Set configurations for the game.
        self.n_players = n_players
        self.player_rotation_key = 0
        self.random_agent = strategy is None
        self.strategy = strategy
        self.lut = lut if lut is not None else {}
        self.low_card_rank = low_card_rank
        self.high_card_rank = high_card_rank
        self.include_dumb_players = include_dumb_players
        if include_dumb_players:
            self.names = [f"Player {i + 1}" for i in range(n_players - 3)] + ["Agent 1", "Agent 2", "You"]
        else:
            self.names = [f"Player {i + 1}" for i in range(n_players - 1)] + ["You"]

        # Initialize the event log list.
        self.events = []
        self.state_dicts = []

        # Initilize the game state.
        self._init_state()

        # Play until the player input is needed.
        self.play()

    def _get_players(self):
        return rotate_list(self.state.players[::-1], self.player_rotation_key)
    
    def _rotate_players(self):
        self.player_rotation_key -= 1
        if self.player_rotation_key < 0:
            self.player_rotation_key = self.n_players - 1
        players = self._get_players()
        self.player_name_dict = {
            player.name: name
            for player, name in zip(players, self.names)
        }
        self.client_player_name = players[-1].name
        self.agent_player_names = [players[-3].name, players[-2].name]

    def _add_event(self, action, raw_player_name=None):
        player_name = None
        if raw_player_name is not None:
            player_name = self.player_name_dict[raw_player_name]
        self.events.append({
            "player": player_name,
            "action": action,
        })
        self.state_dicts.append(self.to_dict())

    def _init_state(self):
        include_ranks = list(range(self.low_card_rank, self.high_card_rank + 1))
        self.state = new_game(
            self.n_players,
            self.lut,
            load_card_lut=False,
            include_ranks=include_ranks,
            handle_all_in=True,
            allow_fourth_bet=True,
        )
        self._rotate_players()
        self._add_event("new")
    
    def _reset_state(self):
        chip_counts = [player.n_chips for player in self._get_players()]
        include_ranks = list(range(self.low_card_rank, self.high_card_rank + 1))
        self.state = new_game(
            self.n_players,
            self.lut,
            load_card_lut=False,
            include_ranks=include_ranks,
            without_blinds=True,
            handle_all_in=True,
            allow_fourth_bet=True,
        )
        self._rotate_players()
        for player, chip_count in zip(self._get_players(), chip_counts):
            missing_chip_count = self.state._initial_n_chips - player.n_chips
            player.n_chips = chip_count - missing_chip_count
        self.state.flag_broke_players()
        self.state.smart_assign_blinds()
        self.state.skip_players_with_no_chips()
        self._add_event("new")

    def _get_winner(self, prev_chip_counts, chip_counts):
        for i, (prev, curr) in enumerate(zip(prev_chip_counts, chip_counts)):
            if curr > prev:
                return self._get_players()[i]
        return None

    def _apply_action(self, action):
        raw_player_name = self.state.current_player.name
        prev_chip_counts = [player.n_chips for player in self._get_players()]
        self.state = self.state.apply_action(action)
        chip_counts = [player.n_chips for player in self._get_players()]
        winner = self._get_winner(prev_chip_counts, chip_counts)
        if winner is not None:
            winner_name = self.player_name_dict[winner.name]
            augmented_action = f"{action} - {winner_name} won!"
        else:
            augmented_action = action
        self._add_event(augmented_action, raw_player_name)

    def _calc_action_and_play(self):
        if self.include_dumb_players and self.state.current_player.name in self.agent_player_names:
            action = calc_action(self.state, None)
        else:    
            action = calc_action(self.state, self.strategy)
        self._apply_action(action)

    def to_dict(self):
        return state_to_dict(self.state, self._get_players(), self.names, self.client_player_name)

    def read_events(self):
        event_dicts = []
        for event, state_dict in zip(self.events, self.state_dicts):
            event_dicts.append({
                "player": event["player"],
                "action": event["action"],
                "state": state_dict,
            })
        self.events = []
        self.state_dicts = []
        return event_dicts

    def is_waiting(self):
        return is_waiting(self.state, self.client_player_name)

    def play(self, action=None):
        if action is not None:
            if not self.is_waiting():
                raise ValueError("Client action is not applicable now")
            elif action == "new":
                # TODO: Save the player action history and turns.
                self._reset_state()
            else:
                self._apply_action(action)

        while not self.is_waiting():
            self._calc_action_and_play()