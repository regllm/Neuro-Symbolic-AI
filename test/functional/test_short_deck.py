from typing import Tuple, Union

import pytest
import numpy as np

from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.poker.pot import Pot
from pluribus.utils.random import seed


def _new_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
) -> Tuple[ShortDeckPokerState, Pot]:
    """Create a new game."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=initial_chips)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(
        players=players,
        load_pickle_files=False,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    return state, pot


def test_short_deck_1():
    """Test the short deck poker game state works as expected."""
    n_players = 3
    state, _ = _new_game(n_players=n_players)
    # Call for all players.
    player_i_order = [2, 0, 1]
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    assert state._betting_stage == "flop"
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state._betting_stage == "terminal"


def test_short_deck_2():
    """Test the short deck poker game state works as expected."""
    n_players = 3
    state, _ = _new_game(n_players=3)
    player_i_order = [2, 0, 1]
    # Call for all players.
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "turn"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state._betting_stage == "turn"
        state = state.apply_action(action_str="call")
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "river"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state._betting_stage == "terminal"


@pytest.mark.parametrize(
    "n_players",
    [
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(1, marks=pytest.mark.xfail),
        2,
        3,
        4,
    ],
)
def test_short_deck_3(n_players: int):
    """Check the state fails when the wrong number of players are provided.

    Test the short deck poker game state works as expected - make sure the
    order of the players is correct - for the pre-flop it should be
    [-1, -2, 0, 1, ..., -3].
    """
    state, _ = _new_game(n_players=n_players)
    order = list(range(n_players))
    player_i_order = {
        "pre_flop": order[2:] + order[:2],
        "flop": order,
        "turn": order,
        "river": order,
    }
    prev_stage = ""
    while state._betting_stage in player_i_order:
        if state._betting_stage != prev_stage:
            # If there is a new betting stage, reset the target player index
            # counter.
            order_i = 0
            prev_stage = state._betting_stage
        target_player_i = player_i_order[state._betting_stage][order_i]
        assert (
            state.current_player.name == f"player_{target_player_i}"
        ), f"{state.current_player.name} != player_{target_player_i}"
        assert (
            state.player_i == target_player_i
        ), f"{state.player_i} != {target_player_i}"
        # All players call to keep things simple.
        state = state.apply_action("call")
        order_i += 1


@pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("small_blind", [50, 200])
@pytest.mark.parametrize("big_blind", [100, 1000])
def test_pre_flop_pot(n_players: int, small_blind: int, big_blind: int):
    """Test preflop the state is set up for player 2 to start betting."""
    state, pot = _new_game(
        n_players=n_players, small_blind=small_blind, big_blind=big_blind,
    )
    n_bet_chips = sum(p.n_bet_chips for p in state.players)
    target = small_blind + big_blind
    assert state.player_i == 0 if n_players == 2 else 2
    assert state._betting_stage == "pre_flop"
    assert (
        n_bet_chips == target
    ), f"small and big blind have not bet! {n_bet_chips} == {target}"
    assert (
        n_bet_chips == pot.total
    ), f"small and big blind have are not in pot! {n_bet_chips} == {pot.total}"


@pytest.mark.parametrize("n_players", [2, 3])
def test_call_action_sequence(n_players):
    """
    Make sure we never see an action sequence of "raise", "call", "call" in the same
    round with only two players. There would be a similar analog for more than two players,
    but this should aid in initially finding the bug.
    """
    seed(42)
    # example of a bad sequence in a two-handed game in one round
    bad_seq = ["raise", "call", "call"]
    for _ in range(1000):
        state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)
        betting_round_dict = {"pre_flop": [], "flop": [], "turn": [], "river": []}
        while state._betting_stage not in {"show_down", "terminal"}:
            uniform_probability = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), uniform_probability)
            random_action = np.random.choice(state.legal_actions, p=probabilities)
            betting_stage = state._betting_stage
            if state._poker_engine.n_active_players == 2:
                betting_round_dict[betting_stage].append(random_action)
                no_fold_action_history = [
                    action
                    for action in betting_round_dict[betting_stage]
                    if action != "skip"
                ]
                # Loop through the action history and make sure the bad
                # sequence has not happened.
                for i in range(len(no_fold_action_history)):
                    history_slice = no_fold_action_history[i : i + len(bad_seq)]
                    assert history_slice != bad_seq
            state = state.apply_action(random_action)
