import numpy as np
from gymnasium import spaces
from env.constants import BID_ACTIONS, RANKS, SUITS, TARGET_SCORE
from env.utils.observation_validation import (
    OBSERVATION_SCHEMA,
    OBSERVATION_TRICK_HISTORY_LENGTH,
)
from env.utils.cards import one_hot_index


def require_positive_int(value, name):
    """Converts value to a positive int or raises ValueError with the provided field name."""
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _observation_keys(obs_dict, observation_space=None, exclude=("action_mask",)):
    """Returns observation feature keys in schema order, excluding non-network fields."""
    if observation_space is not None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(f"observation_space must be a gymnasium.spaces.Dict, got {type(observation_space).__name__}")
        space_keys = list(observation_space.spaces.keys())
        if set(space_keys) == set(OBSERVATION_SCHEMA):
            keys = OBSERVATION_SCHEMA.keys()
        else:
            keys = space_keys
    else:
        keys = obs_dict.keys()
    return [key for key in keys if key not in exclude]


def flatten_obs(obs_dict, observation_space=None, exclude=("action_mask",)):
    """Flattens the observation dictionary into a single numpy array."""
    flat_parts = []
    for key in _observation_keys(obs_dict, observation_space, exclude):
        if key not in obs_dict:
            raise KeyError(f"Observation is missing required key '{key}'")
        arr = np.asarray(obs_dict[key], dtype=np.float32)
        if observation_space is not None and arr.shape != observation_space.spaces[key].shape:
            raise ValueError(
                f"Observation '{key}' has shape {arr.shape}, "
                f"expected {observation_space.spaces[key].shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Observation '{key}' contains non-finite values")
        flat_parts.append(arr.ravel())
    if not flat_parts:
        raise ValueError(f"Cannot flatten observation: all keys were excluded (excluded: {exclude})")
    return np.concatenate(flat_parts).astype(np.float32)


def infer_model_dimensions(env, obs_dict=None):
    """Derives MAPPO input and action dimensions from the current environment.

    If obs_dict is omitted, this resets env to sample an observation.
    """
    if obs_dict is None:
        obs_dict = env.reset()
    local_obs = flatten_obs(obs_dict, env.observation_space)
    local_obs_dim = local_obs.shape[0]
    global_state_dim = get_global_state(env).shape[0]
    act_dim = int(env.action_space.n)
    return local_obs_dim, global_state_dim, act_dim

def get_global_state(env):
    """Aggregates critic-only game information into a fixed binary/normalized vector."""

    deck = [(suit, rank) for suit in SUITS for rank in RANKS]
    phase_map = {'bidding': 0, 'playing': 1}
    gt_map = {None: 0, 'Sun': 1, 'Hukoom': 2}
    trump_map = {None: 0, '♠': 1, '♥': 2, '♦': 3, '♣': 4}
    ds_map = {None: 0, 'Double': 1, 'Three': 2, 'Four': 3, 'Gahwa': 4}

    def player_one_hot(player, include_none=False):
        if include_none:
            return one_hot_index(4 if player is None else player, 5)
        if player is None:
            raise ValueError("player cannot be None unless include_none=True")
        return one_hot_index(player, 4)

    def action_one_hot(action, include_none=False):
        size = len(BID_ACTIONS) + (1 if include_none else 0)
        vec = np.zeros(size, dtype=np.float32)
        if action in BID_ACTIONS:
            vec[BID_ACTIONS.index(action)] = 1.0
        elif include_none:
            vec[-1] = 1.0
        return vec

    def card_one_hot(card):
        vec = np.zeros(32, dtype=np.float32)
        if card in deck:
            vec[deck.index(card)] = 1.0
        return vec

    def hand_mask(hand):
        return np.array([1.0 if card in hand else 0.0 for card in deck], dtype=np.float32)

    all_hands = np.concatenate([hand_mask(hand) for hand in env.hands]).astype(np.float32)
    remaining_cards = env.remaining_cards.astype(np.float32)
    played_cards = (1.0 - env.remaining_cards).astype(np.float32)
    current_trick = np.concatenate([card_one_hot(card) for card in env.current_trick]).astype(np.float32)
    last_trick = np.concatenate([card_one_hot(card) for card in env.last_trick]).astype(np.float32)

    trick_history_parts = []
    for entry in list(env.trick_history[-OBSERVATION_TRICK_HISTORY_LENGTH:]):
        order = entry.get("order", range(4))
        trick_history_parts.extend(card_one_hot(entry["cards"][player]) for player in order)
    while len(trick_history_parts) < OBSERVATION_TRICK_HISTORY_LENGTH * 4:
        trick_history_parts.append(np.zeros(32, dtype=np.float32))
    trick_history = np.concatenate(trick_history_parts).astype(np.float32)

    role_context = np.concatenate([
        player_one_hot(env.dealer),
        player_one_hot(env.current_agent),
        player_one_hot(env.trick_leader),
        player_one_hot(env.buyer, include_none=True),
        player_one_hot(env.last_doubler, include_none=True),
    ]).astype(np.float32)

    game_context = np.concatenate([
        one_hot_index(phase_map.get(env.phase, 0), 2),
        one_hot_index(gt_map.get(env.game_type, 0), 3),
        one_hot_index(trump_map.get(env.trump_suit, 0), 5),
        one_hot_index(ds_map.get(env.doubling_state, 0), 5),
        action_one_hot(env.initial_bid, include_none=True),
        action_one_hot(env.final_bid, include_none=True),
    ]).astype(np.float32)

    progress_scores = np.array([
        min(env.bidding_round, 2) / 2.0,
        min(env.pass_count, 8) / 8.0,
        min(env.trick_count, 8) / 8.0,
        np.clip(env.cumulative_scores[0] / TARGET_SCORE, 0.0, 1.0),
        np.clip(env.cumulative_scores[1] / TARGET_SCORE, 0.0, 1.0),
        min(env.team_bant[0], 162) / 162.0,
        min(env.team_bant[1], 162) / 162.0,
        min(env.team_tricks[0], 8) / 8.0,
        min(env.team_tricks[1], 8) / 8.0,
        min(env.round_count, 32) / 32.0,
    ], dtype=np.float32)

    sets_context = np.concatenate([
        (env.declared_sets / 2).astype(np.float32).flatten(),
        (env.revealed_sets / 2).astype(np.float32).flatten(),
        np.array(env.balot, dtype=np.float32),
    ]).astype(np.float32)

    return np.concatenate([
        all_hands,
        remaining_cards,
        played_cards,
        card_one_hot(env.face_up),
        current_trick,
        last_trick,
        trick_history,
        role_context,
        game_context,
        progress_scores,
        sets_context,
    ]).astype(np.float32)
