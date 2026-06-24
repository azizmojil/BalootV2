import numpy as np
from env.constants import BID_ACTIONS, RANKS, SUITS, TARGET_SCORE
from env.utils import one_hot_index

def flatten_obs(obs_dict):
    """Flattens the observation dictionary into a single numpy array."""
    return np.concatenate([v.ravel()
                           for k, v in obs_dict.items()
                           if k != 'action_mask'])

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

    role_context = np.concatenate([
        player_one_hot(env.dealer),
        player_one_hot(env.current_agent),
        player_one_hot(env.trick_leader),
        player_one_hot(env.buyer, include_none=True),
        player_one_hot(getattr(env, "last_doubler", None), include_none=True),
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
        role_context,
        game_context,
        progress_scores,
        sets_context,
    ]).astype(np.float32)
