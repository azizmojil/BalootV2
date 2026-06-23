import numpy as np
from env.constants import TARGET_SCORE

def flatten_obs(obs_dict):
    """Flattens the observation dictionary into a single numpy array."""
    return np.concatenate([v.ravel()
                           for k, v in obs_dict.items()
                           if k != 'action_mask'])

def get_global_state(env):
    """Aggregates all necessary information into a single numerical global state vector."""
    
    # Mappings for categorical data
    suit_map = {None: -1, '♠': 0, '♥': 1, '♦': 2, '♣': 3}
    rank_map = {None: -1, '7': 0, '8': 1, '9': 2, '10': 3, 'J': 4, 'Q': 5, 'K': 6, 'A': 7}
    phase_map = {'bidding': 0, 'playing': 1}
    gt_map = {None: 0, 'Sun': 1, 'Hukoom': 2}
    ds_map = {None: 0, 'Double': 1, 'Three': 2, 'Four': 3, 'Gahwa': 4}

    def encode_card(card):
        """Encodes a card tuple into a normalized scalar; 0 represents no/unknown card."""
        if card is None or not isinstance(card, (tuple, list)) or len(card) != 2:
            return 0.0
        suit, rank = card
        if rank not in rank_map or suit not in suit_map:
            return 0.0
        return ((rank_map[rank] * 4 + suit_map[suit]) + 1) / 32.0

    def encode_player(player):
        """Encodes a player id into a normalized scalar; 0 represents no player."""
        return 0.0 if player is None else (player + 1) / 4.0

    def encode_action(action):
        """Encodes a bidding action into a normalized scalar; 0 represents no bid."""
        return 0.0 if action is None else (action - 31) / 11.0

    def encode_suit(suit):
        """Encodes suits distinctly from no-trump and invalid values."""
        if suit is None:
            return 0.0
        if suit not in suit_map:
            return 1.0
        return (suit_map[suit] + 1) / 5.0

    # Encode hands (assuming 8 cards per hand at the start)
    all_hands = [encode_card(c) for hand in env.hands for c in hand]
    # Pad hands to a fixed size in case the number of cards changes
    expected_hand_cards = 32 # 4 players * 8 cards
    all_hands.extend([0.0] * (expected_hand_cards - len(all_hands)))

    # Encode bidding info
    bidding_info = [
        min(env.bidding_round, 2) / 2.0,
        min(env.pass_count, 8) / 8.0,
        encode_action(env.initial_bid),
        encode_action(env.final_bid),
        encode_player(env.buyer),
        ds_map.get(env.doubling_state, 0) / 4.0
    ]

    # Encode trick info (4 cards per trick)
    trick_info = [encode_card(c) for c in env.current_trick]
    trick_info.extend([0.0] * (4 - len(trick_info)))

    # Encode game info
    game_info = [
        encode_player(env.dealer),
        encode_player(env.current_agent),
        encode_player(env.trick_leader),
        phase_map.get(env.phase, 0),
        gt_map.get(env.game_type, 0) / 2.0,
        encode_suit(env.trump_suit),
        encode_card(env.face_up)
    ]

    # Scores
    scores = np.clip(np.array(env.cumulative_scores, dtype=np.float32) / TARGET_SCORE, 0.0, 1.0)

    # Concatenate all features into a single vector
    return np.concatenate([
        np.array(all_hands, dtype=np.float32),
        np.array(bidding_info, dtype=np.float32),
        np.array(trick_info, dtype=np.float32),
        np.array(game_info, dtype=np.float32),
        np.array(scores, dtype=np.float32)
    ])
