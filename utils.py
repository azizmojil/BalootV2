import numpy as np

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
        """Encodes a card tuple (rank, suit) into a single integer."""
        if card is None or not isinstance(card, (tuple, list)) or len(card) != 2:
            return -1 # Represents no card or invalid card
        suit, rank = card
        if rank not in rank_map or suit not in suit_map:
            return -1
        return rank_map[rank] * 4 + suit_map[suit]

    # Encode hands (assuming 8 cards per hand at the start)
    all_hands = [encode_card(c) for hand in env.hands for c in hand]
    # Pad hands to a fixed size in case the number of cards changes
    expected_hand_cards = 32 # 4 players * 8 cards
    all_hands.extend([-1] * (expected_hand_cards - len(all_hands)))

    # Encode bidding info
    bidding_info = [
        env.bidding_round,
        env.final_bid if env.final_bid is not None else -1,
        env.buyer if env.buyer is not None else -1,
        ds_map.get(env.doubling_state, -1)
    ]

    # Encode trick info (4 cards per trick)
    trick_info = [encode_card(c) for c in env.current_trick]
    trick_info.extend([-1] * (4 - len(trick_info)))

    # Encode game info
    game_info = [
        env.dealer if env.dealer is not None else -1,
        env.current_agent,
        phase_map.get(env.phase, -1),
        gt_map.get(env.game_type, -1),
        suit_map.get(env.trump_suit, -1),
        encode_card(env.face_up)
    ]

    # Scores
    scores = env.cumulative_scores

    # Concatenate all features into a single vector
    return np.concatenate([
        np.array(all_hands, dtype=np.float32),
        np.array(bidding_info, dtype=np.float32),
        np.array(trick_info, dtype=np.float32),
        np.array(game_info, dtype=np.float32),
        np.array(scores, dtype=np.float32)
    ])
