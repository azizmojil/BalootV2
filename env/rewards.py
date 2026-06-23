import numpy as np

# --- Reward Shaping Constants ---
# These values are hyperparameters and can be tuned.

# 1. Game-Level Rewards (The most important signal)
REWARD_WIN_GAME = 10.0
REWARD_LOSE_GAME = -10.0

# 2. Round-Level Rewards (For making/breaking the contract)
REWARD_CONTRACT_SUCCESS = 2.0

# 3. Trick-Level Rewards (Frequent, smaller signals)
REWARD_WIN_TRICK_BASE = 0.05
REWARD_TRICK_POINT_SCALAR = 0.002  # Scales reward with points won

# 4. Action-shaping rewards
REWARD_PASS_PENALTY = -0.02 # Small penalty to discourage always passing
REWARD_BID_SET_BONUS = { # Reward for bidding with a good set in hand
    "Sera": 0.1, "Khamseen": 0.2, "Mia_c": 0.4, "Mia_s": 0.4, "Arbamia": 0.5
}
REWARD_ALL_PASS_PENALTY = -0.1 # Penalty if a round fails because everyone passed


SUN_CARD_POINTS = {'A': 11, '10': 10, 'K': 4, 'Q': 3, 'J': 2, '9': 0, '8': 0, '7': 0}
HUKOOM_CARD_POINTS = {'J': 20, '9': 14, 'A': 11, '10': 10, 'K': 4, 'Q': 3, '8': 0, '7': 0}


def get_card_points(card, game_type, trump_suit):
    """Gets the point value of a single card based on the game type."""
    try:
        suit, rank = card
        if game_type == "Hukoom":
            points_map = HUKOOM_CARD_POINTS if suit == trump_suit else SUN_CARD_POINTS
        else:
            points_map = SUN_CARD_POINTS
        return points_map.get(rank, 0)
    except (TypeError, ValueError):
        # Handles cases where card is None or not a valid tuple
        return 0


def calculate_trick_reward(trick_cards, trick_winner, game_type, trump_suit):
    """
    Calculates a zero-sum reward for all 4 players at the end of a trick.
    The winning team gets a positive reward, the losing team a negative one.
    """
    try:
        rewards = np.zeros(4)
        if trick_winner is None:
            return rewards

        trick_points = sum(get_card_points(c, game_type, trump_suit) for c in trick_cards)
        trick_reward = REWARD_WIN_TRICK_BASE + (trick_points * REWARD_TRICK_POINT_SCALAR)

        for player_id in range(4):
            # Check if the player is on the winning team for this trick
            if player_id % 2 == trick_winner % 2:
                rewards[player_id] = trick_reward
            else:
                rewards[player_id] = -trick_reward

        return rewards
    except Exception:
        # In case of any unexpected error, return a neutral reward to avoid crashing.
        return np.zeros(4)


def calculate_end_of_round_reward(env):
    """
    Calculates a zero-sum reward for all 4 players based on the round's outcome.
    This should be called only once when the round terminates.
    """
    try:
        rewards = np.zeros(4)
        buyer = env.buyer
        if buyer is None:
            return rewards

        buying_team_id = buyer % 2

        # If the buying team scored > 0, they succeeded (since losing results in 0 score for them)
        if env.final_scores[buying_team_id] > 0:
            contract_reward = REWARD_CONTRACT_SUCCESS
        else:
            contract_reward = -REWARD_CONTRACT_SUCCESS

        for player_id in range(4):
            if player_id % 2 == buying_team_id:
                rewards[player_id] = contract_reward
            else:
                rewards[player_id] = -contract_reward

        return rewards
    except Exception:
        return np.zeros(4)


def calculate_bidding_reward(env, agent_id, action):
    """
    Calculates an immediate reward during the bidding phase.
    - Penalizes passing.
    - Rewards bidding when the agent has strong cards.
    """
    high_card_counts = getattr(env, "hand_high_card_counts", None)
    if high_card_counts is not None:
        high_card_count = high_card_counts[agent_id]
    else:
        agent_hand = env.hands[agent_id]
        high_card_count = sum(1 for (suit, rank) in agent_hand if rank in ('A', 'K', 'Q', 'J'))

    # Action 32 is "Pass"
    if action == 32:
        return REWARD_PASS_PENALTY if high_card_count >= 3 else 0.0

    # If the agent made a bid (not a pass), give a small positive signal.
    # We can't use declared_sets_info here because it's not populated until
    # bidding ends. Instead, use a simple heuristic based on high cards.

    # Scale reward by hand quality
    if high_card_count >= 4:
        return 0.12
    elif high_card_count >= 3:
        return 0.10
    elif high_card_count >= 2:
        return 0.03
    else:
        return -0.03


def calculate_end_of_game_reward(env):
    try:
        rewards = np.zeros(4)
        if not env.match_over:
            return rewards

        # Determine winning team based on cumulative scores
        if env.cumulative_scores[0] > env.cumulative_scores[1]:
            winning_team_id = 0
        elif env.cumulative_scores[1] > env.cumulative_scores[0]:
            winning_team_id = 1
        else:
            return rewards

        for player_id in range(4):
            if player_id % 2 == winning_team_id:
                rewards[player_id] = REWARD_WIN_GAME
            else:
                rewards[player_id] = REWARD_LOSE_GAME

        return rewards
    except Exception:
        return np.zeros(4)
