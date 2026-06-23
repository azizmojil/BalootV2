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
REWARD_ALL_PASS_PENALTY = -0.1 # Penalty if a round fails because everyone passed


SUN_CARD_POINTS = {'A': 11, '10': 10, 'K': 4, 'Q': 3, 'J': 2, '9': 0, '8': 0, '7': 0}
HUKOOM_CARD_POINTS = {'J': 20, '9': 14, 'A': 11, '10': 10, 'K': 4, 'Q': 3, '8': 0, '7': 0}
BIDDING_SET_STRENGTH_BONUS = {
    "Sera": 20, "Khamseen": 50, "Mia_c": 100, "Mia_s": 100, "Arbamia": 200
}
BIDDING_STRONG_HAND_THRESHOLD = 50
BIDDING_MONSTER_HAND_THRESHOLD = 80
BIDDING_WEAK_HAND_THRESHOLD = 35
BIDDING_CLOSE_CHOICE_MARGIN = 8
BIDDING_TRUMP_JACK_BONUS = 20
BIDDING_TRUMP_NINE_BONUS = 10
BIDDING_SUIT_ACTIONS = {34: '♠', 35: '♥', 36: '♦', 37: '♣'}


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


def _get_augmented_bidding_hand(env, agent_id):
    hand = list(env.hands[agent_id])
    face_up = env.face_up
    if face_up is not None and face_up not in hand:
        hand.append(face_up)
    return hand


def _get_detected_sets(hand):
    from env.utils import detect_sets
    return detect_sets(hand)


def _set_strength_bonus(hand):
    return sum(BIDDING_SET_STRENGTH_BONUS.get(s["type"], 0) for s in _get_detected_sets(hand))


def _calculate_sun_bidding_strength(hand):
    card_points = sum(SUN_CARD_POINTS.get(rank, 0) for suit, rank in hand)
    return card_points + _set_strength_bonus(hand)


def _calculate_hukoom_bidding_strength(hand, trump_suit):
    score = sum(get_card_points(card, "Hukoom", trump_suit) for card in hand)
    score += _set_strength_bonus(hand)
    if (trump_suit, "J") in hand:
        score += BIDDING_TRUMP_JACK_BONUS
    if (trump_suit, "9") in hand:
        score += BIDDING_TRUMP_NINE_BONUS
    return score


def _calculate_bidding_strengths(env, agent_id):
    hand = _get_augmented_bidding_hand(env, agent_id)
    hukoom_scores = {
        suit: _calculate_hukoom_bidding_strength(hand, suit)
        for suit in BIDDING_SUIT_ACTIONS.values()
    }
    return {
        "Sun": _calculate_sun_bidding_strength(hand),
        "Hukoom": hukoom_scores,
        "best_hukoom": max(hukoom_scores.values()),
    }


def _reward_for_passing(best_strength):
    if best_strength >= BIDDING_MONSTER_HAND_THRESHOLD:
        return -0.12
    if best_strength >= BIDDING_STRONG_HAND_THRESHOLD:
        return -0.06
    if best_strength < BIDDING_WEAK_HAND_THRESHOLD:
        return 0.03
    return 0.0


def _reward_for_sun_bid(strengths):
    sun_score = strengths["Sun"]
    best_hukoom = strengths["best_hukoom"]
    if sun_score < BIDDING_WEAK_HAND_THRESHOLD:
        return -0.08
    if sun_score >= BIDDING_STRONG_HAND_THRESHOLD:
        if sun_score + BIDDING_CLOSE_CHOICE_MARGIN < best_hukoom:
            return 0.04
        return 0.12
    if best_hukoom >= BIDDING_STRONG_HAND_THRESHOLD:
        return -0.04
    return 0.02


def _reward_for_hukoom_bid(strengths, trump_suit):
    suit_score = strengths["Hukoom"][trump_suit]
    best_hukoom = strengths["best_hukoom"]
    sun_score = strengths["Sun"]
    if suit_score < BIDDING_WEAK_HAND_THRESHOLD:
        return -0.08
    if suit_score + BIDDING_CLOSE_CHOICE_MARGIN < best_hukoom:
        return -0.05
    if suit_score >= BIDDING_STRONG_HAND_THRESHOLD:
        if sun_score > suit_score + BIDDING_CLOSE_CHOICE_MARGIN:
            return 0.04
        return 0.12
    return 0.02


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
    Grades pass, Sun, and Hukoom actions against the hand's Sun/Hukoom potential.
    """
    # Action 38 buys Sun for the acting agent's partner.
    scoring_agent = (agent_id + 2) % 4 if action == 38 else agent_id
    strengths = _calculate_bidding_strengths(env, scoring_agent)

    if action == 32:
        best_strength = max(strengths["Sun"], strengths["best_hukoom"])
        return _reward_for_passing(best_strength)

    if action in (33, 38):
        return _reward_for_sun_bid(strengths)

    if action in BIDDING_SUIT_ACTIONS:
        return _reward_for_hukoom_bid(strengths, BIDDING_SUIT_ACTIONS[action])

    return 0.0


def calculate_end_of_game_reward(env):
    try:
        rewards = np.zeros(4)
        if not env.match_over:
            return rewards

        # Determine winning team based on cumulative scores
        score_delta = env.cumulative_scores[0] - env.cumulative_scores[1]
        if score_delta == 0:
            # Tied matches get no terminal win/loss shaping.
            return rewards
        winning_team_id = 0 if score_delta > 0 else 1

        for player_id in range(4):
            if player_id % 2 == winning_team_id:
                rewards[player_id] = REWARD_WIN_GAME
            else:
                rewards[player_id] = REWARD_LOSE_GAME

        return rewards
    except Exception:
        return np.zeros(4)
