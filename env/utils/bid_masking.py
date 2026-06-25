import numpy as np

from env.utils.cards import team


def compute_bidding_order(dealer, num_agents=4):
    return [(dealer + i) % num_agents for i in range(1, num_agents + 1)]


def can_takweesh(agent_hand, current_bid_type=None, trump_suit=None):
    for suit, rank in agent_hand:
        if rank not in ('7', '8', '9'):
            return False
        if current_bid_type == "Hukoom" and rank == '9' and suit == trump_suit:
            return False
    return True


def initial_bidding_actions(current_agent, dealer, bidding_round, face_up, agent_hand):
    suit_to_action = {'♠': 34, '♥': 35, '♦': 36, '♣': 37}
    revealed = face_up[0]
    left_of_dealer = (dealer - 1) % 4

    if bidding_round == 1:
        if current_agent in (dealer, left_of_dealer):
            allowed = [32, 33, suit_to_action.get(revealed, 32), 38]
        else:
            allowed = [32, 33, suit_to_action.get(revealed, 32)]
    else:
        allowed = [32, 33] + [action for suit, action in suit_to_action.items() if suit != revealed]
        if current_agent in (dealer, left_of_dealer):
            allowed.append(38)

    if can_takweesh(agent_hand):
        allowed.append(43)

    mask = np.zeros(44, dtype=np.float32)
    for a in allowed:
        mask[a] = 1
    return mask


def allowed_overbids(buyer, dealer, bid_type, doubling_status, bidding_round, agent, face_up, agent_hand, trump_suit):
    if doubling_status is not None:
        mask = np.zeros(44, dtype=np.float32)
        mask[32] = 1
        return mask

    first_bidder = (dealer + 1) % 4
    buyer_order = (buyer - first_bidder) % 4
    agent_order = (agent - first_bidder) % 4
    tm_order = (((buyer + 2) % 4) - first_bidder) % 4
    has_priority = agent_order < buyer_order
    left_of_dealer = (dealer - 1) % 4

    suit_to_action = {'♠': 34, '♥': 35, '♦': 36, '♣': 37}
    revealed = face_up[0]

    allowed = {32}

    if bid_type == "Sun":
        if has_priority:
            allowed.add(33)
        if tm_order < buyer_order and agent in (dealer, left_of_dealer):
            allowed.add(38)
    else:
        allowed.add(33)
        if agent in (dealer, left_of_dealer):
            allowed.add(38)
        if has_priority:
            if bidding_round == 1:
                allowed.add(suit_to_action[revealed])
            else:
                for act in suit_to_action.values():
                    if act != suit_to_action[revealed]:
                        allowed.add(act)

    if can_takweesh(agent_hand, current_bid_type=bid_type, trump_suit=trump_suit):
        allowed.add(43)

    mask = np.zeros(44, dtype=np.float32)
    for a in allowed:
        mask[a] = 1
    return mask


def allowed_doubling_action(buy_type, buyer, agent, cumulative_scores, current_doubling_state, last_doubler):
    mask = np.zeros(44, dtype=np.float32)
    mask[32] = 1

    buyer_team = team(buyer)
    agent_team = team(agent)
    opp_team = 1 - buyer_team

    if current_doubling_state is None:
        if buy_type == "Sun":
            if cumulative_scores[buyer_team] > 100 >= cumulative_scores[opp_team] and agent_team != buyer_team:
                mask[39] = 1
        else:
            if agent_team != buyer_team:
                mask[39] = 1
    elif buy_type != "Sun":
        if current_doubling_state == "Double":
            if agent == buyer:
                mask[40] = 1
        elif current_doubling_state == "Three":
            if agent_team != buyer_team:
                mask[41] = 1
        elif current_doubling_state == "Four":
            if agent == buyer:
                mask[42] = 1
    return mask


__all__ = [
    "compute_bidding_order",
    "initial_bidding_actions",
    "allowed_overbids",
    "allowed_doubling_action",
]
