import numpy as np
from collections import defaultdict
from env.constants import *
from env.rewards import *


def create_deck():
    deck = [(s, r)
            for s in SUITS
            for r in RANKS]
    return deck


def full_card_mask(mask32):
    full = np.zeros(43, dtype=np.float32)
    full[:len(mask32)] = mask32
    return full


def sort_hand_canonical(hand):
    canonical_deck = create_deck()
    return sorted(hand, key=lambda card: canonical_deck.index(card))


def get_canonical_hand(hand):
    suit_mapping = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
    canonical = []
    for card in hand:
        if isinstance(card, dict):
            suit = card.get("suit")
            rank = card.get("rank")
        else:
            suit, rank = card
        if isinstance(suit, int):
            suit = suit_mapping.get(suit, suit)
        canonical.append((suit, rank))
    return canonical


def card_value(card):
    mapping = {'7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return mapping[card[1]]


def team(player):
    return player % 2


def one_hot_card(card):
    canonical_deck = create_deck()
    try:
        idx = canonical_deck.index(card)
        vec = np.zeros(32, dtype=np.float32)
        vec[idx] = 1.0
    except:
        vec = np.zeros(32, dtype=np.float32)
    return vec


def pad_array(arr, target_length, pad_value=0.0, axis=0):
    arr = np.array(arr)
    current_length = arr.shape[axis]
    if current_length < target_length:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_length - current_length)
        arr = np.pad(arr, pad_width, constant_values=pad_value)
    elif current_length > target_length:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_length)
        arr = arr[tuple(slices)]
    return arr


def value_to_rank(value):
    mapping = {7: '7', 8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    return mapping.get(value, None)


def record_consecutive_set(suit, seq, sets_found):
    if len(seq) < 3:
        return
    if len(seq) == 3:
        set_type = "Sera"
    elif len(seq) == 4:
        set_type = "Khamseen"
    elif len(seq) == 5:
        set_type = "Mia_c"
    else:
        set_type = "Mia_c"
        seq = seq[:5]
    cards_in_set = [(suit, value_to_rank(val)) for val in seq]
    sets_found.append({"type": set_type, "cards": cards_in_set})


def check_set_balot(declared_sets, trump_suit, balot):
    if not trump_suit:
        return
    for p, sets in enumerate(declared_sets):
        for s in sets:
            t = s.get("type", "")
            if t in ("Mia_s", "Mia_c"):
                continue
            cards = s.get("cards", [])
            ranks = {c[1] for c in cards}
            suits = {c[0] for c in cards}
            if trump_suit in suits and {"K", "Q"}.issubset(ranks):
                balot[p] = True


def _make_set(suit, seq_vals, set_type):
    """Helper to build the {type, cards} dict from integer sequence."""
    cards = [(suit, value_to_rank(v)) for v in seq_vals]
    return {"type": set_type, "cards": cards}


def _record_consecutive_all(suit, seq, sets_found):
    """Records all consecutive sets, including overlapping windows for runs >= 5."""
    if not seq:
        return

    current_seq = [seq[0]]
    for val in seq[1:]:
        if val == current_seq[-1] + 1:
            current_seq.append(val)
        else:
            if len(current_seq) == 3:
                sets_found.append(_make_set(suit, current_seq, "Sera"))
            elif len(current_seq) == 4:
                sets_found.append(_make_set(suit, current_seq, "Khamseen"))
            elif len(current_seq) >= 5:
                for i in range(len(current_seq) - 4):
                    window = current_seq[i:i+5]
                    sets_found.append(_make_set(suit, window, "Mia_c"))
            current_seq = [val]
    
    if len(current_seq) == 3:
        sets_found.append(_make_set(suit, current_seq, "Sera"))
    elif len(current_seq) == 4:
        sets_found.append(_make_set(suit, current_seq, "Khamseen"))
    elif len(current_seq) >= 5:
        for i in range(len(current_seq) - 4):
            window = current_seq[i:i+5]
            sets_found.append(_make_set(suit, window, "Mia_c"))


def detect_sets(hand):
    """Detects all valid sets in a hand, including overlapping runs."""
    candidate_sets = []

    if sum(1 for (s, r) in hand if r == "A") == 4:
        candidate_sets.append({"type": "Arbamia", "cards": [card for card in hand if card[1] == "A"]})
    for rank in ['10', 'J', 'Q', 'K']:
        if sum(1 for (s, r) in hand if r == rank) == 4:
            candidate_sets.append({"type": "Mia_s", "cards": [card for card in hand if card[1] == rank]})

    rank_value = {'7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    suit_groups = defaultdict(list)
    for card in hand:
        suit_groups[card[0]].append(card)
    for suit, cards in suit_groups.items():
        values = sorted({rank_value[c[1]] for c in cards})
        _record_consecutive_all(suit, values, candidate_sets)

    declared_sets = []
    used_cards = set()
    sorted_candidates = sorted(candidate_sets, key=lambda s: SET_PRIORITY[s["type"]], reverse=True)
    for s in sorted_candidates:
        if not any(tuple(card) in used_cards for card in s["cards"]):
            declared_sets.append(s)
            for card in s["cards"]:
                used_cards.add(tuple(card))

    return declared_sets


def compute_potential(hand, face_up, game_type, set_bonus_hukoom, set_bonus_sun):
    canonical_hand = get_canonical_hand(hand)
    original_sets = detect_sets(canonical_hand)
    if original_sets:
        if game_type == "Hukoom":
            original_bonus = max(set_bonus_hukoom.get(s["type"], 0) for s in original_sets)
        else:
            original_bonus = max(set_bonus_sun.get(s["type"], 0) for s in original_sets)
    else:
        original_bonus = 0

    augmented_hand = canonical_hand + [face_up]
    augmented_sets = detect_sets(augmented_hand)
    if augmented_sets:
        if game_type == "Hukoom":
            augmented_bonus = max(set_bonus_hukoom.get(s["type"], 0) for s in augmented_sets)
        else:
            augmented_bonus = max(set_bonus_sun.get(s["type"], 0) for s in augmented_sets)
    else:
        augmented_bonus = 0

    improvement = augmented_bonus - original_bonus
    improvement_flag = True if improvement > 0 else False

    return improvement_flag, improvement


def compute_bidding_order(dealer, num_agents=4):
    return [(dealer + i) % num_agents for i in range(1, num_agents + 1)]


def initial_bidding_actions(current_agent, dealer, bidding_round, face_up):
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

    mask = np.zeros(43, dtype=np.float32)
    for a in allowed:
        mask[a] = 1
    return mask


def allowed_overbids(buyer, dealer, bid_type, doubling_status, bidding_round, agent, face_up):
    if doubling_status is not None:
        mask = np.zeros(43, dtype=np.float32)
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
        if agent == buyer:
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
        if bidding_round > 1 and buyer != dealer:
            allowed.add(33)

    mask = np.zeros(43, dtype=np.float32)
    for a in allowed:
        mask[a] = 1
    return mask


def allowed_doubling_action(buy_type, buyer, agent, cumulative_scores, current_doubling_state, last_doubler):
    mask = np.zeros(43, dtype=np.float32)
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
        if last_doubler is not None:
            last_doubler_team = team(last_doubler)
            if agent_team != last_doubler_team:
                if current_doubling_state == "Double":
                    mask[40] = 1
                elif current_doubling_state == "Three":
                    mask[41] = 1
                elif current_doubling_state == "Four":
                    mask[42] = 1
    return mask


def enforce_follow_suit_mask(agent_hand, trick_suit):
    canonical_deck = create_deck()
    hand_mask_32 = np.array([1.0 if card in agent_hand else 0.0 for card in canonical_deck], dtype=np.float32)

    if trick_suit is None or not any(card[0] == trick_suit for card in agent_hand):
        return full_card_mask(hand_mask_32)

    follow_mask_32 = np.zeros_like(hand_mask_32)
    for i, card in enumerate(canonical_deck):
        if hand_mask_32[i] == 1.0 and card[0] == trick_suit:
            follow_mask_32[i] = 1.0

    return full_card_mask(follow_mask_32)


def trumping_rules(agent_hand, current_trick, trump_suit, agent_index):
    canonical_deck = create_deck()
    hand32  = np.array([1.0 if c in agent_hand else 0.0 for c in canonical_deck], dtype=np.float32)
    trump32 = np.array([1.0 if (c in agent_hand and c[0]==trump_suit) else 0.0
                        for c in canonical_deck], dtype=np.float32)

    highest, owner = None, None
    for idx, c in enumerate(current_trick):
        if c and c[0]==trump_suit:
            if highest is None or HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]]:
                highest, owner = c, idx

    if highest is None:
        return full_card_mask(trump32) if trump32.sum()>0 else full_card_mask(hand32)

    beat32 = np.array([1.0 if (c in agent_hand
                               and c[0]==trump_suit
                               and HUKOOM_ORDER[c[1]]>HUKOOM_ORDER[highest[1]])
                       else 0.0 for c in canonical_deck], dtype=np.float32)

    if beat32.sum()>0:
        return full_card_mask(beat32)
    return full_card_mask(trump32) if trump32.sum()>0 else full_card_mask(hand32)


def non_trump_lead_mask(agent_hand, current_trick, trick_suit, trump_suit, agent_index):
    canonical_deck = create_deck()
    hand32  = np.array([1.0 if c in agent_hand else 0.0 for c in canonical_deck], dtype=np.float32)
    trump32 = np.array([1.0 if (c in agent_hand and c[0]==trump_suit) else 0.0
                        for c in canonical_deck], dtype=np.float32)

    played = [(i,c) for i,c in enumerate(current_trick) if c and c[0]==trump_suit]

    if not played:
        highest_lead, owner = None, None
        for i, c in enumerate(current_trick):
            if c and c[0] == trick_suit:
                if highest_lead is None or SUN_ORDER[c[1]] > SUN_ORDER[highest_lead[1]]:
                    highest_lead, owner = c, i
        
        if owner is not None and team(owner) == team(agent_index):
            return full_card_mask(hand32)
            
        return full_card_mask(trump32) if trump32.sum()>0 else full_card_mask(hand32)

    owner, highest = played[0]
    for i, c in played[1:]:
        if HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]]:
            highest, owner = c, i

    if team(owner) == team(agent_index):
        return full_card_mask(hand32)

    beat32 = np.array([1.0 if (c in agent_hand
                               and c[0]==trump_suit
                               and HUKOOM_ORDER[c[1]]>HUKOOM_ORDER[highest[1]])
                       else 0.0 for c in canonical_deck], dtype=np.float32)
    if beat32.sum()>0:
        return full_card_mask(beat32)
    return full_card_mask(hand32)


def get_full_play_mask_hukoom(agent_hand, current_trick, agent, trick_suit, trump_suit, doubling_state=None):
    canonical_deck = create_deck()
    if trick_suit is None:
        if doubling_state in ("Double", "Four"):
            non_trump_cards = [card for card in agent_hand if card[0] != trump_suit]
            if non_trump_cards:
                hand_mask_32 = np.array([1.0 if (card in agent_hand and card[0] != trump_suit) else 0.0 for card in canonical_deck], dtype=np.float32)
                return full_card_mask(hand_mask_32)
                
        hand_mask_32 = np.array([1.0 if card in agent_hand else 0.0 for card in canonical_deck], dtype=np.float32)
        return full_card_mask(hand_mask_32)

    if trick_suit == trump_suit:
        return trumping_rules(agent_hand,
                              current_trick,
                              trump_suit,
                              agent)

    has_lead_suit = any(card[0] == trick_suit for card in agent_hand)
    if has_lead_suit:
        canonical_deck = create_deck()
        follow_mask_32 = np.array([
            1.0 if (card in agent_hand and card[0] == trick_suit) else 0.0
            for card in canonical_deck
        ], dtype=np.float32)
        return full_card_mask(follow_mask_32)

    return non_trump_lead_mask(agent_hand,
                               current_trick,
                               trick_suit,
                               trump_suit,
                               agent)


def translate_action(action):
    if action < 32:
        deck = create_deck()
        card = deck[action]
        return f"{card[0]}{card[1]}"

    mapping = {32: "Pass",
               33: "Sun",
               34: "Hukoom ♠",
               35: "Hukoom ♥",
               36: "Hukoom ♦",
               37: "Hukoom ♣",
               38: "Ashkal",
               39: "Double",
               40: "Three",
               41: "Four",
               42: "Gahwa"}
    return mapping.get(action, f"Unknown Action {action}")





def detect_sets_full(hand):
    """
    Just like your detect_sets, but uses record_consecutive_all
    so you catch overlapping Mia_c windows.
    """
    candidate_sets = []

    if sum(1 for (s, r) in hand if r == "A") == 4:
        candidate_sets.append({
            "type": "Arbamia",
            "cards": [card for card in hand if card[1] == "A"]
        })

    for rank in ['10', 'J', 'Q', 'K']:
        if sum(1 for (s, r) in hand if r == rank) == 4:
            candidate_sets.append({
                "type": "Mia_s",
                "cards": [card for card in hand if card[1] == rank]
            })

    rank_value = {'7': 7, '8': 8, '9': 9, '10': 10,
                  'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    suit_groups = defaultdict(list)
    for card in hand:
        suit_groups[card[0]].append(card)
    for suit, cards in suit_groups.items():
        values = sorted({rank_value[c[1]] for c in cards})
        _record_consecutive_all(suit, values, candidate_sets)

    declared_sets = []
    used = set()
    for s in sorted(candidate_sets,
                    key=lambda x: SET_PRIORITY[x["type"]],
                    reverse=True):
        if not any(tuple(c) in used for c in s["cards"]):
            declared_sets.append(s)
            for c in s["cards"]:
                used.add(tuple(c))

    return declared_sets
