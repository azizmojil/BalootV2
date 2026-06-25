from collections import defaultdict

from env.constants import RANKS, SET_PRIORITY, SUITS
from env.utils.cards import get_canonical_hand, value_to_rank


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


__all__ = [
    "record_consecutive_set",
    "check_set_balot",
    "detect_sets",
    "compute_potential",
    "detect_sets_full",
]
