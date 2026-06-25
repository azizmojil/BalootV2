import numpy as np

from env.constants import BID_ACTIONS, RANKS, SUITS


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


def relative_player_index(player, observer):
    if player is None:
        return None
    return (player - observer) % 4


def relative_player_order(observer, num_players=4):
    return [(observer + offset) % num_players for offset in range(num_players)]


def one_hot_card(card):
    canonical_deck = create_deck()
    try:
        idx = canonical_deck.index(card)
        vec = np.zeros(32, dtype=np.float32)
        vec[idx] = 1.0
    except ValueError:
        vec = np.zeros(32, dtype=np.float32)
    return vec


def one_hot_index(index, size):
    """Returns a one-hot vector, all zeros for None, and raises for out-of-range indices."""
    vec = np.zeros(size, dtype=np.float32)
    if index is None:
        return vec
    if 0 <= index < size:
        vec[index] = 1.0
        return vec
    raise ValueError(f"index must be in [0, {size}), got {index}")


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


__all__ = [
    "BID_ACTIONS",
    "create_deck",
    "full_card_mask",
    "sort_hand_canonical",
    "get_canonical_hand",
    "card_value",
    "team",
    "relative_player_index",
    "relative_player_order",
    "one_hot_card",
    "one_hot_index",
    "pad_array",
    "value_to_rank",
    "translate_action",
]
