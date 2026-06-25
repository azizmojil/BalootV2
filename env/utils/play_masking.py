import numpy as np

from env.constants import HUKOOM_ORDER, SUN_ORDER
from env.utils.cards import create_deck, full_card_mask, team


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
    hand32 = np.array([1.0 if c in agent_hand else 0.0 for c in canonical_deck], dtype=np.float32)
    trump32 = np.array([1.0 if (c in agent_hand and c[0] == trump_suit) else 0.0
                        for c in canonical_deck], dtype=np.float32)

    highest, owner = None, None
    for idx, c in enumerate(current_trick):
        if c and c[0] == trump_suit:
            if highest is None or HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]]:
                highest, owner = c, idx

    if highest is None:
        if trump32.sum() > 0:
            return full_card_mask(trump32)
        return full_card_mask(hand32)

    if owner is not None and team(owner) == team(agent_index):
        if trump32.sum() > 0:
            return full_card_mask(trump32)
        return full_card_mask(hand32)

    beat32 = np.array([1.0 if (c in agent_hand
                               and c[0] == trump_suit
                               and HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]])
                       else 0.0 for c in canonical_deck], dtype=np.float32)

    if beat32.sum() > 0:
        return full_card_mask(beat32)
        
    if trump32.sum() > 0:
        return full_card_mask(trump32)
        
    return full_card_mask(hand32)


def non_trump_lead_mask(agent_hand, current_trick, trick_suit, trump_suit, agent_index):
    canonical_deck = create_deck()
    hand32 = np.array([1.0 if c in agent_hand else 0.0 for c in canonical_deck], dtype=np.float32)
    trump32 = np.array([1.0 if (c in agent_hand and c[0] == trump_suit) else 0.0
                        for c in canonical_deck], dtype=np.float32)

    played = [(i, c) for i, c in enumerate(current_trick) if c and c[0] == trump_suit]

    if not played:
        highest_lead, owner = None, None
        for i, c in enumerate(current_trick):
            if c and c[0] == trick_suit:
                if highest_lead is None or SUN_ORDER[c[1]] > SUN_ORDER[highest_lead[1]]:
                    highest_lead, owner = c, i

        if owner is not None and team(owner) == team(agent_index):
            return full_card_mask(hand32)

        return full_card_mask(trump32) if trump32.sum() > 0 else full_card_mask(hand32)

    owner, highest = played[0]
    for i, c in played[1:]:
        if HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]]:
            highest, owner = c, i

    if team(owner) == team(agent_index):
        return full_card_mask(hand32)

    beat32 = np.array([1.0 if (c in agent_hand
                               and c[0] == trump_suit
                               and HUKOOM_ORDER[c[1]] > HUKOOM_ORDER[highest[1]])
                       else 0.0 for c in canonical_deck], dtype=np.float32)
    if beat32.sum() > 0:
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

    has_lead_suit = any(card[0] == trick_suit for card in agent_hand)

    if trick_suit == trump_suit:
        return trumping_rules(agent_hand,
                              current_trick,
                              trump_suit,
                              agent)

    if has_lead_suit:
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


__all__ = [
    "enforce_follow_suit_mask",
    "trumping_rules",
    "non_trump_lead_mask",
    "get_full_play_mask_hukoom",
]
