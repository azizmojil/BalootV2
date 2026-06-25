from env.constants import SET_CATEGORY_PRIORITY, SET_PRIORITY
from env.utils.cards import card_value


def set_category(set_type):
    if set_type in ("Mia_c", "Mia_s"):
        return "Mia"
    return set_type


def set_category_priority(set_info):
    category = set_category(set_info["type"])
    if category not in SET_CATEGORY_PRIORITY:
        raise ValueError(
            f"Unknown set category '{category}' from set type '{set_info['type']}'. "
            f"Valid categories: {list(SET_CATEGORY_PRIORITY.keys())}"
        )
    return SET_CATEGORY_PRIORITY[category]


def set_resolution_value(set_info):
    if not set_info["cards"]:
        raise ValueError(f"Set {set_info['type']} has no cards")
    return max(card_value(card) for card in set_info["cards"])


def set_resolution_key(set_info):
    return (
        set_category_priority(set_info),
        SET_PRIORITY[set_info["type"]],
        set_resolution_value(set_info),
    )


def set_value_label(value):
    return {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(value, str(value))


__all__ = [
    "set_category",
    "set_category_priority",
    "set_resolution_key",
    "set_resolution_value",
    "set_value_label",
]
