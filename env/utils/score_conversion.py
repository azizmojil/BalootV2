def convert_bant(bant, game_type):
    if bant <= 0:
        return 0
    if game_type == "Hukoom":
        rem = bant % 10
        tens = bant // 10
        lower = tens * 10
        upper = lower + 10
        if rem == 5:
            rounded = lower
        else:
            rounds_down = (bant - lower) <= (upper - bant)
            rounded = lower if rounds_down else upper
        return rounded // 10

    rem = bant % 10
    tens = bant // 10
    if rem == 5:
        return tens * 2 + 1
    lower = tens * 10
    upper = lower + 10
    rounded = lower if (bant - lower) <= (upper - bant) else upper
    return (rounded // 10) * 2


__all__ = [
    "convert_bant",
]
