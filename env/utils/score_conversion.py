def convert_bant(bant, game_type):
    if bant <= 0:
        return 0
    if game_type == "Hukoom":
        rem = bant % 10
        tens = bant // 10
        rounded = tens * 10 if rem == 5 else (tens *
                                              10 if (bant - tens * 10) <= ((tens + 1) * 10 - bant)
                                              else (tens + 1) * 10)
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
