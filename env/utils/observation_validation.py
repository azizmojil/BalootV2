import numpy as np

from env.constants import BID_ACTIONS, NUM_PLAYERS


OBSERVATION_NUM_PLAYERS = NUM_PLAYERS
OBSERVATION_BIDDING_HISTORY_LENGTH = OBSERVATION_NUM_PLAYERS
OBSERVATION_BIDDING_HISTORY_FEATURES = (
    OBSERVATION_BIDDING_HISTORY_LENGTH * (OBSERVATION_NUM_PLAYERS + len(BID_ACTIONS))
)
OBSERVATION_TRICK_HISTORY_LENGTH = 8
OBSERVATION_TRICK_HISTORY_FEATURES = (
    OBSERVATION_TRICK_HISTORY_LENGTH * OBSERVATION_NUM_PLAYERS * 32
)
OBSERVATION_SCHEMA = {
    # 5 relative player indicators: dealer, teammate, buyer, trick leader, last doubler.
    "player_roles": (22,),
    # Phase, game type, trump, doubling, initial bid, and final bid one-hots.
    "game_context": (41,),
    "score_context": (10,),
    "faceup_card": (32,),
    "own_hand": (32,),
    "played_cards": (32,),
    "unknown_cards": (32,),
    "cards_ownership": (128,),
    "trick": (128,),
    "last_trick": (128,),
    "trick_history": (OBSERVATION_TRICK_HISTORY_FEATURES,),
    "declared_sets": (16,),
    "revealed_sets": (20,),
    "bidding_history": (OBSERVATION_BIDDING_HISTORY_FEATURES,),
    "action_mask": (44,),
}


def validate_observation(obs, schema=OBSERVATION_SCHEMA, epsilon=1e-3):
    expected_keys = tuple(schema.keys())
    actual_keys = tuple(obs.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"Observation keys do not match schema; expected {expected_keys}, got {actual_keys}")
    for key, shape in schema.items():
        arr = obs[key]
        if arr.shape != shape:
            raise ValueError(f"Observation '{key}' has shape {arr.shape}, expected {shape}")
        if arr.dtype != np.float32:
            raise ValueError(f"Observation '{key}' has dtype {arr.dtype}, expected float32")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Observation '{key}' contains non-finite values")
        if np.any(arr < -epsilon) or np.any(arr > 1.0 + epsilon):
            raise ValueError(
                f"Observation '{key}' contains values outside [0, 1]: "
                f"min={arr.min()}, max={arr.max()}"
            )


__all__ = [
    "OBSERVATION_BIDDING_HISTORY_FEATURES",
    "OBSERVATION_BIDDING_HISTORY_LENGTH",
    "OBSERVATION_NUM_PLAYERS",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_TRICK_HISTORY_FEATURES",
    "OBSERVATION_TRICK_HISTORY_LENGTH",
    "validate_observation",
]
