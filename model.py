import tensorflow as tf
from math import sqrt
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Lambda, LeakyReLU, LayerNormalization
from tensorflow.keras.initializers import Orthogonal, Zeros
from tensorflow.keras.models import Model
from env.utils import OBSERVATION_SCHEMA
from env.utils import require_positive_int


CARD_VECTOR_KEYS = ("faceup_card", "own_hand", "played_cards", "unknown_cards")
CARD_BLOCK_FEATURES = {
    "cards_ownership": 4,
    "trick": 4,
    "last_trick": 4,
    "trick_history": 32,
}
CARD_FEATURE_KEYS = set(CARD_VECTOR_KEYS) | set(CARD_BLOCK_FEATURES)


def _schema_offsets():
    offsets = {}
    start = 0
    for key, shape in OBSERVATION_SCHEMA.items():
        size = 1
        for dim in shape:
            size *= dim
        offsets[key] = (start, size)
        start += size
    return offsets


def build_mappo_network(local_obs_dim, global_state_dim, act_dim):
    """
    Builds a Keras model for MAPPO with Centralized Training and Decentralized Execution (CTDE).
    The Actor (policy) uses ONLY the local observation.
    The Critic (value) uses BOTH the local observation and global state.
    """
    local_obs_dim = require_positive_int(local_obs_dim, "local_obs_dim")
    global_state_dim = require_positive_int(global_state_dim, "global_state_dim")
    act_dim = require_positive_int(act_dim, "act_dim")
    local_obs_input = Input(shape=(local_obs_dim,), name='local_obs_input')
    global_state_input = Input(shape=(global_state_dim,), name='global_state_input')
    bias_init = Zeros()

    def hidden_init():
        return Orthogonal(gain=sqrt(2))

    offsets = _schema_offsets()
    expected_local_obs_dim = sum(
        size for key, (_, size) in offsets.items() if key != "action_mask"
    )
    if local_obs_dim != expected_local_obs_dim:
        raise ValueError(
            f"local_obs_dim {local_obs_dim} does not match OBSERVATION_SCHEMA "
            f"network input size {expected_local_obs_dim}"
        )

    def slice_feature(key):
        start, size = offsets[key]
        return Lambda(
            lambda tensor, begin=start, end=start + size: tensor[:, begin:end],
            name=f"{key}_slice",
        )(local_obs_input)

    card_feature_parts = []
    for key in CARD_VECTOR_KEYS:
        card_feature_parts.append(
            Lambda(lambda tensor: tf.expand_dims(tensor, axis=-1), name=f"{key}_card_axis")(
                slice_feature(key)
            )
        )
    for key, feature_count in CARD_BLOCK_FEATURES.items():
        block = slice_feature(key)
        if key == "cards_ownership":
            card_feature_parts.append(
                Lambda(
                    lambda tensor, features=feature_count: tf.reshape(tensor, (-1, 32, features)),
                    name=f"{key}_card_features",
                )(block)
            )
        else:
            card_feature_parts.append(
                Lambda(
                    lambda tensor, features=feature_count: tf.transpose(
                        tf.reshape(tensor, (-1, features, 32)),
                        perm=(0, 2, 1),
                    ),
                    name=f"{key}_card_features",
                )(block)
            )

    card_states = Concatenate(axis=-1, name="card_states")(card_feature_parts)
    card_ids = Lambda(
        lambda tensor: tf.tile(
            tf.expand_dims(tf.range(32, dtype=tf.int32), axis=0),
            [tf.shape(tensor)[0], 1],
        ),
        name="card_ids",
    )(local_obs_input)
    card_embeddings = Embedding(input_dim=32, output_dim=16, name="card_embedding")(card_ids)
    card_inputs = Concatenate(axis=-1, name="card_identity_and_state")([card_embeddings, card_states])
    card_net = Dense(64, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="shared_card_dense_1")(card_inputs)
    card_net = LayerNormalization(name="shared_card_norm_1")(card_net)
    card_net = LeakyReLU(alpha=0.01, name="shared_card_activation_1")(card_net)
    card_net = Dense(64, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="shared_card_dense_2")(card_net)
    card_net = LayerNormalization(name="shared_card_norm_2")(card_net)
    card_net = LeakyReLU(alpha=0.01, name="shared_card_activation_2")(card_net)
    cards_flattened = Flatten(name="cards_flattened")(card_net)

    context_parts = [
        slice_feature(key)
        for key in OBSERVATION_SCHEMA
        if key not in CARD_FEATURE_KEYS and key != "action_mask"
    ]
    game_context = Concatenate(name="non_card_context")(context_parts)
    context_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="context_dense")(game_context)
    context_net = LayerNormalization(name="context_norm")(context_net)
    context_net = LeakyReLU(alpha=0.01, name="context_activation")(context_net)

    local_features = Concatenate(name="local_card_context_features")([cards_flattened, context_net])

    actor_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(local_features)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    policy_logits = Dense(act_dim, kernel_initializer=Orthogonal(gain=0.01),
                          bias_initializer=bias_init, name='policy_logits')(actor_net)

    global_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_state_input)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    global_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_net)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    
    stopped_local_features = Lambda(lambda x: tf.stop_gradient(x), name="stopped_local_features")(local_features)
    critic_concat = Concatenate()([stopped_local_features, global_net])
    critic_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_concat)
    critic_net = LayerNormalization()(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    critic_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_net)
    critic_net = LayerNormalization()(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    critic_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    value_output = Dense(1, kernel_initializer=Orthogonal(gain=1.0),
                         bias_initializer=bias_init, name='value_output')(critic_net)

    model = Model(
        inputs=[local_obs_input, global_state_input],
        outputs=[policy_logits, value_output]
    )

    return model

def build_old_mappo_network(local_obs_dim, global_state_dim, act_dim):
    """Builds the exact architecture from before the stop_gradient fix to allow loading old weights."""
    local_obs_input = Input(shape=(local_obs_dim,), name='local_obs_input')
    global_state_input = Input(shape=(global_state_dim,), name='global_state_input')
    bias_init = Zeros()
    def hidden_init(): return Orthogonal(gain=sqrt(2))

    offsets = _schema_offsets()
    def slice_feature(key):
        start, size = offsets[key]
        return Lambda(lambda tensor, begin=start, end=start + size: tensor[:, begin:end], name=f"{key}_slice")(local_obs_input)

    card_feature_parts = []
    for key in CARD_VECTOR_KEYS:
        card_feature_parts.append(Lambda(lambda tensor: tf.expand_dims(tensor, axis=-1), name=f"{key}_card_axis")(slice_feature(key)))
    for key, feature_count in CARD_BLOCK_FEATURES.items():
        block = slice_feature(key)
        if key == "cards_ownership":
            card_feature_parts.append(Lambda(lambda tensor, features=feature_count: tf.reshape(tensor, (-1, 32, features)), name=f"{key}_card_features")(block))
        else:
            card_feature_parts.append(Lambda(lambda tensor, features=feature_count: tf.transpose(tf.reshape(tensor, (-1, features, 32)), perm=(0, 2, 1)), name=f"{key}_card_features")(block))

    card_states = Concatenate(axis=-1, name="card_states")(card_feature_parts)
    card_ids = Lambda(lambda tensor: tf.tile(tf.expand_dims(tf.range(32, dtype=tf.int32), axis=0), [tf.shape(tensor)[0], 1]), name="card_ids")(local_obs_input)
    card_embeddings = Embedding(input_dim=32, output_dim=16, name="card_embedding")(card_ids)
    card_inputs = Concatenate(axis=-1, name="card_identity_and_state")([card_embeddings, card_states])
    card_net = Dense(64, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="shared_card_dense_1")(card_inputs)
    card_net = LayerNormalization(name="shared_card_norm_1")(card_net)
    card_net = LeakyReLU(alpha=0.01, name="shared_card_activation_1")(card_net)
    card_net = Dense(64, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="shared_card_dense_2")(card_net)
    card_net = LayerNormalization(name="shared_card_norm_2")(card_net)
    card_net = LeakyReLU(alpha=0.01, name="shared_card_activation_2")(card_net)
    cards_flattened = Flatten(name="cards_flattened")(card_net)

    context_parts = [slice_feature(key) for key in OBSERVATION_SCHEMA if key not in CARD_FEATURE_KEYS and key != "action_mask"]
    game_context = Concatenate(name="non_card_context")(context_parts)
    context_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init, name="context_dense")(game_context)
    context_net = LayerNormalization(name="context_norm")(context_net)
    context_net = LeakyReLU(alpha=0.01, name="context_activation")(context_net)

    local_features = Concatenate(name="local_card_context_features")([cards_flattened, context_net])

    actor_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(local_features)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    policy_logits = Dense(act_dim, kernel_initializer=Orthogonal(gain=0.01), bias_initializer=bias_init, name='policy_logits')(actor_net)

    global_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_state_input)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    global_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_net)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    
    critic_concat = Concatenate()([local_features, global_net])
    critic_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_concat)
    critic_net = LayerNormalization()(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    critic_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_net)
    critic_net = LayerNormalization()(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    critic_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init)(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    value_output = Dense(1, kernel_initializer=Orthogonal(gain=1.0), bias_initializer=bias_init, name='value_output')(critic_net)

    return Model(inputs=[local_obs_input, global_state_input], outputs=[policy_logits, value_output])

