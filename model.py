import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, LeakyReLU, LayerNormalization
from tensorflow.keras.models import Model


def build_mappo_network(local_obs_dim, global_state_dim, act_dim, dropout_rate=0.1):
    """
    Builds a Keras model for MAPPO with Centralized Training and Decentralized Execution (CTDE).
    The Actor (policy) uses ONLY the local observation.
    The Critic (value) uses BOTH the local observation and global state.
    """
    local_obs_input = Input(shape=(local_obs_dim,), name='local_obs_input')
    global_state_input = Input(shape=(global_state_dim,), name='global_state_input')

    # --- ACTOR (Policy) Stream - uses ONLY local obs ---
    actor_net = Dense(256)(local_obs_input)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    if dropout_rate > 0:
        actor_net = Dropout(dropout_rate)(actor_net)
    actor_net = Dense(128, activation=LeakyReLU(alpha=0.01))(actor_net)
    policy_logits = Dense(act_dim, name='policy_logits')(actor_net)

    # --- CRITIC (Value) Stream - uses BOTH local obs and global state ---
    global_net = Dense(256)(global_state_input)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    
    critic_concat = Concatenate()([local_obs_input, global_net])
    critic_net = Dense(256)(critic_concat)
    critic_net = LayerNormalization()(critic_net)
    critic_net = LeakyReLU(alpha=0.01)(critic_net)
    if dropout_rate > 0:
        critic_net = Dropout(dropout_rate)(critic_net)
    critic_net = Dense(128, activation=LeakyReLU(alpha=0.01))(critic_net)
    value_output = Dense(1, name='value_output')(critic_net)

    model = Model(
        inputs=[local_obs_input, global_state_input],
        outputs=[policy_logits, value_output]
    )

    return model