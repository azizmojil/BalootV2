import tensorflow as tf
from math import sqrt
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU, LayerNormalization
from tensorflow.keras.initializers import Orthogonal, Zeros
from tensorflow.keras.models import Model


def build_mappo_network(local_obs_dim, global_state_dim, act_dim):
    """
    Builds a Keras model for MAPPO with Centralized Training and Decentralized Execution (CTDE).
    The Actor (policy) uses ONLY the local observation.
    The Critic (value) uses BOTH the local observation and global state.
    """
    act_dim = int(act_dim)
    local_obs_input = Input(shape=(local_obs_dim,), name='local_obs_input')
    global_state_input = Input(shape=(global_state_dim,), name='global_state_input')
    bias_init = Zeros()

    def hidden_init():
        return Orthogonal(gain=sqrt(2))

    # --- ACTOR (Policy) Stream - uses ONLY local obs ---
    actor_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(local_obs_input)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LayerNormalization()(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    actor_net = Dense(128, kernel_initializer=hidden_init(), bias_initializer=bias_init)(actor_net)
    actor_net = LeakyReLU(alpha=0.01)(actor_net)
    policy_logits = Dense(act_dim, kernel_initializer=Orthogonal(gain=0.01),
                          bias_initializer=bias_init, name='policy_logits')(actor_net)

    # --- CRITIC (Value) Stream - uses BOTH local obs and global state ---
    global_net = Dense(512, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_state_input)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    global_net = Dense(256, kernel_initializer=hidden_init(), bias_initializer=bias_init)(global_net)
    global_net = LayerNormalization()(global_net)
    global_net = LeakyReLU(alpha=0.01)(global_net)
    
    critic_concat = Concatenate()([local_obs_input, global_net])
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