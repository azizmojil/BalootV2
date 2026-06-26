import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from env.utils import require_positive_int

class MAPPOAgent:
    def __init__(self, local_obs_dim, global_state_dim, act_dim, model_builder,
                 lr=3e-4, gamma=0.99, clip_range=0.2, epochs=10, 
                 batch_size=64, value_coef=0.5, entropy_coef=0.01, gae_lambda=0.95, strategy=None):
        self.act_dim = require_positive_int(act_dim, "act_dim")
        self.local_obs_dim = require_positive_int(local_obs_dim, "local_obs_dim")
        self.global_state_dim = require_positive_int(global_state_dim, "global_state_dim")
        self.gamma = gamma
        self.clip_range = clip_range
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.strategy = strategy if strategy is not None else tf.distribute.get_strategy()
        
        with self.strategy.scope():
            self.model = model_builder(local_obs_dim, global_state_dim, act_dim)
            self.optimizer = Adam(learning_rate=lr)
            if hasattr(self.optimizer, 'build'):
                self.optimizer.build(self.model.trainable_variables)
            
        self.value_func = self.get_value_for_single_obs
        self.last_update_stats = {}

    def _as_vector(self, name, value, expected_dim):
        """Converts an input to a finite 1-D float32 vector of the expected length."""
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"{name} has length {arr.shape[0]}, expected {expected_dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        return arr

    def _validate_batch_shape(self, arr, name, expected_dim):
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(f"{name} must have shape (batch, {expected_dim}), got {arr.shape}")

    @tf.function
    def _forward_pass(self, local_obs_t, global_state_t, mask_t, deterministic=False):
        logits, value = self.model([local_obs_t, global_state_t], training=False)
        value = tf.squeeze(value, axis=1)
        
        very_negative = -1e10 * tf.ones_like(logits)
        masked_logits = tf.where(mask_t > 0, logits, very_negative)
        masked_log_probs = tf.nn.log_softmax(masked_logits, axis=1)

        if deterministic:
            action = tf.argmax(masked_logits, axis=1, output_type=tf.int64)
        else:
            action_tensor = tf.random.categorical(masked_log_probs, num_samples=1)
            action = tf.squeeze(action_tensor, axis=1)
            
        action = tf.cast(action, tf.int32)
        
        action_onehot = tf.one_hot(action, self.act_dim)
        log_prob = tf.reduce_sum(action_onehot * masked_log_probs, axis=1)

        return action, log_prob, value

    def select_action(self, local_obs, global_state, mask, deterministic=False):
        local_obs = self._as_vector("local_obs", local_obs, self.local_obs_dim)
        global_state = self._as_vector("global_state", global_state, self.global_state_dim)
        mask = self._as_vector("mask", mask, self.act_dim)
        if not np.any(mask > 0):
            raise ValueError(f"Cannot select an action because the action mask has no valid actions. "
                             f"mask_shape={np.shape(mask)}")

        local_obs_t = tf.convert_to_tensor(local_obs[None, :], dtype=tf.float32)
        global_state_t = tf.convert_to_tensor(global_state[None, :], dtype=tf.float32)
        mask_t = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)
        
        action_t, log_prob_t, value_t = self._forward_pass(local_obs_t, global_state_t, mask_t, deterministic=tf.constant(deterministic))

        return int(action_t[0].numpy()), log_prob_t[0], value_t[0]

    def select_actions(self, local_obs_batch, global_state_batch, mask_batch, deterministic=False):
        local_obs_t = tf.convert_to_tensor(local_obs_batch, dtype=tf.float32)
        global_state_t = tf.convert_to_tensor(global_state_batch, dtype=tf.float32)
        mask_t = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
        
        actions_t, log_probs_t, values_t = self._forward_pass(local_obs_t, global_state_t, mask_t, deterministic=tf.constant(deterministic))
        return actions_t.numpy(), log_probs_t.numpy(), values_t.numpy()

    def compute_advantages_and_returns(self, rewards, dones, values, last_value):
        """
        Computes advantages and returns using Generalized Advantage Estimation (GAE).
        """
        advantages = []
        last_advantage = 0
        
        extended_values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * extended_values[t + 1] * (1 - dones[t]) - extended_values[t]
            advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            advantages.insert(0, advantage)
            last_advantage = advantage

        returns = np.array(advantages) + np.array(values)
        return advantages, returns

    @tf.function
    def _value_forward_pass(self, local_obs_t, global_state_t):
        _, value = self.model([local_obs_t, global_state_t], training=False)
        return tf.squeeze(value, axis=1)

    def get_value_for_single_obs(self, local_obs, global_state):
        local_obs_t = tf.convert_to_tensor(local_obs[None, :], dtype=tf.float32)
        global_state_t = tf.convert_to_tensor(global_state[None, :], dtype=tf.float32)
        value_t = self._value_forward_pass(local_obs_t, global_state_t)
        return value_t[0].numpy()
    
    def update(self, memory):
        local_states = np.array(memory["local_states"], dtype=np.float32)
        global_states = np.array(memory["global_states"], dtype=np.float32)
        masks = np.array(memory["action_masks"], dtype=np.float32)
        actions = np.array(memory["actions"], dtype=np.int32)
        if len(actions) == 0:
            raise ValueError("Cannot update MAPPOAgent with an empty memory buffer.")
        self._validate_batch_shape(local_states, "local_states", self.local_obs_dim)
        self._validate_batch_shape(global_states, "global_states", self.global_state_dim)
        self._validate_batch_shape(masks, "action_masks", self.act_dim)
        if not np.all(np.isfinite(local_states)) or not np.all(np.isfinite(global_states)) or not np.all(np.isfinite(masks)):
            raise ValueError("Cannot update MAPPOAgent with non-finite states or action masks.")
        batch_size = len(actions)
        if local_states.shape[0] != batch_size or global_states.shape[0] != batch_size or masks.shape[0] != batch_size:
            raise ValueError(
                "MAPPOAgent memory arrays must have the same batch length: "
                f"actions={batch_size}, local_states={local_states.shape[0]}, "
                f"global_states={global_states.shape[0]}, action_masks={masks.shape[0]}"
            )
        if np.any(actions < 0) or np.any(actions >= self.act_dim):
            raise ValueError(f"actions must be in [0, {self.act_dim})")
        old_log_probs = np.array(memory["log_probs"], dtype=np.float32).flatten()
        old_values = np.array(memory["values"], dtype=np.float32).flatten()
        advantages = np.array(memory["advantages"], dtype=np.float32).flatten()
        returns = np.array(memory["returns"], dtype=np.float32).flatten()
        if not all(len(arr) == batch_size for arr in (old_log_probs, old_values, advantages, returns)):
            raise ValueError(
                "MAPPOAgent memory statistics must match the action batch length: "
                f"actions={batch_size}, log_probs={len(old_log_probs)}, values={len(old_values)}, "
                f"advantages={len(advantages)}, returns={len(returns)}"
            )

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices(
            (local_states, global_states, masks, actions, old_log_probs, old_values, advantages, returns)
        ).shuffle(buffer_size=1024).batch(self.batch_size)

        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

        def train_step(batch_local, batch_global, batch_masks, batch_actions, batch_old_log_probs, batch_old_values, batch_advantages, batch_returns):
            with tf.GradientTape() as tape:
                logits, new_values = self.model([batch_local, batch_global], training=True)
                new_values = tf.squeeze(new_values, axis=1)
                
                masked_logits = logits + (batch_masks - 1) * 1e9
                
                log_softmax_val = tf.nn.log_softmax(masked_logits, axis=1)
                new_probs = tf.nn.softmax(masked_logits, axis=1)
                
                batch_actions_onehot = tf.one_hot(batch_actions, self.act_dim)
                new_log_probs = tf.reduce_sum(batch_actions_onehot * log_softmax_val, axis=1)
                
                log_ratio = new_log_probs - tf.stop_gradient(batch_old_log_probs)
                ratio = tf.exp(log_ratio)
                clip_fraction = tf.reduce_mean(
                    tf.cast(tf.abs(ratio - 1.0) > self.clip_range, tf.float32)
                )
                approx_kl = 0.5 * tf.reduce_mean(tf.square(log_ratio))
                
                surr1 = ratio * batch_advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                new_values_clipped = batch_old_values + tf.clip_by_value(
                    new_values - batch_old_values, -self.clip_range, self.clip_range
                )
                value_loss_unclipped = tf.square(new_values - batch_returns)
                value_loss_clipped = tf.square(new_values_clipped - batch_returns)
                value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))

                entropy_per_step = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-9), axis=1)
                entropy = tf.reduce_mean(entropy_per_step)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                loss = loss / self.strategy.num_replicas_in_sync
                
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            loss = loss * self.strategy.num_replicas_in_sync
            return loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction

        @tf.function
        def distributed_train_step(batch_data):
            per_replica_results = self.strategy.run(train_step, args=batch_data)
            return (
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None),
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None),
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None),
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[3], axis=None),
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[4], axis=None),
                self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[5], axis=None)
            )

        total_loss, total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0, 0.0
        total_kl, total_clip_fraction = 0.0, 0.0
        num_batches = 0
        target_kl = 0.015
        for epoch in range(self.epochs):
            for batch_data in dist_dataset:
                loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction = distributed_train_step(batch_data)
                total_loss += loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                total_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_batches += 1
            
            if (total_kl / num_batches) > target_kl * 1.5:
                # Early stopping to prevent policy collapse
                break

        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        avg_kl = total_kl / num_batches
        avg_clip_fraction = total_clip_fraction / num_batches

        returns_var = np.var(returns)
        explained_variance = 1.0 - (
            np.var(returns - old_values) / (returns_var + 1e-8)
        )
        self.last_update_stats = {
            "approx_kl": float(avg_kl.numpy()),
            "clip_fraction": float(avg_clip_fraction.numpy()),
            "explained_variance": float(explained_variance),
        }

        return avg_loss.numpy(), avg_policy_loss.numpy(), avg_value_loss.numpy(), avg_entropy.numpy()
