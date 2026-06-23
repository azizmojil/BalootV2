import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import flatten_obs

class MAPPOAgent:
    def __init__(self, local_obs_dim, global_state_dim, act_dim, model_builder,
                 lr=3e-4, gamma=0.99, clip_range=0.2, epochs=10, 
                 batch_size=64, value_coef=0.5, entropy_coef=0.01, dropout_rate=0.0, gae_lambda=0.95):
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.model = model_builder(local_obs_dim, global_state_dim, act_dim, dropout_rate)
        self.optimizer = Adam(learning_rate=lr)
        self.value_func = self.get_value_for_single_obs # Start with the single-obs version
        self.last_update_stats = {}

    def select_action(self, local_obs, global_state, mask):
        if not np.any(mask > 0):
            raise ValueError("Cannot select an action because the action mask has no valid actions.")

        local_obs_t = tf.convert_to_tensor(local_obs[None, :], dtype=tf.float32)
        global_state_t = tf.convert_to_tensor(global_state[None, :], dtype=tf.float32)
        mask_t = tf.convert_to_tensor(mask[None, :], dtype=tf.float32)
        
        logits, value = self.model([local_obs_t, global_state_t], training=False)
        value = tf.squeeze(value, axis=0)
        
        # Apply mask for action selection
        very_negative = -1e10 * tf.ones_like(logits)
        masked_logits = tf.where(mask_t > 0, logits, very_negative)

        # Sample action from the masked distribution
        action_tensor = tf.random.categorical(tf.nn.log_softmax(masked_logits, axis=1), num_samples=1)
        action = int(tf.squeeze(action_tensor).numpy())

        # The log probability must come from the masked distribution
        log_prob = tf.nn.log_softmax(masked_logits, axis=1)[0, action]

        return action, log_prob, value

    def compute_advantages_and_returns(self, rewards, dones, values, last_value):
        """
        Computes advantages and returns using Generalized Advantage Estimation (GAE).
        """
        advantages = []
        last_advantage = 0
        
        # Append the last value for easier calculation of deltas
        extended_values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * extended_values[t + 1] * (1 - dones[t]) - extended_values[t]
            advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            advantages.insert(0, advantage)
            last_advantage = advantage

        returns = np.array(advantages) + np.array(values)
        return advantages, returns

    def get_value_for_single_obs(self, local_obs, global_state):
        local_obs_t = tf.convert_to_tensor(local_obs[None, :], dtype=tf.float32)
        global_state_t = tf.convert_to_tensor(global_state[None, :], dtype=tf.float32)
        _, value = self.model([local_obs_t, global_state_t], training=False)
        return tf.squeeze(value, axis=0).numpy()
    
    def update(self, memory):
        local_states = np.array(memory["local_states"], dtype=np.float32)
        global_states = np.array(memory["global_states"], dtype=np.float32)
        masks = np.array(memory["action_masks"], dtype=np.float32)
        actions = np.array(memory["actions"], dtype=np.int32)
        if len(actions) == 0:
            raise ValueError("Cannot update MAPPOAgent with an empty memory buffer.")
        old_log_probs = np.array(memory["log_probs"], dtype=np.float32).flatten()
        old_values = np.array(memory["values"], dtype=np.float32).flatten()
        advantages = np.array(memory["advantages"], dtype=np.float32).flatten()
        returns = np.array(memory["returns"], dtype=np.float32).flatten()

        # Normalize advantages over the entire batch of experience
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices(
            (local_states, global_states, masks, actions, old_log_probs, old_values, advantages, returns)
        ).shuffle(buffer_size=1024).batch(self.batch_size)

        @tf.function
        def train_step(batch_local, batch_global, batch_masks, batch_actions, batch_old_log_probs, batch_old_values, batch_advantages, batch_returns):
            with tf.GradientTape() as tape:
                logits, new_values = self.model([batch_local, batch_global], training=True)
                new_values = tf.squeeze(new_values, axis=1)
                
                # Apply mask before softmax to prevent NaN from 0 * -inf in entropy
                masked_logits = logits + (batch_masks - 1) * 1e9
                
                log_softmax_val = tf.nn.log_softmax(masked_logits, axis=1)
                new_probs = tf.nn.softmax(masked_logits, axis=1)
                
                batch_actions_onehot = tf.one_hot(batch_actions, self.act_dim)
                new_log_probs = tf.reduce_sum(batch_actions_onehot * log_softmax_val, axis=1)
                
                # Calculate ratio in log-space and clip for stability
                log_ratio = new_log_probs - tf.stop_gradient(batch_old_log_probs)
                ratio = tf.exp(log_ratio)
                clip_fraction = tf.reduce_mean(
                    tf.cast(tf.abs(ratio - 1.0) > self.clip_range, tf.float32)
                )
                approx_kl = 0.5 * tf.reduce_mean(tf.square(log_ratio))
                
                surr1 = ratio * batch_advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                # Clipping value loss
                # Clip the value to reduce variance
                new_values_clipped = batch_old_values + tf.clip_by_value(
                    new_values - batch_old_values, -self.clip_range, self.clip_range
                )
                value_loss_unclipped = tf.square(new_values - batch_returns)
                value_loss_clipped = tf.square(new_values_clipped - batch_returns)
                value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))

                # Entropy of the policy (calculated on the masked distribution)
                # Add a small epsilon to prevent log(0)
                entropy_per_step = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-9), axis=1)
                entropy = tf.reduce_mean(entropy_per_step)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction

        total_loss, total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0, 0.0
        total_kl, total_clip_fraction = 0.0, 0.0
        num_batches = 0
        for epoch in range(self.epochs):
            for batch_data in dataset:
                loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction = train_step(*batch_data)
                total_loss += loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                total_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        avg_kl = total_kl / num_batches
        avg_clip_fraction = total_clip_fraction / num_batches

        _, predicted_values = self.model([local_states, global_states], training=False)
        predicted_values = np.array(predicted_values, dtype=np.float32).flatten()
        returns_var = np.var(returns)
        explained_variance = 1.0 - (
            np.var(returns - predicted_values) / (returns_var + 1e-8)
        )
        self.last_update_stats = {
            "approx_kl": float(avg_kl.numpy()),
            "clip_fraction": float(avg_clip_fraction.numpy()),
            "explained_variance": float(explained_variance),
        }

        return avg_loss.numpy(), avg_policy_loss.numpy(), avg_value_loss.numpy(), avg_entropy.numpy()
