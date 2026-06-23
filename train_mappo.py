import sys
import os
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from utils import flatten_obs, get_global_state

# ─── Configuration ───────────────────────────────────────────────────────────
config = {
    "num_episodes": 10000,
    "update_interval": 4096,      # Min steps per player before an update
    "start_lr": 3e-4,
    "end_lr": 1e-5,
    "gamma": 0.99,
    "clip_range": 0.2,
    "epochs": 10,
    "batch_size": 1024,
    "gae_lambda": 0.95,
    "dropout_rate": 0.0,
    "start_entropy": 0.05,
    "end_entropy": 0.01,
}

NUM_PLAYERS = 4


# ─── Helper: per-player memory buffers ────────────────────────────────────────
def make_empty_buffers():
    """Create an empty per-player memory buffer."""
    return {
        p: {k: [] for k in ["local_states", "global_states", "action_masks",
                             "actions", "log_probs", "rewards", "dones", "values"]}
        for p in range(NUM_PLAYERS)
    }


def buffer_len(buffers):
    """Return total transitions across all players."""
    return sum(len(buffers[p]["rewards"]) for p in range(NUM_PLAYERS))


def compute_gae_for_player(agent, player_buf, last_local_obs, last_global_state):
    """
    Compute GAE advantages and returns for a single player's buffer.
    Bootstraps the final value from the current observation.
    """
    rewards = player_buf["rewards"]
    dones = player_buf["dones"]
    values = player_buf["values"]

    if len(rewards) == 0:
        return [], []

    # Bootstrap value for the last step
    last_value = agent.get_value_for_single_obs(last_local_obs, last_global_state)

    advantages, returns = agent.compute_advantages_and_returns(
        rewards, dones, values, last_value
    )
    return advantages, returns


def merge_player_buffers(buffers, all_advantages, all_returns):
    """
    Merge all per-player buffers into a single update dict for PPO.
    All players share one model, so we concatenate their experiences.
    """
    merged = {k: [] for k in ["local_states", "global_states", "action_masks",
                                "actions", "log_probs", "values", "advantages", "returns"]}

    for p in range(NUM_PLAYERS):
        buf = buffers[p]
        adv = all_advantages[p]
        ret = all_returns[p]

        if len(buf["rewards"]) == 0:
            continue

        merged["local_states"].extend(buf["local_states"])
        merged["global_states"].extend(buf["global_states"])
        merged["action_masks"].extend(buf["action_masks"])
        merged["actions"].extend(buf["actions"])
        merged["log_probs"].extend(buf["log_probs"])
        merged["values"].extend(buf["values"])
        merged["advantages"].extend(adv)
        merged["returns"].extend(ret)

    return merged


# ─── Annealing schedules ──────────────────────────────────────────────────────
def get_entropy_coef(ep):
    frac = ep / config["num_episodes"]
    return config["start_entropy"] + frac * (config["end_entropy"] - config["start_entropy"])


def get_learning_rate(ep):
    frac = min(1.0, ep / (config["num_episodes"] * 0.8))
    return config["start_lr"] + frac * (config["end_lr"] - config["start_lr"])


if __name__ == "__main__":
    # ─── Setup ────────────────────────────────────────────────────────────────────
    env = BalootMultiAgentEnv()
    sample_obs = env.reset()
    local_obs_dim = flatten_obs(sample_obs).shape[0]
    global_state_dim = get_global_state(env).shape[0]
    act_dim = env.action_space.n

    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network,
                       lr=config["start_lr"], gamma=config["gamma"],
                       clip_range=config["clip_range"], epochs=config["epochs"],
                       batch_size=config["batch_size"], gae_lambda=config["gae_lambda"],
                       dropout_rate=config["dropout_rate"], entropy_coef=config["start_entropy"])

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-MAPPO"
    log_dir = os.path.join("logs", "monitor", run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # ─── Training loop ────────────────────────────────────────────────────────────
    buffers = make_empty_buffers()
    update_count = 0

    episode_bar = tqdm(range(config["num_episodes"]), desc="Training MAPPO", unit="ep")
    for ep in episode_bar:
        # Anneal hyperparameters
        agent.entropy_coef = get_entropy_coef(ep)
        new_lr = get_learning_rate(ep)
        agent.optimizer.learning_rate.assign(new_lr)

        obs_dict = env.reset()
        episode_rewards = [0.0] * NUM_PLAYERS
        match_done = False

        while not match_done:
            current_player = env.current_agent
            local_obs = flatten_obs(obs_dict)
            global_state = get_global_state(env)
            mask = obs_dict["action_mask"]

            # Select action
            action, logp, value = agent.select_action(local_obs, global_state, mask)

            # Step environment
            next_obs_dict, rewards, dones, infos = env.step(action)

            # Determine if this is a round end or match end
            round_done = any(v for k, v in dones.items() if k != '__all__')
            match_done = dones.get('__all__', False)

            # Store transition for the CURRENT player
            player_buf = buffers[current_player]
            player_buf["local_states"].append(local_obs)
            player_buf["global_states"].append(global_state)
            player_buf["action_masks"].append(mask)
            player_buf["actions"].append(action)
            player_buf["log_probs"].append(logp.numpy())
            player_buf["values"].append(value.numpy())
            player_buf["rewards"].append(0.0)  # Will be updated below
            player_buf["dones"].append(0.0)    # Will be updated below

            # The environment returns rewards/dones for ALL players (e.g., at the end of a trick/round).
            # We must apply these to the LAST action each player took.
            for p in range(NUM_PLAYERS):
                if len(buffers[p]["rewards"]) > 0:
                    buffers[p]["rewards"][-1] += rewards.get(f"player_{p}", 0.0)
                    if match_done:
                        buffers[p]["dones"][-1] = 1.0

            # Accumulate per-episode rewards for logging
            for p in range(NUM_PLAYERS):
                episode_rewards[p] += rewards.get(f"player_{p}", 0.0)

            obs_dict = next_obs_dict

        # ─── Log per-episode metrics ──────────────────────────────────────────
        team0_reward = (episode_rewards[0] + episode_rewards[2]) / 2.0
        team1_reward = (episode_rewards[1] + episode_rewards[3]) / 2.0
        with summary_writer.as_default():
            tf.summary.scalar("Reward/Team0_Episode", team0_reward, step=ep)
            tf.summary.scalar("Reward/Team1_Episode", team1_reward, step=ep)

        # ─── Check if it's time to update ─────────────────────────────────────
        if buffer_len(buffers) >= config["update_interval"]:
            last_global_state = get_global_state(env)

            all_advantages = {}
            all_returns = {}
            original_agent = env.current_agent
            for p in range(NUM_PLAYERS):
                env.current_agent = p
                p_obs_dict = env.get_observation()
                p_last_local_obs = flatten_obs(p_obs_dict)
                
                adv, ret = compute_gae_for_player(
                    agent, buffers[p], p_last_local_obs, last_global_state
                )
                all_advantages[p] = adv
                all_returns[p] = ret
                
            env.current_agent = original_agent

            update_memory = merge_player_buffers(buffers, all_advantages, all_returns)

            loss, policy_loss, value_loss, entropy = agent.update(update_memory)
            update_count += 1

            total_rewards_per_player = [sum(buffers[p]["rewards"]) for p in range(NUM_PLAYERS)]
            num_round_ends = sum(sum(buffers[p]["dones"]) for p in range(NUM_PLAYERS)) / NUM_PLAYERS
            num_round_ends = max(1, num_round_ends)

            avg_t0 = (total_rewards_per_player[0] + total_rewards_per_player[2]) / (2 * num_round_ends)
            avg_t1 = (total_rewards_per_player[1] + total_rewards_per_player[3]) / (2 * num_round_ends)

            with summary_writer.as_default():
                tf.summary.scalar("Loss/Total", loss, step=update_count)
                tf.summary.scalar("Loss/Policy", policy_loss, step=update_count)
                tf.summary.scalar("Loss/Value", value_loss, step=update_count)
                tf.summary.scalar("Loss/Entropy", entropy, step=update_count)
                tf.summary.scalar("PPO/ApproxKL", agent.last_update_stats.get("approx_kl", 0.0), step=update_count)
                tf.summary.scalar("PPO/ClipFraction", agent.last_update_stats.get("clip_fraction", 0.0), step=update_count)
                tf.summary.scalar("PPO/ExplainedVariance", agent.last_update_stats.get("explained_variance", 0.0), step=update_count)
                tf.summary.scalar("Reward/Avg_T0_per_Update", avg_t0, step=update_count)
                tf.summary.scalar("Reward/Avg_T1_per_Update", avg_t1, step=update_count)
                tf.summary.scalar("Params/LearningRate", new_lr, step=update_count)
                tf.summary.scalar("Params/EntropyCoef", agent.entropy_coef, step=update_count)
                summary_writer.flush()

            tqdm.write(
                f"Update {update_count:4d} (Ep {ep:5d}) | "
                f"Loss: {loss:.3f} | Pol: {policy_loss:.3f} | Val: {value_loss:.3f} | "
                f"Ent: {entropy:.3f} | KL: {agent.last_update_stats.get('approx_kl', 0.0):.4f} | "
                f"Clip: {agent.last_update_stats.get('clip_fraction', 0.0):.2f} | "
                f"Avg T0: {avg_t0:.1f} | Avg T1: {avg_t1:.1f}"
            )

            # ─── Save checkpoints periodically ────────────────────────────────
            if update_count % 25 == 0:
                checkpoint_path = os.path.join(model_dir, f"mappo_update_{update_count}.h5")
                agent.model.save_weights(checkpoint_path)
                tqdm.write(f"  ★ Checkpoint saved to {checkpoint_path}")

            buffers = make_empty_buffers()

    final_path = os.path.join(model_dir, "final_mappo.h5")
    agent.model.save_weights(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
