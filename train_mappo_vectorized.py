import sys
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Prevent OpenBLAS/NumPy from spawning 64+ threads per worker!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LLVM_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

import datetime
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from env.utils.state_utils import flatten_obs, get_global_state, infer_model_dimensions

config = {
    "num_episodes": 10000,
    "update_interval": 8192,  # Increased because data collection is much faster!
    "start_lr": 5e-4,
    "end_lr": 5e-5,
    "gamma": 0.99,
    "clip_range": 0.25,
    "epochs": 8,
    "batch_size": 2048, # Increased batch size for Multi-GPU efficiency
    "gae_lambda": 0.95,
    "start_entropy": 0.10,
    "end_entropy": 0.05,
    "num_workers": 96  # Reduced slightly to prevent OS thread limits (was 128)
}

NUM_PLAYERS = 4

# Globals for workers
worker_env = None
worker_agent = None

def init_worker():
    """Initializes environment and CPU-bound agent for each worker process."""
    global worker_env, worker_agent
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU') # Force workers to use CPU
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    worker_env = BalootMultiAgentEnv()
    sample_obs = worker_env.reset()
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(worker_env, sample_obs)
    
    worker_agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network,
                              lr=config["start_lr"], gamma=config["gamma"],
                              clip_range=config["clip_range"], epochs=config["epochs"],
                              batch_size=config["batch_size"], gae_lambda=config["gae_lambda"],
                              entropy_coef=config["start_entropy"], strategy=None) # No strategy on workers

def compute_gae_for_player(agent, player_buf, last_local_obs, last_global_state):
    rewards = player_buf["rewards"]
    dones = player_buf["dones"]
    values = player_buf["values"]

    if len(rewards) == 0:
        return [], []

    last_value = agent.get_value_for_single_obs(last_local_obs, last_global_state)
    advantages, returns = agent.compute_advantages_and_returns(rewards, dones, values, last_value)
    return advantages, returns

def run_episode(weights, entropy_coef):
    """Runs a single episode, computes GAE locally, and returns the memory dict."""
    global worker_env, worker_agent
    worker_agent.model.set_weights(weights)
    worker_agent.entropy_coef = entropy_coef
    
    buffers = {
        p: {k: [] for k in ["local_states", "global_states", "action_masks",
                             "actions", "log_probs", "rewards", "dones", "values"]}
        for p in range(NUM_PLAYERS)
    }
    
    obs_dict = worker_env.reset()
    episode_rewards = [0.0] * NUM_PLAYERS
    match_done = False

    while not match_done:
        current_player = worker_env.current_agent
        local_obs = flatten_obs(obs_dict, worker_env.observation_space)
        global_state = get_global_state(worker_env)
        mask = obs_dict["action_mask"]

        action, logp, value = worker_agent.select_action(local_obs, global_state, mask)
        next_obs_dict, rewards, dones, infos = worker_env.step(action)

        match_done = dones.get('__all__', False)

        player_buf = buffers[current_player]
        player_buf["local_states"].append(local_obs)
        player_buf["global_states"].append(global_state)
        player_buf["action_masks"].append(mask)
        player_buf["actions"].append(action)
        player_buf["log_probs"].append(logp.numpy() if hasattr(logp, 'numpy') else logp)
        player_buf["values"].append(value.numpy() if hasattr(value, 'numpy') else value)
        player_buf["rewards"].append(0.0)
        player_buf["dones"].append(0.0)

        for p in range(NUM_PLAYERS):
            if len(buffers[p]["rewards"]) > 0:
                buffers[p]["rewards"][-1] += rewards.get(f"player_{p}", 0.0)
                if match_done:
                    buffers[p]["dones"][-1] = 1.0
            episode_rewards[p] += rewards.get(f"player_{p}", 0.0)

        obs_dict = next_obs_dict

    # Compute GAE
    last_global_state = get_global_state(worker_env)
    all_advantages = {}
    all_returns = {}
    original_agent = worker_env.current_agent
    for p in range(NUM_PLAYERS):
        worker_env.current_agent = p
        p_obs_dict = worker_env.get_observation()
        p_last_local_obs = flatten_obs(p_obs_dict, worker_env.observation_space)
        adv, ret = compute_gae_for_player(worker_agent, buffers[p], p_last_local_obs, last_global_state)
        all_advantages[p] = adv
        all_returns[p] = ret
    worker_env.current_agent = original_agent

    # Merge buffers
    merged = {k: [] for k in ["local_states", "global_states", "action_masks",
                              "actions", "log_probs", "values", "advantages", "returns"]}
    for p in range(NUM_PLAYERS):
        buf = buffers[p]
        if len(buf["rewards"]) == 0: continue
        merged["local_states"].extend(buf["local_states"])
        merged["global_states"].extend(buf["global_states"])
        merged["action_masks"].extend(buf["action_masks"])
        merged["actions"].extend(buf["actions"])
        merged["log_probs"].extend(buf["log_probs"])
        merged["values"].extend(buf["values"])
        merged["advantages"].extend(all_advantages[p])
        merged["returns"].extend(all_returns[p])

    team0_reward = (episode_rewards[0] + episode_rewards[2]) / 2.0
    team1_reward = (episode_rewards[1] + episode_rewards[3]) / 2.0
    total_steps = sum(len(buffers[p]["rewards"]) for p in range(NUM_PLAYERS))

    return merged, team0_reward, team1_reward, total_steps

def get_entropy_coef(ep):
    frac = ep / config["num_episodes"]
    return config["start_entropy"] + frac * (config["end_entropy"] - config["start_entropy"])

def get_learning_rate(ep):
    frac = min(1.0, ep / (config["num_episodes"] * 0.8))
    return config["start_lr"] + frac * (config["end_lr"] - config["start_lr"])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Train MAPPO Agent for Baloot")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_ep", type=int, default=0, help="Episode to start from")
    parser.add_argument("--start_update", type=int, default=0, help="Update count to start from")
    args = parser.parse_args()

    # Create dummy env to infer shapes
    dummy_env = BalootMultiAgentEnv()
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(dummy_env, dummy_env.reset())

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPUs being used: {strategy.num_replicas_in_sync}")

    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network,
                       lr=config["start_lr"], gamma=config["gamma"],
                       clip_range=config["clip_range"], epochs=config["epochs"],
                       batch_size=config["batch_size"], gae_lambda=config["gae_lambda"],
                       entropy_coef=config["start_entropy"], strategy=strategy)

    if args.resume:
        agent.model.load_weights(args.resume)

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-MAPPO-Vectorized"
    log_dir = os.path.join("logs", "monitor", run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=config["num_workers"], initializer=init_worker)
    
    update_count = args.start_update
    episodes_completed = args.start_ep

    print(f"Starting Vectorized Collection with {config['num_workers']} workers...")
    pbar = tqdm(total=config["num_episodes"], desc="Training MAPPO Vectorized", unit="ep")
    if episodes_completed > 0:
        pbar.update(episodes_completed)
    
    while episodes_completed < config["num_episodes"]:
        agent.entropy_coef = get_entropy_coef(episodes_completed)
        new_lr = get_learning_rate(episodes_completed)
        agent.optimizer.learning_rate.assign(new_lr)
        current_weights = agent.model.get_weights()

        collected_memory = {k: [] for k in ["local_states", "global_states", "action_masks",
                                            "actions", "log_probs", "values", "advantages", "returns"]}
        collected_steps = 0
        futures = []
        jobs_to_submit = max(1, config["update_interval"] // 80)
        
        for _ in range(jobs_to_submit):
            futures.append(executor.submit(run_episode, current_weights, agent.entropy_coef))

        new_episodes = 0
        for future in concurrent.futures.as_completed(futures):
            mem, t0_r, t1_r, steps = future.result()
            
            for k in collected_memory.keys():
                collected_memory[k].extend(mem[k])
                
            collected_steps += steps
            new_episodes += 1

            with summary_writer.as_default():
                tf.summary.scalar("Reward/Team0_Episode", t0_r, step=episodes_completed + new_episodes)
                tf.summary.scalar("Reward/Team1_Episode", t1_r, step=episodes_completed + new_episodes)

        episodes_completed += new_episodes
        pbar.update(new_episodes)

        # Update
        loss, policy_loss, value_loss, entropy = agent.update(collected_memory)
        update_count += 1

        with summary_writer.as_default():
            tf.summary.scalar("Loss/Total", loss, step=update_count)
            tf.summary.scalar("Loss/Policy", policy_loss, step=update_count)
            tf.summary.scalar("Loss/Value", value_loss, step=update_count)
            tf.summary.scalar("Loss/Entropy", entropy, step=update_count)
            tf.summary.scalar("Params/LearningRate", new_lr, step=update_count)
            summary_writer.flush()

        pbar.set_postfix({
            'Loss': f'{loss:.3f}', 
            'Pol': f'{policy_loss:.3f}', 
            'Val': f'{value_loss:.3f}', 
            'Steps': collected_steps
        })

        if update_count % 5 == 0:
            agent.model.save_weights(os.path.join(model_dir, f"mappo_update_{update_count}.weights.h5"))

    pbar.close()

    agent.model.save_weights(os.path.join(model_dir, "final_mappo.weights.h5"))
    executor.shutdown()
    print("Training complete.")
