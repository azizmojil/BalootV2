import sys
import os
import warnings
import argparse
import datetime

# Suppress TF logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Prevent OpenBLAS/NumPy from spawning 64+ threads per worker!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
warnings.filterwarnings("ignore")

import tensorflow as tf
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from env.utils import flatten_obs, get_global_state, infer_model_dimensions

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

# Globals for workers
worker_env = None
worker_agent0 = None
worker_agent1 = None
is_random = False

def init_worker():
    """Initializes environment and CPU-bound agents for each worker."""
    global worker_env, worker_agent0, worker_agent1, is_random
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU') # Force workers to use CPU
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    worker_env = BalootMultiAgentEnv()
    sample_obs = worker_env.reset()
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(worker_env, sample_obs)
    
    worker_agent0 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    worker_agent1 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)

def run_eval_games(weights0, weights1, num_games):
    """Worker function to simulate a set of games."""
    global worker_env, worker_agent0, worker_agent1
    
    worker_agent0.model.set_weights(weights0)
    
    if weights1 == "random":
        use_random = True
    else:
        use_random = False
        worker_agent1.model.set_weights(weights1)

    t0_wins = 0
    t1_wins = 0
    draws = 0

    for _ in range(num_games):
        obs_dict = worker_env.reset()
        done = False
        
        while not done:
            current_player = worker_env.current_agent
            mask = obs_dict["action_mask"]
            
            if current_player in [0, 2]:
                local_obs = flatten_obs(obs_dict, worker_env.observation_space)
                global_state = get_global_state(worker_env)
                action, _, _ = worker_agent0.select_action(local_obs, global_state, mask)
            else:
                if use_random:
                    valid_actions = np.where(mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    local_obs = flatten_obs(obs_dict, worker_env.observation_space)
                    global_state = get_global_state(worker_env)
                    action, _, _ = worker_agent1.select_action(local_obs, global_state, mask)
            
            obs_dict, _, dones, _ = worker_env.step(action)
            done = dones.get('__all__', False)
            
        final_score_t0 = worker_env.cumulative_scores[0]
        final_score_t1 = worker_env.cumulative_scores[1]
        
        if final_score_t0 > final_score_t1:
            t0_wins += 1
        elif final_score_t1 > final_score_t0:
            t1_wins += 1
        else:
            draws += 1

    return t0_wins, t1_wins, draws

def main(args):
    # Dummy setup to extract weights in main thread
    print("Loading models into memory...")
    dummy_env = BalootMultiAgentEnv()
    sample_obs = dummy_env.reset()
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(dummy_env, sample_obs)
    
    agent0 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    agent0.model.load_weights(args.model1)
    weights0 = agent0.model.get_weights()
    print(f"Team 0 loaded model: {args.model1}")

    if args.model2.lower() == "random":
        weights1 = "random"
        print("Team 1 loaded model: Random Agent")
    else:
        agent1 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
        agent1.model.load_weights(args.model2)
        weights1 = agent1.model.get_weights()
        print(f"Team 1 loaded model: {args.model2}")

    # Determine how to distribute games
    num_workers = min(128, args.games)
    games_per_worker = args.games // num_workers
    remainder = args.games % num_workers
    
    distribution = [games_per_worker] * num_workers
    for i in range(remainder):
        distribution[i] += 1

    print(f"\nSimulating {args.games} games across {num_workers} parallel workers...")
    
    team0_wins = 0
    team1_wins = 0
    draws = 0

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)
    
    futures = []
    for g in distribution:
        if g > 0:
            futures.append(executor.submit(run_eval_games, weights0, weights1, g))

    pbar = tqdm(total=args.games, desc="Evaluating", unit="game")
    
    for future in concurrent.futures.as_completed(futures):
        t0, t1, dr = future.result()
        team0_wins += t0
        team1_wins += t1
        draws += dr
        pbar.update(t0 + t1 + dr)

    pbar.close()
    executor.shutdown()

    print("\n" + "="*40)
    print("VECTORIZED EVALUATION RESULTS")
    print("="*40)
    print(f"Team 0 (Model 1): {team0_wins} wins ({team0_wins/args.games*100:.1f}%)")
    print(f"Team 1 (Model 2): {team1_wins} wins ({team1_wins/args.games*100:.1f}%)")
    print(f"Draws:            {draws} ({draws/args.games*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAPPO agents using Vectorized CPU parallelization.")
    parser.add_argument("model1", type=str, help="Path to Team 0 model weights.")
    parser.add_argument("model2", type=str, help="Path to Team 1 model weights (or 'random').")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to simulate.")
    args = parser.parse_args()
    
    main(args)
