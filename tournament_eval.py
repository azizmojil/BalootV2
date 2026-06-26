import sys
import os
import warnings
import argparse
import glob
import itertools

# Suppress TF logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import concurrent.futures
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from env.utils.state_utils import flatten_obs, get_global_state, infer_model_dimensions

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
    
    if weights0 == "random":
        use_random0 = True
    else:
        use_random0 = False
        worker_agent0.model.set_weights(weights0)
    
    if weights1 == "random":
        use_random1 = True
    else:
        use_random1 = False
        worker_agent1.model.set_weights(weights1)

    t0_wins = 0
    t1_wins = 0

    for _ in range(num_games):
        obs_dict = worker_env.reset()
        done = False
        
        while not done:
            current_player = worker_env.current_agent
            mask = obs_dict["action_mask"]
            
            if current_player in [0, 2]:
                if use_random0:
                    valid_actions = np.where(mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    local_obs = flatten_obs(obs_dict, worker_env.observation_space)
                    global_state = get_global_state(worker_env)
                    action, _, _ = worker_agent0.select_action(local_obs, global_state, mask, deterministic=True)
            else:
                if use_random1:
                    valid_actions = np.where(mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    local_obs = flatten_obs(obs_dict, worker_env.observation_space)
                    global_state = get_global_state(worker_env)
                    action, _, _ = worker_agent1.select_action(local_obs, global_state, mask, deterministic=True)
            
            obs_dict, _, dones, _ = worker_env.step(action)
            done = dones.get('__all__', False)
            
        final_score_t0 = worker_env.cumulative_scores[0]
        final_score_t1 = worker_env.cumulative_scores[1]
        
        if final_score_t0 > final_score_t1:
            t0_wins += 1
        elif final_score_t1 > final_score_t0:
            t1_wins += 1

    return t0_wins, t1_wins

def load_weights_robustly(model_path, local_obs_dim, global_state_dim, act_dim):
    if model_path.lower() == "random":
        return "random"
        
    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    try:
        agent.model.load_weights(model_path)
    except ValueError as e:
        print(f"Architecture mismatch for {model_path}. Transferring weights via old architecture...")
        from model import build_old_mappo_network
        old_model = build_old_mappo_network(local_obs_dim, global_state_dim, act_dim)
        old_model.load_weights(model_path)
        for new_layer in agent.model.layers:
            if new_layer.weights:
                try:
                    new_layer.set_weights(old_model.get_layer(name=new_layer.name).get_weights())
                except ValueError:
                    pass
    return agent.model.get_weights()


def main(args):
    model_files = sorted(glob.glob(os.path.join(args.folder, "*.h5")))
    
    if args.include_random:
        model_files.append("random")
        
    if not model_files:
        print(f"No .h5 models found in {args.folder}")
        return

    model_names = [os.path.basename(m).replace('.h5', '') if m != "random" else "Random" for m in model_files]
    num_models = len(model_files)
    
    print(f"Found {num_models} models. Loading weights into memory...")
    dummy_env = BalootMultiAgentEnv()
    sample_obs = dummy_env.reset()
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(dummy_env, sample_obs)
    
    all_weights = []
    for m in model_files:
        weights = load_weights_robustly(m, local_obs_dim, global_state_dim, act_dim)
        all_weights.append(weights)

    win_matrix = np.zeros((num_models, num_models))
    text_matrix = np.empty((num_models, num_models), dtype=object)

    num_workers = min(128, args.games)
    games_per_worker = args.games // num_workers
    remainder = args.games % num_workers
    distribution = [games_per_worker] * num_workers
    for i in range(remainder):
        distribution[i] += 1

    print(f"\nRunning round-robin tournament ({args.games} games per matchup)...")
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)
    
    total_matchups = num_models * num_models
    current_matchup = 1
    
    for i in range(num_models):
        for j in range(num_models):
            print(f"\nMatchup {current_matchup}/{total_matchups}: {model_names[i]} (T0) vs {model_names[j]} (T1)")
            
            if i == j and model_names[i] == "Random":
                t0_wins, t1_wins = args.games // 2, args.games - (args.games // 2)
            else:
                futures = []
                for g in distribution:
                    if g > 0:
                        futures.append(executor.submit(run_eval_games, all_weights[i], all_weights[j], g))

                t0_wins, t1_wins = 0, 0
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating", unit="worker", leave=False):
                    t0, t1 = future.result()
                    t0_wins += t0
                    t1_wins += t1
            
            win_rate = t0_wins / args.games
            win_matrix[i, j] = win_rate
            text_matrix[i, j] = f"{t0_wins}/{args.games}"
            current_matchup += 1

    executor.shutdown()

    print("\nGenerating heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(win_matrix, annot=text_matrix, fmt="", cmap="RdYlGn", vmin=0, vmax=1, 
                xticklabels=model_names, yticklabels=model_names, cbar_kws={'label': 'Win Rate (Team 0)'})
    
    plt.title(f"Baloot Tournament Win Rates\n(Row = Team 0, Column = Team 1)")
    plt.xlabel("Team 1")
    plt.ylabel("Team 0")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_file = "tournament_heatmap.png"
    plt.savefig(out_file, dpi=300)
    print(f"Heatmap saved to {out_file}")

    print("\n" + "="*80)
    print("TOURNAMENT RESULTS (Team 0 Wins / Total Games)")
    print("Row = Team 0, Column = Team 1")
    print("="*80)
    
    header_label = "T0 \\ T1"
    header = f"{header_label:>15} | " + " | ".join([f"{name[:10]:>10}" for name in model_names])
    print(header)
    print("-" * len(header))
    
    for i in range(num_models):
        row_str = f"{model_names[i][:15]:>15} | "
        row_vals = []
        for j in range(num_models):
            row_vals.append(f"{text_matrix[i, j]:>10}")
        print(row_str + " | ".join(row_vals))
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple MAPPO agents in a round-robin tournament.")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing .h5 model weights.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate per matchup.")
    parser.add_argument("--include_random", action="store_true", help="Include a random agent in the tournament.")
    args = parser.parse_args()
    
    main(args)
