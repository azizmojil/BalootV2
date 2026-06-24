import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from utils import flatten_obs, get_global_state

def main(args):
    env = BalootMultiAgentEnv()
    obs_dict = env.reset()

    local_obs_dim = flatten_obs(obs_dict).shape[0]
    global_state_dim = get_global_state(env).shape[0]
    act_dim = env.action_space.n

    agent0 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    try:
        agent0.model.load_weights(args.model1)
        print(f"Team 0 loaded model: {args.model1}")
    except Exception as e:
        print(f"Error loading {args.model1}: {e}")
        sys.exit(1)

    if args.model2.lower() == "random":
        agent1 = None
        print("Team 1 loaded model: Random Agent")
    else:
        agent1 = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
        try:
            agent1.model.load_weights(args.model2)
            print(f"Team 1 loaded model: {args.model2}")
        except Exception as e:
            print(f"Error loading {args.model2}: {e}")
            sys.exit(1)

    team0_wins = 0
    team1_wins = 0
    draws = 0

    print(f"\nSimulating {args.games} games...")
    
    for game in tqdm(range(args.games)):
        obs_dict = env.reset()
        done = False
        
        while not done:
            current_player = env.current_agent
            mask = obs_dict["action_mask"]
            
            if current_player in [0, 2]:
                local_obs = flatten_obs(obs_dict)
                global_state = get_global_state(env)
                action, _, _ = agent0.select_action(local_obs, global_state, mask)
            else:
                if agent1 is None:
                    valid_actions = np.where(mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    local_obs = flatten_obs(obs_dict)
                    global_state = get_global_state(env)
                    action, _, _ = agent1.select_action(local_obs, global_state, mask)
            
            obs_dict, _, dones, _ = env.step(action)
            done = dones.get('__all__', False)
            
        final_score_t0 = env.cumulative_scores[0]
        final_score_t1 = env.cumulative_scores[1]
        
        if final_score_t0 > final_score_t1:
            team0_wins += 1
        elif final_score_t1 > final_score_t0:
            team1_wins += 1
        else:
            draws += 1

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Team 0 (Model 1): {team0_wins} wins ({team0_wins/args.games*100:.1f}%)")
    print(f"Team 1 (Model 2): {team1_wins} wins ({team1_wins/args.games*100:.1f}%)")
    print(f"Draws:            {draws} ({draws/args.games*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAPPO agents against each other or random.")
    parser.add_argument("model1", type=str, help="Path to Team 0 model weights.")
    parser.add_argument("model2", type=str, help="Path to Team 1 model weights (or type 'random' to play against a random agent).")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate.")
    args = parser.parse_args()
    main(args)
