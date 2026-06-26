import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from env.utils import flatten_obs, get_global_state, infer_model_dimensions
from env.utils import translate_action, sort_hand_canonical, create_deck

os.system('')

CANONICAL_DECK = create_deck()


def get_relative_name(player_id, human_id):
    """Returns a relative string name for a player based on the human's seat."""
    if player_id is None:
        return "None"
    if player_id == human_id:
        return "\033[92mYou\033[0m"
    elif player_id == (human_id + 1) % 4:
        return "Your Right"
    elif player_id == (human_id + 2) % 4:
        return "Teammate"
    else:
        return "Your Left"

def format_card(action_str):
    """Wraps card strings in brackets and applies colors."""
    if "♥" in action_str or "♦" in action_str:
        return f"\033[91m[{action_str:>3}]\033[0m"
    elif "♠" in action_str or "♣" in action_str:
        return f"\033[97m[{action_str:>3}]\033[0m"
    return f"\033[93m[{action_str}]\033[0m"

def print_game_state(env, human_player_id):
    """Prints a beautiful, colored summary of the game state."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"\033[96m{'='*65}\033[0m")
    print(f"\033[1m🏆 BALOOT MATCH | ROUND {env.round_count} 🏆\033[0m".center(75))
    print(f"   Team 0: \033[92m{env.cumulative_scores[0]:<3}\033[0m  vs  Team 1: \033[92m{env.cumulative_scores[1]:<3}\033[0m".center(80))
    print(f"\033[96m{'='*65}\033[0m\n")

    if env.phase == 'bidding':
        print(f" 🗣️  \033[93mPHASE: BIDDING\033[0m | Dealer: {get_relative_name(env.dealer, human_player_id)}")
        face_up_idx = CANONICAL_DECK.index(env.face_up)
        print(f" 🎴 Face-Up Card:  {format_card(translate_action(face_up_idx))}")
        if env.buyer is not None:
            print(f" 📢 Current Bid:   \033[93m{env.game_type}\033[0m by {get_relative_name(env.buyer, human_player_id)}")
            if env.doubling_state:
                doubler_name = get_relative_name(getattr(env, 'last_doubler', None), human_player_id)
                print(f" 💥 Doubling:      \033[91m{env.doubling_state.upper()}\033[0m by {doubler_name}")
    else:
        print(f" 🃏 \033[93mPHASE: PLAYING\033[0m")
        trump_str = f" | Trump: \033[91m{env.trump_suit}\033[0m" if env.trump_suit in ("♥", "♦") else (f" | Trump: \033[97m{env.trump_suit}\033[0m" if env.trump_suit else "")
        dbl_str = f" | \033[91m{env.doubling_state.upper()}\033[0m" if env.doubling_state else ""
        print(f" 📜 Contract: \033[93m{env.game_type.upper()}\033[0m by {get_relative_name(env.buyer, human_player_id)}{trump_str}{dbl_str}")
        print(f" 👑 Trick Leader: {get_relative_name(env.trick_leader, human_player_id)}\n")

        if hasattr(env, 'declared_sets_info') and any(env.declared_sets_info):
            sets_str_list = []
            type_map = {"Sera": "Sira", "Khamseen": "50", "Mia_c": "100", "Mia_s": "100", "Arbamia": "400"}
            for p_idx, p_sets in enumerate(env.declared_sets_info):
                if not p_sets: continue
                
                if not getattr(env, 'sets_resolved', False):
                    has_played = env.current_trick[p_idx] is not None
                    is_turn = (env.current_agent == p_idx)
                    if not has_played and not is_turn: continue
                    
                p_name = get_relative_name(p_idx, human_player_id)
                pos_sets = []
                for s in p_sets:
                    name = type_map.get(s["type"], s["type"])
                    if getattr(env, 'sets_resolved', False) and "cards" in s:
                        cards_str = " ".join([format_card(translate_action(CANONICAL_DECK.index(c))) for c in s["cards"]])
                        name = f"{name} ({cards_str})"
                    pos_sets.append(name)
                    
                sets_str_list.append(f"\033[93m{p_name}\033[0m: {', '.join(pos_sets)}")
                
            if sets_str_list:
                print(" " * 22 + "--- ACTIVE SETS ---")
                print("    " + " | ".join(sets_str_list) + "\n")

        if hasattr(env, 'balot') and any(env.balot):
            balot_str_list = []
            for p_idx, has_balot in enumerate(env.balot):
                if has_balot:
                    p_name = get_relative_name(p_idx, human_player_id)
                    balot_str_list.append(f"\033[93m{p_name}\033[0m")
            if balot_str_list:
                print(" " * 23 + "--- BALOOT ---")
                print("    " + " & ".join(balot_str_list) + " declared Baloot!\n")
        
        if getattr(env, 'trick_history', []):
            print(" " * 20 + "--- PAST TRICKS ---")
            for i, trick_data in enumerate(env.trick_history):
                trick_cards = trick_data["cards"]
                winner_name = get_relative_name(trick_data["winner"], human_player_id)
                
                last_trick_str = []
                for p_idx, card in enumerate(trick_cards):
                    p_name = get_relative_name(p_idx, human_player_id)
                    if card:
                        last_trick_str.append(f"{p_name}: {format_card(translate_action(CANONICAL_DECK.index(card)))}")
                    else:
                        last_trick_str.append(f"{p_name}: \033[90m[___]\033[0m")
                print(f" Trick {i+1}: " + " | ".join(last_trick_str) + f"  👉 \033[93mWinner: {winner_name}\033[0m")
            print("")

        print(" " * 22 + "--- THE TABLE ---")
        trick_str = []
        for p_idx, card in enumerate(env.current_trick):
            p_name = get_relative_name(p_idx, human_player_id)
            if card:
                trick_str.append(f"{p_name}: {format_card(translate_action(CANONICAL_DECK.index(card)))}")
            else:
                trick_str.append(f"{p_name}: \033[90m[___]\033[0m")
        print("  " + " | ".join(trick_str) + "\n")

    print(f"\033[96m{'-'*65}\033[0m")
    print(f" 🖐️ YOUR HAND:")
    hand = env.hands[human_player_id]
    sorted_hand = sort_hand_canonical(hand)
    hand_str = [format_card(translate_action(CANONICAL_DECK.index(card))) for card in sorted_hand]
    print("    " + " ".join(hand_str))
    print(f"\033[96m{'='*65}\033[0m\n")


def get_human_action(env, obs_dict):
    """Gets a valid action from human input with a styled menu."""
    mask = obs_dict['action_mask']
    valid_action_indices = np.where(mask == 1)[0]

    print(" ✨ Available Actions:")
    action_map = {}
    for i, action_idx in enumerate(valid_action_indices):
        action_map[i] = action_idx
        action_str = translate_action(action_idx)
        print(f"    \033[96m[{i:>2}]\033[0m {format_card(action_str)}")

    while True:
        try:
            choice = int(input("\n 🎯 Choose an action by number: \033[92m"))
            print("\033[0m", end="")
            if choice in action_map:
                return action_map[choice]
            else:
                print("\033[91m ❌ Invalid choice. Please select a number from the list.\033[0m")
        except ValueError:
            print("\033[0m\033[91m ❌ Invalid input. Please enter a number.\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\033[0m\n 👋 Exiting game.")
            sys.exit(0)


def main(args):
    """Main function to run the game."""
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    env = BalootMultiAgentEnv()
    obs_dict = env.reset()

    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(env, obs_dict)

    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    try:
        agent.model.load_weights(args.model_path)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    human_player_id = args.player
    print(f"You are Player {human_player_id}. Your teammate is Player {(human_player_id + 2) % 4}.")

    done = False
    while not done:
        current_player = env.current_agent

        if current_player == human_player_id:
            print_game_state(env, human_player_id)
            action = get_human_action(env, obs_dict)
        else:
            local_obs = flatten_obs(obs_dict, env.observation_space)
            global_state = get_global_state(env)
            mask = obs_dict["action_mask"]
            action, _, _ = agent.select_action(local_obs, global_state, mask)
            ai_name = get_relative_name(current_player, human_player_id)
            print(f"\n--- {ai_name} (AI) plays: {translate_action(action)} ---", flush=True)
            time.sleep(0.5)

        obs_dict, _, dones, _ = env.step(action)
        
        if hasattr(env, 'resolution_logs') and env.resolution_logs:
            print("\n \033[93m" + "\n ".join(env.resolution_logs) + "\033[0m")
            env.resolution_logs.clear()
        done = dones.get('__all__', False)

    print("\n" + "="*50)
    print("GAME OVER")
    final_score_t0 = env.cumulative_scores[0]
    final_score_t1 = env.cumulative_scores[1]
    print(f"Final Score: Team 0: {final_score_t0} | Team 1: {final_score_t1}")

    if final_score_t0 == final_score_t1:
        winner = "It's a draw!"
    elif final_score_t0 > final_score_t1:
        winner = "Team 0"
    else:
        winner = "Team 1"
    print(f"{winner} wins the match!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Baloot against a trained MAPPO agent.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model weights (.h5 file)."
    )
    parser.add_argument(
        "--player",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Which player you want to be (0, 1, 2, or 3). Default is 0."
    )
    parsed_args = parser.parse_args()
    main(parsed_args)