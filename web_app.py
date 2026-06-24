from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.environment import BalootMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from model import build_mappo_network
from utils import flatten_obs, get_global_state, infer_model_dimensions
from env.utils import translate_action, sort_hand_canonical, create_deck

app = Flask(__name__)

game_state = {
    "env": None,
    "obs_dict": None,
    "agent": None,
    "human_player_id": 0,
    "done": False,
    "deck": create_deck(),
    "logs": []
}

def get_relative_name(player_id, human_id):
    if player_id is None: return "None"
    if player_id == human_id: return "You"
    if player_id == (human_id + 1) % 4: return "Your Right"
    if player_id == (human_id + 2) % 4: return "Teammate"
    return "Your Left"

def add_log(msg):
    game_state["logs"].append(msg)

def run_ai_turn():
    env = game_state["env"]
    obs_dict = game_state["obs_dict"]
    agent = game_state["agent"]
    human_id = game_state["human_player_id"]
    
    if game_state["done"] or env.current_agent == human_id:
        return

    current_player = env.current_agent
    local_obs = flatten_obs(obs_dict, env.observation_space)
    global_state = get_global_state(env)
    mask = obs_dict["action_mask"]
    
    action, _, _ = agent.select_action(local_obs, global_state, mask)
    
    obs_dict, _, dones, _ = env.step(action)
    game_state["obs_dict"] = obs_dict
    game_state["done"] = dones.get('__all__', False)
    
    action_str = translate_action(action)
    p_name = get_relative_name(current_player, human_id)
    add_log(f"{p_name} played: {action_str}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    env = BalootMultiAgentEnv()
    obs_dict = env.reset()
    
    game_state["env"] = env
    game_state["obs_dict"] = obs_dict
    game_state["done"] = False
    game_state["logs"] = ["Game Started!"]
    
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(env, obs_dict)
    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    
    model_path = os.path.join("models", "mappo_update_100.h5")
    if os.path.exists(model_path):
        agent.model.load_weights(model_path)
        add_log(f"Model loaded: {model_path}")
    else:
        add_log("Failed to load model! Path not found.")
        
    game_state["agent"] = agent
    
    return jsonify({"status": "started"})

@app.route('/state', methods=['GET'])
def get_state():
    env = game_state["env"]
    if env is None:
        return jsonify({"error": "Game not started"}), 400

    if request.args.get("advance_ai") == "1":
        run_ai_turn()
        
    human_id = game_state["human_player_id"]
    deck = game_state["deck"]
    
    state = {
        "round": env.round_count,
        "scores": env.cumulative_scores,
        "phase": env.phase,
        "dealer": get_relative_name(env.dealer, human_id),
        "trick_leader": get_relative_name(getattr(env, 'trick_leader', None), human_id),
        "done": game_state["done"],
        "is_human_turn": not game_state["done"] and env.current_agent == human_id,
        "logs": game_state["logs"][-15:],
    }
    
    if env.phase == 'bidding':
        face_up_idx = deck.index(env.face_up) if hasattr(env, 'face_up') and env.face_up else None
        state["face_up"] = translate_action(face_up_idx) if face_up_idx is not None else None
        state["buyer"] = get_relative_name(env.buyer, human_id) if hasattr(env, 'buyer') and env.buyer is not None else None
        state["game_type"] = getattr(env, 'game_type', None)
        state["doubling_state"] = getattr(env, 'doubling_state', None)
    else:
        state["contract"] = getattr(env, 'game_type', None)
        state["buyer"] = get_relative_name(getattr(env, 'buyer', None), human_id)
        state["trump"] = getattr(env, 'trump_suit', None)
        state["doubling_state"] = getattr(env, 'doubling_state', None)
        
        trick = []
        for p_idx, card in enumerate(getattr(env, 'current_trick', [])):
            p_name = get_relative_name(p_idx, human_id)
            if card:
                trick.append({"player": p_name, "card": translate_action(deck.index(card))})
            else:
                trick.append({"player": p_name, "card": None})
        state["trick"] = trick

        trick_history = []
        for i, trick_data in enumerate(getattr(env, 'trick_history', [])):
            h_cards = trick_data.get("cards", [])
            w_idx = trick_data.get("winner")
            
            t_cards = []
            for p_idx, card in enumerate(h_cards):
                p_name = get_relative_name(p_idx, human_id)
                t_cards.append({
                    "player": p_name,
                    "card": translate_action(deck.index(card)) if card else None
                })
            
            trick_history.append({
                "trick_number": i + 1,
                "winner": w_idx if w_idx is None else get_relative_name(w_idx, human_id),
                "cards": t_cards
            })
        state["trick_history"] = trick_history

    valid_actions = []
    valid_indices = set()
    if state["is_human_turn"]:
        mask = game_state["obs_dict"]['action_mask']
        valid_indices = set(map(int, np.where(mask == 1)[0]))
        for idx in valid_indices:
            valid_actions.append({"index": int(idx), "text": translate_action(idx)})
    state["valid_actions"] = valid_actions

    hand = env.hands[human_id]
    sorted_hand = sort_hand_canonical(hand)
    state["hand"] = []
    for card in sorted_hand:
        action_index = int(deck.index(card))
        is_playable_card = action_index in valid_indices
        state["hand"].append({
            "text": translate_action(action_index),
            "action_index": action_index if is_playable_card else None
        })
    
    return jsonify(state)

@app.route('/action', methods=['POST'])
def take_action():
    env = game_state["env"]
    human_id = game_state["human_player_id"]
    
    if env is None or game_state["done"] or env.current_agent != human_id:
        return jsonify({"error": "Invalid action request"}), 400
        
    data = request.json
    action = data.get("action_index")
    
    obs_dict, _, dones, _ = env.step(action)
    game_state["obs_dict"] = obs_dict
    game_state["done"] = dones.get('__all__', False)
    
    action_str = translate_action(action)
    add_log(f"You played: {action_str}")
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
