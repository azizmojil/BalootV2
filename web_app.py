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
from env.utils import flatten_obs, get_global_state, infer_model_dimensions
from env.utils import translate_action, sort_hand_canonical, create_deck

app = Flask(__name__)

# Human-friendly labels for the internal set type identifiers.
SET_DISPLAY_MAP = {"Sera": "Sira", "Khamseen": "50", "Mia_c": "100", "Mia_s": "100", "Arbamia": "400"}

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

def get_relative_position(player_id, human_id):
    """Stable position key used by the frontend to anchor seat-specific UI."""
    if player_id is None: return None
    if player_id == human_id: return "you"
    if player_id == (human_id + 1) % 4: return "right"
    if player_id == (human_id + 2) % 4: return "teammate"
    return "left"

def get_last_bids(env, human_id):
    """Most recent bid action each player took, keyed by relative position."""
    last_bids = {}
    for actor, action in getattr(env, 'bidding_history', []):
        pos = get_relative_position(actor, human_id)
        if pos is not None:
            last_bids[pos] = translate_action(action)
    return last_bids

def get_player_sets(env, human_id):
    """Sets to surface per player (only winning-team sets survive _resolve_sets), keyed by position."""
    sets_by_pos = {}
    declared_info = getattr(env, 'declared_sets_info', None)
    if not declared_info:
        return sets_by_pos
        
    for p_idx, p_sets in enumerate(declared_info):
        if not p_sets:
            continue
            
        if not getattr(env, 'sets_resolved', False):
            has_played = env.current_trick[p_idx] is not None
            is_turn = (env.current_agent == p_idx)
            if not has_played and not is_turn:
                continue

        pos = get_relative_position(p_idx, human_id)
        if pos is None:
            continue
            
        pos_sets = []
        for s in p_sets:
            name = SET_DISPLAY_MAP.get(s["type"], s["type"])
            if getattr(env, 'sets_resolved', False) and "cards" in s:
                cards_str = " ".join([f"{c[0]}{c[1]}" for c in s["cards"]])
                name = f"{name} ({cards_str})"
            pos_sets.append(name)
        sets_by_pos[pos] = pos_sets
        
    return sets_by_pos

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
    if hasattr(env, 'resolution_logs') and env.resolution_logs:
        game_state["logs"].extend(env.resolution_logs)
        env.resolution_logs.clear()
        
    game_state["obs_dict"] = obs_dict
    game_state["done"] = dones.get('__all__', False)
    
    action_str = translate_action(action)
    p_name = get_relative_name(current_player, human_id)
    add_log(f"{p_name} played: {action_str}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def list_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return jsonify([])
    files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.keras')]
    return jsonify(files)

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json or {}
    selected_model = data.get("model", "mappo_update_100.h5")

    env = BalootMultiAgentEnv()
    obs_dict = env.reset()
    
    game_state["env"] = env
    game_state["obs_dict"] = obs_dict
    game_state["done"] = False
    game_state["logs"] = ["Game Started!"]
    
    local_obs_dim, global_state_dim, act_dim = infer_model_dimensions(env, obs_dict)
    agent = MAPPOAgent(local_obs_dim, global_state_dim, act_dim, build_mappo_network)
    
    model_path = os.path.join("models", selected_model)
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
        "dealer_position": get_relative_position(env.dealer, human_id),
        "trick_leader": get_relative_name(getattr(env, 'trick_leader', None), human_id),
        "trick_count": getattr(env, 'trick_count', 0),
        "last_round_score": getattr(env, 'last_round_score', None),
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
        state["last_bids"] = get_last_bids(env, human_id)
    else:
        state["contract"] = getattr(env, 'game_type', None)
        state["buyer"] = get_relative_name(getattr(env, 'buyer', None), human_id)
        state["trump"] = getattr(env, 'trump_suit', None)
        state["doubling_state"] = getattr(env, 'doubling_state', None)
        state["player_sets"] = get_player_sets(env, human_id)
        
        trick = []
        for p_idx, card in enumerate(getattr(env, 'current_trick', [])):
            p_name = get_relative_name(p_idx, human_id)
            if card:
                trick.append({"player": p_name, "card": translate_action(deck.index(card))})
            else:
                trick.append({"player": p_name, "card": None})
        state["trick"] = trick

        # Last completed trick, so the frontend can hold it on the table briefly before clearing.
        last_trick = []
        for p_idx, card in enumerate(getattr(env, 'last_trick', []) or []):
            if card:
                last_trick.append({"player": get_relative_name(p_idx, human_id),
                                   "card": translate_action(deck.index(card))})
            else:
                last_trick.append({"player": get_relative_name(p_idx, human_id), "card": None})
        state["last_trick"] = last_trick
        last_winner = None
        if getattr(env, 'trick_history', None):
            last_winner = env.trick_history[-1].get("winner")
        state["last_trick_winner"] = get_relative_position(last_winner, human_id) if last_winner is not None else None

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
        valid_indices = set(np.where(mask == 1)[0])
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

    # Provide opponent hand card counts for the table layout
    state["opponent_hand_counts"] = {
        "right": len(env.hands[(human_id + 1) % 4]),
        "teammate": len(env.hands[(human_id + 2) % 4]),
        "left": len(env.hands[(human_id + 3) % 4]),
    }
    
    def get_hand_strings(p_id):
        return [translate_action(int(deck.index(c))) for c in sort_hand_canonical(env.hands[p_id])]

    state["opponent_hands_revealed"] = {
        "right": get_hand_strings((human_id + 1) % 4),
        "teammate": get_hand_strings((human_id + 2) % 4),
        "left": get_hand_strings((human_id + 3) % 4),
    }

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
    if hasattr(env, 'resolution_logs') and env.resolution_logs:
        game_state["logs"].extend(env.resolution_logs)
        env.resolution_logs.clear()
        
    game_state["obs_dict"] = obs_dict
    game_state["done"] = dones.get('__all__', False)
    
    action_str = translate_action(action)
    add_log(f"You played: {action_str}")
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
