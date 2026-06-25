import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from env.utils import *
from env.rewards import calculate_trick_reward, calculate_end_of_round_reward, calculate_end_of_game_reward, calculate_bidding_reward, REWARD_ALL_PASS_PENALTY


class BalootMultiAgentEnv(gym.Env):
    """Baloot environment with explicit per-round bidding and trick state."""

    metadata = {"render_modes": ["human"]}
    NUM_PLAYERS = OBSERVATION_NUM_PLAYERS
    BIDDING_HISTORY_LENGTH = OBSERVATION_BIDDING_HISTORY_LENGTH
    BIDDING_HISTORY_FEATURES = OBSERVATION_BIDDING_HISTORY_FEATURES
    INFERENCE_EPSILON = 1e-3
    SET_INFERENCE_STRENGTH = 2.0
    OBSERVATION_SCHEMA = OBSERVATION_SCHEMA

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        self.cumulative_scores = [0, 0]
        self.round_count = 0
        self.match_over = False
        self.dealer = self._rng.randint(0, 3)
        # Last player who raised the doubling state; None when the contract is not doubled.
        self.last_doubler = None
        # Player order for cards in the active trick; None before a trick is led.
        self.trick_order = None
        # Player order for cards in last_trick; None until a trick has completed.
        self.last_trick_order = None
        self.action_space = spaces.Discrete(44)
        spaces_dict = {
            name: spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)
            for name, shape in self.OBSERVATION_SCHEMA.items()
        }
        self.observation_space = spaces.Dict(spaces_dict)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng.seed(seed)
        self.cumulative_scores = [0, 0]
        self.round_count = 0
        self.match_over = False
        self.dealer = self._rng.randint(0, 3)
        self.last_doubler = None
        self.trick_order = None
        self.last_trick_order = None
        return self._reset_round()

    def _reset_round(self):
        self.round_count += 1
        if self.round_count > 1:
            self.dealer = (self.dealer + 1) % 4
        self.phase = 'bidding'
        self.bidding_round = 1
        self.pass_count = 0
        self.initial_bid = None
        self.final_bid = None
        self.doubling_state = None
        self.last_doubler = None
        self.buyer = None
        self.game_type = None
        self.trump_suit = None
        self.card_ownership = np.zeros((32, 4, 4), dtype=np.float32)
        self.remaining_cards = np.ones((32,), dtype=np.float32)
        self.deck = create_deck()
        self._rng.shuffle(self.deck)
        canonical_deck = create_deck()
        self.hands = []
        for p in range(4):
            hand = [self.deck.pop(0) for _ in range(5)]
            self.hands.append(hand)
        for p, hand in enumerate(self.hands):
            for card in hand:
                idx = canonical_deck.index(card)
                self._set_known_card_owner(idx, p, observers=[p])
        self.face_up = self.deck.pop(0)
        # Active trick play order; set after the leader plays the first card.
        self.trick_order = self._default_player_order()
        self.trick_suit = None
        self.trick_count = 0
        self.trick_leader = (self.dealer + 1) % 4
        self.current_agent = self.trick_leader
        self.current_trick = [None] * 4
        self.last_trick = [None] * 4
        self.last_trick_order = None
        self.trick_history = []
        self.bidding_history = []
        self.declared_sets = np.zeros((4, 4), dtype=np.float32)
        self.revealed_sets = np.zeros((4, 5), dtype=np.float32)
        self.declared_sets_info = None
        self.set_declaration_done = [False] * 4
        self.sets_resolved = False
        self.set_resolution_reveals = {}
        self.public_revealed_set_keys = [set() for _ in range(4)]
        self.resolution_logs = []
        self.balot = [False] * 4
        self.detect_balot = [None] * 4
        self.team_bant = [0, 0]
        self.final_scores = [0, 0]
        self.team_tricks = [0, 0]
        self.last_trick_reward = {f'player_{i}': 0 for i in range(4)}
        self.bidding_order = np.array(compute_bidding_order(self.dealer), dtype=np.int32)
        self._refresh_card_ownership_beliefs()
        return self.get_observation()

    def _set_known_card_owner(self, card_idx, owner, observers=None):
        if not 0 <= card_idx < 32:
            raise ValueError(f"card_idx must be in [0, 32), got {card_idx}")
        if not 0 <= owner < 4:
            raise ValueError(f"owner must be in [0, 4), got {owner}")
        if observers is None:
            observers = range(4)
        for observer in observers:
            self.card_ownership[card_idx, :, observer] = 0.0
            self.card_ownership[card_idx, owner, observer] = 1.0

    def _clear_card_owner_belief(self, card_idx, observers=None):
        if not 0 <= card_idx < 32:
            raise ValueError(f"card_idx must be in [0, 32), got {card_idx}")
        if observers is None:
            observers = range(4)
        for observer in observers:
            self.card_ownership[card_idx, :, observer] = 0.0

    def _eliminate_card_owner(self, card_idx, owner, observers=None):
        if not 0 <= card_idx < 32:
            raise ValueError(f"card_idx must be in [0, 32), got {card_idx}")
        if not 0 <= owner < 4:
            raise ValueError(f"owner must be in [0, 4), got {owner}")
        if observers is None:
            observers = range(4)
        for observer in observers:
            if np.isclose(self.card_ownership[card_idx, owner, observer], 1.0):
                continue
            self.card_ownership[card_idx, owner, observer] = 0.0

    def _eliminate_void_suit(self, player, suit):
        canonical_deck = create_deck()
        for card_idx, card in enumerate(canonical_deck):
            if card[0] == suit and self.remaining_cards[card_idx] == 1:
                self._eliminate_card_owner(card_idx, player)

    def _is_known_to_observer(self, card_idx, observer):
        return np.any(np.isclose(self.card_ownership[card_idx, :, observer], 1.0))

    def _refresh_card_ownership_beliefs(self, observers=None):
        if observers is None:
            observers = range(4)

        canonical_deck = create_deck()
        face_up_idx = canonical_deck.index(self.face_up) if self.face_up is not None else None
        hand_sizes = np.array([len(hand) for hand in self.hands], dtype=np.float32)

        for observer in observers:
            known_remaining = np.zeros(4, dtype=np.float32)
            for card_idx in range(32):
                if self.remaining_cards[card_idx] == 0:
                    continue
                owners = np.where(np.isclose(self.card_ownership[card_idx, :, observer], 1.0))[0]
                if len(owners) == 1:
                    known_remaining[owners[0]] += 1.0

            hidden_slots = np.array([
                max(0.0, hand_sizes[player] - known_remaining[player])
                for player in range(4)
            ], dtype=np.float32)
            total_hidden_slots = hidden_slots.sum()

            for card_idx in range(32):
                if self.remaining_cards[card_idx] == 0:
                    continue
                if face_up_idx == card_idx and self.buyer is None:
                    self._clear_card_owner_belief(card_idx, observers=[observer])
                    continue
                if self._is_known_to_observer(card_idx, observer):
                    continue
                if np.isclose(total_hidden_slots, 0.0, rtol=0, atol=self.INFERENCE_EPSILON):
                    continue

                mask = (self.card_ownership[card_idx, :, observer] > 0).astype(np.float32)
                prior = mask * hidden_slots
                prior_sum = prior.sum()
                if prior_sum <= 0:
                    prior = hidden_slots
                    prior_sum = total_hidden_slots
                self.card_ownership[card_idx, :, observer] = prior / prior_sum

    def _relative_player_one_hot(self, player, observer, include_none=False):
        rel = relative_player_index(player, observer)
        if include_none:
            return one_hot_index(self.NUM_PLAYERS if rel is None else rel, self.NUM_PLAYERS + 1)
        return one_hot_index(rel, self.NUM_PLAYERS)

    def _default_player_order(self):
        return list(range(self.NUM_PLAYERS))

    def _relative_rows(self, values, observer):
        rows = relative_player_order(observer, self.NUM_PLAYERS)
        return values[rows]

    def _bid_action_one_hot(self, action, include_none=False):
        size = len(BID_ACTIONS) + (1 if include_none else 0)
        vec = np.zeros(size, dtype=np.float32)
        if action in BID_ACTIONS:
            vec[BID_ACTIONS.index(action)] = 1.0
        elif include_none:
            vec[-1] = 1.0
        return vec

    def _hand_mask(self, hand):
        canonical_deck = create_deck()
        return np.array([1.0 if card in hand else 0.0 for card in canonical_deck],
                        dtype=np.float32)

    def _cards_by_player_order(self, cards_by_player, order):
        if order is None:
            return np.zeros(128, dtype=np.float32)
        return np.concatenate([
            one_hot_card(cards_by_player[player])
            if cards_by_player[player] is not None
            else np.zeros(32, dtype=np.float32)
            for player in order
        ]).astype(np.float32)

    def _bidding_history_features(self, observer):
        features = []
        entries = list(self.bidding_history[-self.BIDDING_HISTORY_LENGTH:])
        # Recent recorded bids keep chronological order; unused trailing slots are zero-padded.
        while len(entries) < self.BIDDING_HISTORY_LENGTH:
            entries.append((None, None))
        for item in entries:
            actor, action = item
            if actor is None:
                actor_feat = np.zeros(self.NUM_PLAYERS, dtype=np.float32)
            else:
                actor_feat = self._relative_player_one_hot(actor, observer)
            features.append(actor_feat)
            features.append(self._bid_action_one_hot(action))
        return np.concatenate(features).astype(np.float32)

    def _validate_observation(self, obs):
        validate_observation(obs, self.OBSERVATION_SCHEMA, self.INFERENCE_EPSILON)

    def get_observation(self):
        if not hasattr(self, "hands"):
            raise RuntimeError("Call reset() before requesting an observation.")
        ag = self.current_agent

        player_roles = np.concatenate([
            self._relative_player_one_hot(self.dealer, ag),
            self._relative_player_one_hot((ag + 2) % 4, ag),
            self._relative_player_one_hot(self.buyer, ag, include_none=True),
            self._relative_player_one_hot(self.trick_leader, ag),
            self._relative_player_one_hot(self.last_doubler, ag, include_none=True),
        ]).astype(np.float32)

        phase_map = {'bidding': 0, 'playing': 1}
        gt_map = {None: 0, 'Sun': 1, 'Hukoom': 2}
        trump_map = {None: 0, '♠': 1, '♥': 2, '♦': 3, '♣': 4}
        ds_map = {None: 0, 'Double': 1, 'Three': 2, 'Four': 3, 'Gahwa': 4}
        game_context = np.concatenate([
            one_hot_index(phase_map[self.phase], 2),
            one_hot_index(gt_map[self.game_type], 3),
            one_hot_index(trump_map[self.trump_suit], 5),
            one_hot_index(ds_map[self.doubling_state], 5),
            self._bid_action_one_hot(self.initial_bid, include_none=True),
            self._bid_action_one_hot(self.final_bid, include_none=True),
        ]).astype(np.float32)

        own_team = team(ag)
        opp_team = 1 - own_team
        score_context = np.array([
            min(self.bidding_round, 2) / 2.0,
            min(self.pass_count, 8) / 8.0,
            len(self.hands[ag]) / 8.0,
            min(self.trick_count, 8) / 8.0,
            np.clip(self.cumulative_scores[own_team] / TARGET_SCORE, 0.0, 1.0),
            np.clip(self.cumulative_scores[opp_team] / TARGET_SCORE, 0.0, 1.0),
            np.clip((self.cumulative_scores[own_team] - self.cumulative_scores[opp_team] + TARGET_SCORE) / (2 * TARGET_SCORE), 0.0, 1.0),
            min(self.team_bant[own_team], 162) / 162.0,
            min(self.team_bant[opp_team], 162) / 162.0,
            min(self.team_tricks[own_team], 8) / 8.0,
        ], dtype=np.float32)

        own_hand = self._hand_mask(self.hands[ag])
        played_cards = (1.0 - self.remaining_cards).astype(np.float32)
        unknown_cards = np.clip(self.remaining_cards - own_hand, 0.0, 1.0).astype(np.float32)

        relative_owner_order = relative_player_order(ag, self.NUM_PLAYERS)
        own_knowledge = self.card_ownership[:, relative_owner_order, ag].astype(np.float32)
        own_knowledge_flat = own_knowledge.flatten()

        current_order = self.trick_order
        trick_feat = self._cards_by_player_order(self.current_trick, current_order)
        last_trick_feat = self._cards_by_player_order(self.last_trick, self.last_trick_order)

        declared = (self._relative_rows(self.declared_sets, ag) / 2).astype(np.float32).flatten()
        revealed = (self._relative_rows(self.revealed_sets, ag) / 2).astype(np.float32).flatten()

        mask = (self._bidding_action() if self.phase == 'bidding'
                else self._playing_action()).astype(np.float32)

        obs = {'player_roles': player_roles,
               'game_context': game_context,
               'score_context': score_context,
               'faceup_card': one_hot_card(self.face_up),
               'own_hand': own_hand,
               'played_cards': played_cards,
               'unknown_cards': unknown_cards,
               'cards_ownership': own_knowledge_flat,
               'trick': trick_feat,
               'last_trick': last_trick_feat,
               'declared_sets': declared,
               'revealed_sets': revealed,
               'bidding_history': self._bidding_history_features(ag),
               'action_mask': mask}
        self._validate_observation(obs)
        return obs

    def _bidding_action(self):
        if self.buyer is None:
            return initial_bidding_actions(current_agent=self.current_agent,
                                           dealer=self.dealer,
                                           bidding_round=self.bidding_round,
                                           face_up=self.face_up,
                                           agent_hand=self.hands[self.current_agent])

        overbid_mask = allowed_overbids(buyer=self.buyer,
                                        dealer=self.dealer,
                                        bid_type=self.game_type,
                                        doubling_status=self.doubling_state,
                                        bidding_round=self.bidding_round,
                                        agent=self.current_agent,
                                        face_up=self.face_up,
                                        agent_hand=self.hands[self.current_agent],
                                        trump_suit=self.trump_suit)

        doubling_mask = allowed_doubling_action(buy_type=self.game_type,
                                                buyer=self.buyer,
                                                agent=self.current_agent,
                                                cumulative_scores=self.cumulative_scores,
                                                current_doubling_state=self.doubling_state,
                                                last_doubler=self.last_doubler)

        if self.doubling_state is None:
            return np.maximum(overbid_mask, doubling_mask)
        return doubling_mask

    def _playing_action(self):
        if self.game_type != "Hukoom":
            return enforce_follow_suit_mask(self.hands[self.current_agent],
                                            self.trick_suit)

        return get_full_play_mask_hukoom(agent_hand=self.hands[self.current_agent],
                                         current_trick=self.current_trick,
                                         agent=self.current_agent,
                                         trick_suit=self.trick_suit,
                                         trump_suit=self.trump_suit,
                                         doubling_state=self.doubling_state)

    def _bidding_step(self, agent, action):
        self.bidding_history.append((agent, action))
        action_to_suit = {34: '♠', 35: '♥', 36: '♦', 37: '♣'}
        
        if action == 43:
            self.takweesh = True
            return
            
        if self.buyer is None:
            if action == 32:
                self.pass_count += 1
            elif action == 38:
                self.buyer = (agent + 2) % 4
                self.game_type = "Sun"
                self.trump_suit = None
                self.pass_count = 0
                self.initial_bid = action
            elif action in (34, 35, 36, 37):
                self.buyer = agent
                self.game_type = "Hukoom"
                self.trump_suit = action_to_suit.get(action)
                self.pass_count = 0
                self.initial_bid = action
            else:
                self.buyer = agent
                self.game_type = "Sun"
                self.trump_suit = None
                self.pass_count = 0
                self.initial_bid = action
        else:
            if action in (39, 40, 41, 42):
                doubling_map = {39: "Double", 40: "Three", 41: "Four", 42: "Gahwa"}
                self.doubling_state = doubling_map[action]
                self.last_doubler = agent
                self.pass_count = 0
            elif action == 32:
                self.pass_count += 1
            elif action == 38:
                self.buyer = (agent + 2) % 4
                self.game_type = "Sun"
                self.trump_suit = None
                self.pass_count = 0
            elif action == 33:
                self.buyer = agent
                self.game_type = "Sun"
                self.trump_suit = None
                self.final_bid = action
                self.pass_count = 0
            elif action in (34, 35, 36, 37):
                self.buyer = agent
                self.game_type = "Hukoom"
                self.trump_suit = action_to_suit.get(action)
                self.final_bid = action
                self.pass_count = 0
        self.current_agent = (self.current_agent + 1) % 4
        if self.pass_count == 4 and self.buyer is None:
            self.bidding_round = 2
        if self.pass_count == 8:
            return

        if self.pass_count == 4 and self.buyer is not None:
            self.phase = "playing"
            canonical_deck = create_deck()
            self.hands[self.buyer].append(self.face_up)
            idx = canonical_deck.index(self.face_up)
            self._set_known_card_owner(idx, self.buyer)
            for _ in range(2):
                card = self.deck.pop(0)
                self.hands[self.buyer].append(card)
                idx = canonical_deck.index(card)
                self._set_known_card_owner(idx, self.buyer, observers=[self.buyer])
            for p in range(4):
                if p == self.buyer:
                    continue
                for _ in range(3):
                    card = self.deck.pop(0)
                    self.hands[p].append(card)
                    idx = canonical_deck.index(card)
                    self._set_known_card_owner(idx, p, observers=[p])
            for p in range(4):
                self.hands[p] = sort_hand_canonical(self.hands[p])
            self._refresh_card_ownership_beliefs()

            self.declared_sets_info = [detect_sets(self.hands[p]) for p in range(4)]
            check_set_balot(self.declared_sets_info, self.trump_suit, self.balot)
            self.current_agent = self.trick_leader

    def _playing_step(self, agent, action):
        canonical = create_deck()
        if self.trick_count == 0:
            self._declare_sets_for_player(agent)
        if self.trick_count == 1:
            if not self.sets_resolved:
                self._resolve_sets_by_second_trick_reveals(agent)
            for s in self.declared_sets_info[agent]:
                self._reveal_declared_set(agent, s, canonical)

        chosen_card = canonical[action]
        had_trick_suit = (
            self.trick_suit is not None
            and any(card[0] == self.trick_suit for card in self.hands[agent])
        )
        proved_void_in_trick_suit = (
            self.trick_suit is not None
            and chosen_card[0] != self.trick_suit
            and not had_trick_suit
        )
        self.hands[agent].remove(chosen_card)
        idx = canonical.index(chosen_card)
        if proved_void_in_trick_suit:
            self._eliminate_void_suit(agent, self.trick_suit)
        self.remaining_cards[idx] = 0.0
        self._set_known_card_owner(idx, agent)
        self._refresh_card_ownership_beliefs()
        self._infer_all_cards()

        if self.game_type == "Hukoom" and self.trump_suit:
            suit, rank = chosen_card
            if suit == self.trump_suit and rank in ("K", "Q"):
                prev = self.detect_balot[agent]
                if prev is None:
                    if not any(s.get("type") in ("Mia_s", "Mia_c") and chosen_card in s.get("cards", [])
                               for s in self.declared_sets_info[agent]):
                        self.detect_balot[agent] = rank
                else:
                    complement = "Q" if prev == "K" else "K"
                    comp_card = (self.trump_suit, complement)
                    if rank != prev and not any(s.get("type") in ("Mia_s", "Mia_c") and comp_card in s.get("cards", [])
                                                for s in self.declared_sets_info[agent]):
                        self.balot[agent] = True

        if self.trick_suit is None and agent == self.trick_leader:
            self.trick_suit = chosen_card[0]
            self.trick_order = [(self.trick_leader + i) % 4 for i in range(4)]

        self.current_trick[agent] = chosen_card
        self.current_agent = (self.current_agent + 1) % 4

        if all(c is not None for c in self.current_trick):
            self.last_trick = list(self.current_trick)
            self.last_trick_order = list(self.trick_order)

            winner = self._evaluate_trick_winner()
            self.trick_history.append({"cards": list(self.current_trick), "winner": winner})
            trick_points = sum((SUN_POINTS[c[1]] if self.game_type == "Sun" or c[0] != self.trump_suit
                                else HUKOOM_POINTS[c[1]]) for c in self.current_trick)

            win_team = team(winner)
            self.team_bant[win_team] += trick_points
            self.team_tricks[win_team] += 1

            trick_rewards_arr = calculate_trick_reward(self.last_trick, winner, self.game_type, self.trump_suit)
            self.last_trick_reward = {f"player_{i}": trick_rewards_arr[i] for i in range(4)}

            self.current_trick = [None] * 4
            self.trick_leader = winner
            self.current_agent = winner
            self.trick_suit = None
            self.trick_count += 1

            if self.trick_count >= 8:
                self.team_bant[win_team] += 10

            if self.trick_count == 1 and not self.sets_resolved:
                self._resolve_sets_after_first_trick()

    def _evaluate_trick_winner(self):
        lead = self.trick_suit
        trump = self.trump_suit
        plays = [(card, idx) for idx, card in enumerate(self.last_trick) if card is not None]

        def score(card):
            suit, rank = card
            is_trump = 1 if suit == trump else 0
            is_lead = 1 if suit == lead else 0
            val = (HUKOOM_ORDER[rank] if is_trump else SUN_ORDER[rank])
            return is_trump, is_lead, val

        _, winner_idx = max(plays, key=lambda ci: score(ci[0]))
        return winner_idx

    def step(self, action):
        acting_agent = self.current_agent
        obs_dict = self.get_observation()
        if obs_dict["action_mask"][action] == 0:
            raise ValueError(f"Agent {acting_agent} attempted invalid action {action} in phase '{self.phase}'.")

        if self.phase == "bidding":
            bidding_reward = calculate_bidding_reward(self, acting_agent, action)
            self._bidding_step(acting_agent, action)

            if getattr(self, "takweesh", False):
                rewards = {f"player_{i}": 0.0 for i in range(4)}
                rewards[f"player_{acting_agent}"] = bidding_reward
                dones = {f"player_{i}": self.match_over for i in range(4)}
                self._reset_round()
                self.takweesh = False
            elif self.pass_count >= 8:
                rewards = {f"player_{i}": REWARD_ALL_PASS_PENALTY for i in range(4)}
                dones = {f"player_{i}": self.match_over for i in range(4)}
                self._reset_round()
            else:
                rewards = {f"player_{i}": 0.0 for i in range(4)}
                rewards[f"player_{acting_agent}"] = bidding_reward
                dones = {f"player_{i}": self.match_over for i in range(4)}
        else:
            previous_trick_count = self.trick_count
            self._playing_step(acting_agent, action)
            trick_completed = self.trick_count != previous_trick_count
            if self.trick_count >= 8:
                rewards = self._compute_score()
                self._update_cumulative_scores()
                dones = {f"player_{i}": self.match_over for i in range(4)}

                end_of_round_rewards_arr = calculate_end_of_round_reward(self)
                for i in range(4):
                    rewards[f"player_{i}"] += end_of_round_rewards_arr[i]
                    if hasattr(self, "last_trick_reward"):
                        rewards[f"player_{i}"] += self.last_trick_reward[f"player_{i}"]

                if self.match_over:
                    match_rewards = calculate_end_of_game_reward(self)
                    for i in range(4):
                        rewards[f"player_{i}"] += match_rewards[i]

                self._reset_round()
            else:
                if trick_completed:
                    rewards = getattr(self, "last_trick_reward", {f"player_{i}": 0.0 for i in range(4)})
                else:
                    rewards = {f"player_{i}": 0.0 for i in range(4)}
                dones = {f"player_{i}": self.match_over for i in range(4)}
        obs_dict = self.get_observation()
        infos = {f"player_{i}": {"cumulative_scores": self.cumulative_scores} for i in range(4)}

        if self.match_over:
            dones['__all__'] = True

        return obs_dict, rewards, dones, infos

    def _compute_score(self):
        if not self.sets_resolved:
            self._resolve_sets_by_full_information()

        team_set_bonus = [0, 0]
        team_balot_bonus = [0, 0]
        buyer_team = team(self.buyer)

        for p, sets in enumerate(self.declared_sets_info):
            for s in sets:
                bonus = (SET_BONUS_HUKOOM if self.game_type == "Hukoom"
                         else SET_BONUS_SUN)[s["type"]]
                team_set_bonus[team(p)] += bonus

        for p, got in enumerate(self.balot):
            if got:
                team_balot_bonus[team(p)] += 20

        divisor = 10 if self.game_type == "Hukoom" else 5
        set_bonus0 = (team_set_bonus[0] + team_balot_bonus[0]) // divisor
        set_bonus1 = (team_set_bonus[1] + team_balot_bonus[1]) // divisor

        base0 = convert_bant(self.team_bant[0], self.game_type)
        base1 = convert_bant(self.team_bant[1], self.game_type)
        total = [base0 + set_bonus0, base1 + set_bonus1]

        if self.game_type == "Hukoom" and self.team_bant[buyer_team] % 10 == 6:
            total[buyer_team] = max(0, total[buyer_team] - 1)

        final = [0, 0]
        if self.doubling_state:
            mult = {"Double": 2, "Three": 3, "Four": 4, "Gahwa": 999}[self.doubling_state]
            win_team = 0 if total[0] > total[1] else 1
            final[win_team] = ((BASE_SCORE_HUKOOM if self.game_type == "Hukoom"
                                else BASE_SCORE_SUN) * mult)
            if self.doubling_state == "Gahwa":
                final[win_team] = 152
        else:
            if total[buyer_team] < total[1 - buyer_team]:
                base = BASE_SCORE_HUKOOM if self.game_type == "Hukoom" else BASE_SCORE_SUN
                final = [0, 0]
                final[1 - buyer_team] = base + set_bonus0 + set_bonus1
            else:
                final = [total[0], total[1]]

        if max(self.team_tricks) == 8:
            kap_team = 0 if total[0] > total[1] else 1
            kap_bonus = (KAPUT_HUKOOM if self.game_type == "Hukoom" else KAPUT_SUN)
            final = [0, 0]
            final[kap_team] = kap_bonus + (set_bonus0 if kap_team == 0 else set_bonus1)

        self.last_round_score = final.copy()
        self.final_scores = final.copy()
        reward_scale = float(TARGET_SCORE)
        if reward_scale <= 1.0:
            raise ValueError(f"TARGET_SCORE must be greater than 1.0 for reward normalization. Current value: {reward_scale}")
        diff0 = (final[0] - final[1]) / reward_scale
        diff1 = (final[1] - final[0]) / reward_scale
        rewards = {f"player_{i}": float(diff0 if team(i) == 0 else diff1) for i in range(4)}

        return rewards

    def _update_cumulative_scores(self):
        self.cumulative_scores[0] += self.final_scores[0]
        self.cumulative_scores[1] += self.final_scores[1]
        if (self.cumulative_scores[0] >= TARGET_SCORE or self.cumulative_scores[1] >= TARGET_SCORE) \
                and self.cumulative_scores[0] != self.cumulative_scores[1]:
            self.match_over = True

    def _resolve_sets(self, declaring_agent=None):
        self._resolve_sets_by_full_information()

    def _declare_sets_for_player(self, player):
        if self.set_declaration_done[player]:
            return

        set_type_to_index = {"Sera": 0, "Khamseen": 1, "Mia_c": 2, "Mia_s": 2, "Arbamia": 3}
        for set_info in self.declared_sets_info[player]:
            set_index = set_type_to_index.get(set_info["type"])
            if set_index is None:
                raise ValueError(
                    f"Unknown set type for player {player}: {set_info['type']}. "
                    f"Valid types: {list(set_type_to_index.keys())}"
                )
            self.declared_sets[player, set_index] += 1.0
        self.set_declaration_done[player] = True

    def _filter_declared_sets_to_team(self, winning_team):
        revealed = [[] for _ in range(4)]
        for player, sets in enumerate(self.declared_sets_info):
            if team(player) == winning_team:
                revealed[player] = [set_info.copy() for set_info in sets]

        self.declared_sets_info = revealed
        self.sets_resolved = True

    def _clear_declared_sets(self):
        self.declared_sets_info = [[] for _ in range(4)]
        self.sets_resolved = True

    def _top_set_candidates(self):
        candidates = []
        top_priority = None
        for player, sets in enumerate(self.declared_sets_info):
            for set_info in sets:
                priority = set_category_priority(set_info)
                if top_priority is None or priority > top_priority:
                    candidates = [(player, set_info)]
                    top_priority = priority
                elif priority == top_priority:
                    candidates.append((player, set_info))
        return candidates

    def _resolve_sets_after_first_trick(self):
        candidates = self._top_set_candidates()
        if not candidates:
            self._clear_declared_sets()
            return

        candidate_teams = {team(player) for player, _ in candidates}
        if len(candidate_teams) == 1:
            self._filter_declared_sets_to_team(candidate_teams.pop())

    def _record_set_resolution_reveal(self, player, set_info):
        canonical = create_deck()
        self._reveal_declared_set(player, set_info, canonical)
        value = set_resolution_value(set_info)
        category = set_category(set_info["type"])
        if self.set_resolution_reveals:
            best_revealed = max(self.set_resolution_reveals.values(), key=lambda reveal: reveal["key"])
            if set_resolution_key(set_info) > best_revealed["key"]:
                self.resolution_logs.append(f"Player {player} replies with {set_value_label(value)}.")
            else:
                self.resolution_logs.append(
                    f"Player {player} cannot beat {set_value_label(best_revealed['value'])}."
                )
        else:
            self.resolution_logs.append(
                f"Player {player} asks with {set_value_label(value)} for {category}."
            )
        self.set_resolution_reveals[player] = {
            "type": set_info["type"],
            "category": category,
            "value": value,
            "priority": SET_PRIORITY[set_info["type"]],
            "key": set_resolution_key(set_info),
        }

    def _revealed_set_key(self, set_info):
        return (
            set_info["type"],
            tuple(sorted(tuple(card) for card in set_info["cards"])),
        )

    def _reveal_declared_set(self, player, set_info, canonical=None):
        if canonical is None:
            canonical = create_deck()

        reveal_key = self._revealed_set_key(set_info)
        if reveal_key in self.public_revealed_set_keys[player]:
            return

        sets_list = list(SET_PRIORITY.keys())
        i = sets_list.index(set_info["type"])
        self.revealed_sets[player, i] += 1.0
        self.public_revealed_set_keys[player].add(reveal_key)

        for card in set_info["cards"]:
            idx = canonical.index(card)
            self._set_known_card_owner(idx, player)

    def _resolve_sets_by_second_trick_reveals(self, start_player):
        candidates = self._top_set_candidates()
        if not candidates:
            self._clear_declared_sets()
            return

        candidate_players = {player for player, _ in candidates}
        turn_sequence = [(start_player + turn_offset) % 4 for turn_offset in range(4)]

        for player in turn_sequence:
            if player not in candidate_players or player in self.set_resolution_reveals:
                continue

            player_sets = [set_info for candidate_player, set_info in candidates if candidate_player == player]
            if not player_sets:
                continue
            best_set = max(
                player_sets,
                key=set_resolution_key,
            )
            self._record_set_resolution_reveal(player, best_set)

            best_revealed_player, best_revealed = max(
                self.set_resolution_reveals.items(),
                key=lambda item: item[1]["key"],
            )
            unrevealed_candidates = [
                set_info
                for candidate_player, set_info in candidates
                if candidate_player not in self.set_resolution_reveals
            ]
            best_unrevealed_key = (
                max(set_resolution_key(set_info) for set_info in unrevealed_candidates)
                if unrevealed_candidates else None
            )
            if best_unrevealed_key is None or best_revealed["key"] > best_unrevealed_key:
                self._filter_declared_sets_to_team(team(best_revealed_player))
                return

            if all(candidate_player in self.set_resolution_reveals for candidate_player, _ in candidates):
                self._resolve_sets_by_full_information()
                return

    def _resolve_sets_by_full_information(self):
        best_set = None
        best_player = None

        for player, sets in enumerate(self.declared_sets_info):
            for set_info in sets:
                if best_set is None:
                    best_set, best_player = set_info, player
                    continue

                current_key = set_resolution_key(set_info)
                best_key = set_resolution_key(best_set)
                if current_key > best_key:
                    best_set, best_player = set_info, player

        if best_player is None:
            self._clear_declared_sets()
        else:
            self._filter_declared_sets_to_team(team(best_player))

    def _hidden_declared_set_types(self, player):
        if not self.sets_resolved:
            return []

        declared = self.declared_sets[player]
        revealed_names = list(SET_PRIORITY.keys())
        revealed = {
            set_type: int(round(self.revealed_sets[player, idx]))
            for idx, set_type in enumerate(revealed_names)
        }
        hidden_counts = {
            "Sera": max(0, int(round(declared[0])) - revealed.get("Sera", 0)),
            "Khamseen": max(0, int(round(declared[1])) - revealed.get("Khamseen", 0)),
            # Declared Mia is a single public bucket; revealed Mia splits into consecutive and same-rank variants.
            # Mia_c and Mia_s are alternative revelations: a consecutive run, or all suits of 10/J/Q/K.
            "Mia": max(0, int(round(declared[2])) - revealed.get("Mia_c", 0) - revealed.get("Mia_s", 0)),
            "Arbamia": max(0, int(round(declared[3])) - revealed.get("Arbamia", 0)),
        }

        hidden_types = []
        for set_type in SET_TYPE_BY_INDEX:
            hidden_types.extend([set_type] * hidden_counts[set_type])
        return hidden_types

    def _possible_declared_sets(self, pool_cards, hidden_types_list):
        pool = set(pool_cards)
        hidden_types = set(hidden_types_list)
        candidates = []

        if "Arbamia" in hidden_types:
            aces = [(suit, "A") for suit in SUITS]
            if all(card in pool for card in aces):
                candidates.append(aces)

        if "Mia" in hidden_types:
            for rank in ("10", "J", "Q", "K"):
                same_rank = [(suit, rank) for suit in SUITS]
                if all(card in pool for card in same_rank):
                    candidates.append(same_rank)

        run_lengths = []
        if "Sera" in hidden_types:
            run_lengths.append(3)
        if "Khamseen" in hidden_types:
            run_lengths.append(4)
        if "Mia" in hidden_types:
            run_lengths.append(5)

        for suit in SUITS:
            for length in run_lengths:
                for start in range(len(RANKS) - length + 1):
                    run = [(suit, rank) for rank in RANKS[start:start + length]]
                    if all(card in pool for card in run):
                        candidates.append(run)

        return candidates

    def _infer_all_cards(self):
        for observer in range(self.NUM_PLAYERS):
            self._infer_cards(observer)

    def _infer_cards(self, agent):
        """Apply hidden declared-set likelihoods to one observer."""
        eps = self.INFERENCE_EPSILON
        canonical_deck = create_deck()
        card_to_idx = {card: idx for idx, card in enumerate(canonical_deck)}
        hidden = [c for c in range(32)
                  if self.remaining_cards[c] == 1
                  and np.isclose(self.card_ownership[c, :, agent].sum(), 1.0, rtol=0, atol=eps)
                  and not self._is_known_to_observer(c, agent)]

        for player in range(4):
            hidden_types = self._hidden_declared_set_types(player)
            if not hidden_types:
                continue

            known = [c for c in range(32)
                     if self.remaining_cards[c] == 1
                     and np.isclose(self.card_ownership[c, player, agent], 1.0)]

            played_by_player = [c for c in range(32)
                               if self.remaining_cards[c] == 0
                               and np.isclose(self.card_ownership[c, player, agent], 1.0)]

            pool_idxs = known + hidden + played_by_player
            pool_cards = [canonical_deck[c] for c in pool_idxs]
            candidates = self._possible_declared_sets(pool_cards, hidden_types)
            if not candidates:
                continue

            counts = np.zeros(32, dtype=np.float32)
            for candidate in candidates:
                for card in candidate:
                    counts[card_to_idx[card]] += 1.0

            max_count = counts.max()
            if max_count <= eps:
                continue

            for card_idx in hidden:
                prior = self.card_ownership[card_idx, :, agent]
                if prior[player] <= eps:
                    continue

                set_support = counts[card_idx] / max_count
                # Neutral means other owners keep their priors while the declaring owner is boosted.
                post = prior.copy()
                post[player] *= 1.0 + self.SET_INFERENCE_STRENGTH * set_support
                post_sum = post.sum()
                if post_sum > 0:
                    self.card_ownership[card_idx, :, agent] = post / post_sum
