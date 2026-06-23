import gymnasium as gym
from gymnasium import spaces
import random
from env.utils import *
from env.rewards import calculate_trick_reward, calculate_end_of_round_reward, calculate_end_of_game_reward, calculate_bidding_reward, REWARD_ALL_PASS_PENALTY


class BalootMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    INFERENCE_EPSILON = 1e-3
    SET_TYPE_BY_INDEX = ("Sera", "Khamseen", "Mia", "Arbamia")

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        self.cumulative_scores = [0, 0]
        self.round_count = 0
        self.match_over = False
        self.dealer = self._rng.randint(0, 3)
        self.action_space = spaces.Discrete(43)
        sample = self.reset()
        spaces_dict = {}
        for name, arr in sample.items():
            spaces_dict[name] = spaces.Box(0.0, 1.0,
                                           shape=arr.shape,
                                           dtype=np.float32)
        self.observation_space = spaces.Dict(spaces_dict)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng.seed(seed)
        self.cumulative_scores = [0, 0]
        self.round_count = 0
        self.match_over = False
        self.dealer = self._rng.randint(0, 3)
        return self._reset_round()

    def _reset_round(self):
        self.round_count += 1
        self.phase = 'bidding'
        self.bidding_round = 1
        self.pass_count = 0
        self.initial_bid = None
        self.final_bid = None
        self.doubling_state = None
        self.original_doubler = None
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
        self.trick_order = None
        self.trick_suit = None
        self.trick_count = 0
        self.trick_leader = (self.dealer + 1) % 4
        self.current_agent = self.trick_leader
        self.current_trick = [None] * 4
        self.last_trick = [None] * 4
        self.trick_history = []
        self.declared_sets = np.zeros((4, 4), dtype=np.float32)
        self.revealed_sets = np.zeros((4, 5), dtype=np.float32)
        self.declared_sets_info = None
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
                if total_hidden_slots <= 0:
                    raise ValueError(
                        "Cannot assign ownership belief for an unknown remaining card when no hidden "
                        "hand slots are available. This likely indicates a logic error in card tracking "
                        "or game state management. "
                        f"card_idx={card_idx}, observer={observer}; "
                        f"total_hidden_slots={total_hidden_slots}, hidden_slots={hidden_slots.tolist()}, "
                        f"hand_sizes={hand_sizes.tolist()}, known_remaining={known_remaining.tolist()}"
                    )

                prior = self.card_ownership[card_idx, :, observer] * hidden_slots
                prior_sum = prior.sum()
                if prior_sum <= 0:
                    prior = hidden_slots
                    prior_sum = total_hidden_slots
                self.card_ownership[card_idx, :, observer] = prior / prior_sum

    def get_observation(self):
        ag = self.current_agent

        who_am_i = np.eye(4, dtype=np.float32)[ag]
        dealer = np.eye(4, dtype=np.float32)[self.dealer]
        partner = np.eye(4, dtype=np.float32)[(ag + 2) % 4]
        trick_leader = np.eye(4, dtype=np.float32)[self.trick_leader]

        buyer = np.zeros(5, dtype=np.float32)
        buyer[4 if self.buyer is None else self.buyer] = 1.0

        remaining_tricks = np.array([len(self.hands[ag]) / 8.0], dtype=np.float32)
        bidding_progress = np.array([
            min(self.bidding_round, 2) / 2.0,
            min(self.pass_count, 8) / 8.0
        ], dtype=np.float32)
        score_context = np.clip(np.array(self.cumulative_scores, dtype=np.float32) / TARGET_SCORE, 0.0, 1.0)

        faceup_feat = one_hot_card(self.face_up)

        phase_map = {'bidding': 0, 'playing': 1}
        phase = np.eye(2, dtype=np.float32)[phase_map[self.phase]].flatten()

        gt_map = {None: 0, 'Sun': 1, 'Hukoom': 2}
        game_type = np.eye(3, dtype=np.float32)[gt_map[self.game_type]].flatten()

        trump_map = {None: 0, '♠': 1, '♥': 2, '♦': 3, '♣': 4}
        trump_suit = np.eye(5, dtype=np.float32)[trump_map[self.trump_suit]].flatten()

        ds_map = {None: 0, 'Double': 1, 'Three': 2, 'Four': 3, 'Gahwa': 4}
        doubling = np.eye(5, dtype=np.float32)[ds_map[self.doubling_state]].flatten()

        own_knowledge = self.card_ownership[:, :, ag].astype(np.float32)
        own_knowledge_flat = own_knowledge.flatten()

        trick_feat = np.concatenate([one_hot_card(c)
                                     if c is not None
                                     else np.zeros(32, dtype=np.float32)
                                     for c in self.current_trick])
        last_trick_feat = np.concatenate([one_hot_card(c)
                                          if c is not None
                                          else np.zeros(32, dtype=np.float32)
                                          for c in self.last_trick])
        played_cards = (1.0 - self.remaining_cards).astype(np.float32)

        declared = (self.declared_sets / 2).astype(np.float32).flatten()
        revealed = (self.revealed_sets / 2).astype(np.float32).flatten()

        mask = (self._bidding_action() if self.phase == 'bidding'
                else self._playing_action()).astype(np.float32)

        return {'identity': who_am_i,
                'dealer': dealer,
                'partner': partner,
                'buyer': buyer,
                'trick_leader': trick_leader,
                'countdown': remaining_tricks,
                'bidding_progress': bidding_progress,
                'score_context': score_context,
                'faceup_card': faceup_feat,
                'phase': phase,
                'game_type': game_type,
                'trump_suit': trump_suit,
                'doubling': doubling,
                'remaining_cards': self.remaining_cards,
                'played_cards': played_cards,
                'cards_ownership': own_knowledge_flat,
                'trick': trick_feat,
                'last_trick': last_trick_feat,
                'declared_sets': declared,
                'revealed_sets': revealed,
                'action_mask': mask}

    def _bidding_action(self):
        if self.buyer is None:
            return initial_bidding_actions(current_agent=self.current_agent,
                                           dealer=self.dealer,
                                           bidding_round=self.bidding_round,
                                           face_up=self.face_up)

        overbid_mask = allowed_overbids(buyer=self.buyer,
                                        dealer=self.dealer,
                                        bid_type=self.game_type,
                                        doubling_status=self.doubling_state,
                                        bidding_round=self.bidding_round,
                                        agent=self.current_agent,
                                        face_up=self.face_up)

        doubling_mask = allowed_doubling_action(buy_type=self.game_type,
                                                buyer=self.buyer,
                                                agent=self.current_agent,
                                                cumulative_scores=self.cumulative_scores,
                                                current_doubling_state=self.doubling_state,
                                                original_doubler=self.original_doubler)

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
        action_to_suit = {34: '♠', 35: '♥', 36: '♦', 37: '♣'}
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
                if self.doubling_state is not None:
                    self.original_doubler = agent
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
            # This indicates a failed round where everyone passed.
            # The step function will handle the reset and rewards.
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
            self._resolve_sets()
            self.current_agent = self.trick_leader

    def _playing_step(self, agent, action):
        canonical = create_deck()
        if self.trick_count == 0:
            sets = detect_sets(self.hands[agent])
            set_type_to_index = {"Sera": 0, "Khamseen": 1, "Mia_c": 2, "Mia_s": 2, "Arbamia": 3}
            for s in sets:
                set_index = set_type_to_index.get(s["type"])
                if set_index is None:
                    raise ValueError(f"Unknown set type: {s['type']}")
                self.declared_sets[agent, set_index] += 1.0
        if self.trick_count == 1:
            for s in self.declared_sets_info[agent]:
                sets_list = list(SET_PRIORITY.keys())
                i = sets_list.index(s["type"])
                self.revealed_sets[agent, i] += 1.0
                for card in s["cards"]:
                    idx = canonical.index(card)
                    self._set_known_card_owner(idx, agent)

        chosen_card = canonical[action]
        self.hands[agent].remove(chosen_card)
        idx = canonical.index(chosen_card)
        self.remaining_cards[idx] = 0.0
        self._set_known_card_owner(idx, agent)
        self._refresh_card_ownership_beliefs()
        self._infer_cards(agent)

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

            winner = self._evaluate_trick_winner()
            self.trick_history.append({"cards": list(self.current_trick), "winner": winner})
            trick_points = sum((SUN_POINTS[c[1]] if self.game_type == "Sun" or c[0] != self.trump_suit
                                else HUKOOM_POINTS[c[1]]) for c in self.current_trick)

            win_team = team(winner)
            self.team_bant[win_team] += trick_points
            self.team_tricks[win_team] += 1
            
            # Calculate dense trick rewards
            trick_rewards_arr = calculate_trick_reward(self.last_trick, winner, self.game_type, self.trump_suit)
            self.last_trick_reward = {f"player_{i}": trick_rewards_arr[i] for i in range(4)}

            self.current_trick = [None] * 4
            self.trick_leader = winner
            self.current_agent = winner
            self.trick_suit = None
            self.trick_count += 1

            if self.trick_count >= 8:
                self.team_bant[win_team] += 10

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
        # --- Strict Action Validation ---
        acting_agent = self.current_agent
        obs_dict = self.get_observation()
        if obs_dict["action_mask"][action] == 0:
            raise ValueError(f"Agent {acting_agent} attempted invalid action {action} in phase '{self.phase}'.")
            
        if self.phase == "bidding":
            # Calculate before _bidding_step mutates the bidder, phase, and hand state.
            bidding_reward = calculate_bidding_reward(self, acting_agent, action)
            self._bidding_step(acting_agent, action)

            # Check for failed round condition
            if self.pass_count >= 8:
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
            # A single play can complete at most one trick.
            trick_completed = self.trick_count != previous_trick_count
            if self.trick_count >= 8:
                rewards = self._compute_score()
                self._update_cumulative_scores()
                dones = {f"player_{i}": self.match_over for i in range(4)}
                
                # Add shaped end-of-round rewards and the final trick reward
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

        # Convert set and balot bant to final score points
        divisor = 10 if self.game_type == "Hukoom" else 5
        set_bonus0 = (team_set_bonus[0] + team_balot_bonus[0]) // divisor
        set_bonus1 = (team_set_bonus[1] + team_balot_bonus[1]) // divisor

        base0 = self._convert_bant(self.team_bant[0])
        base1 = self._convert_bant(self.team_bant[1])
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
                # If buyer loses, the opponent gets the base score PLUS all sets played in the round
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
        # Normalize round score-difference rewards by the match target score.
        reward_scale = float(TARGET_SCORE)
        if reward_scale <= 1.0:
            raise ValueError(f"TARGET_SCORE must be greater than 1.0 for reward normalization. Current value: {reward_scale}")
        diff0 = (final[0] - final[1]) / reward_scale
        diff1 = (final[1] - final[0]) / reward_scale
        rewards = {f"player_{i}": float(diff0 if team(i) == 0 else diff1) for i in range(4)}

        return rewards

    def _convert_bant(self, bant):
        if bant <= 0:
            return 0
        if self.game_type == "Hukoom":
            rem = bant % 10
            tens = bant // 10
            rounded = tens * 10 if rem == 5 else (tens *
                                                  10 if (bant - tens * 10) <= ((tens + 1) * 10 - bant)
                                                  else (tens + 1) * 10)
            return rounded // 10
        else:
            rem = bant % 10
            tens = bant // 10
            if rem == 5:
                return tens * 2 + 1
            else:
                lower = tens * 10
                upper = lower + 10
                rounded = lower if (bant - lower) <= (upper - bant) else upper
                return (rounded // 10) * 2

    def _update_cumulative_scores(self):
        self.cumulative_scores[0] += self.final_scores[0]
        self.cumulative_scores[1] += self.final_scores[1]
        if (self.cumulative_scores[0] >= TARGET_SCORE or self.cumulative_scores[1] >= TARGET_SCORE) \
                and self.cumulative_scores[0] != self.cumulative_scores[1]:
            self.match_over = True

    def _resolve_sets(self):
        best_set = None
        best_player = None

        for p, sets in enumerate(self.declared_sets_info):
            for s in sets:
                if best_set is None:
                    best_set, best_player = s, p
                else:
                    pri_s = SET_PRIORITY[s["type"]]
                    pri_best = SET_PRIORITY[best_set["type"]]
                    if pri_s > pri_best:
                        best_set, best_player = s, p
                    elif pri_s == pri_best:
                        if max(card_value(c) for c in s["cards"]) > \
                                max(card_value(c) for c in best_set["cards"]):
                            best_set, best_player = s, p

        if best_player is None:
            self.declared_sets_info = [[] for _ in range(4)]

        else:
            winning_team = team(best_player)
            revealed = [[] for _ in range(4)]
            for p, sets in enumerate(self.declared_sets_info):
                if team(p) == winning_team:
                    revealed[p] = [s.copy() for s in sets]

            self.declared_sets_info = revealed

    def _infer_cards(self, agent):
        eps = self.INFERENCE_EPSILON

        hidden = [c for c in range(32)
                  if self.remaining_cards[c] == 1
                  and np.isclose(self.card_ownership[c, :, agent].sum(), 1.0, atol=eps)
                  and not np.any(np.isclose(self.card_ownership[c, :, agent], 1.0))]

        for player in range(4):
            to_find = int(self.declared_sets[player].sum()
                          - self.revealed_sets[player].sum())
            if to_find <= 0:
                continue

            known = [c for c in range(32)
                     if np.isclose(self.card_ownership[c, player, agent], 1.0)]

            pool_idxs = known + hidden
            pool_cards = [create_deck()[c] for c in pool_idxs]
            all_sets = detect_sets_full(pool_cards)

            declared_types = []
            for idx, count in enumerate(self.declared_sets[player]):
                declared_types += [self.SET_TYPE_BY_INDEX[idx]] * int(count)

            candidates = []
            for s in all_sets:
                t = s["type"]
                ok = (t in declared_types) or ("Mia" in declared_types
                                               and t in ("Mia_s", "Mia_c"))
                if not ok:
                    continue

                idxs = [pool_idxs[pool_cards.index(card)]
                        for card in s["cards"]]
                candidates.append(idxs)

            counts = np.zeros((32, 4), dtype=float)
            for cands in candidates:
                for c in cands:
                    counts[c, player] += 1

            for c in hidden:
                if np.any(np.isclose(self.card_ownership[c, :, agent], 1.0)):
                    continue

                prior = self.card_ownership[c, :, agent]
                likelihood = counts[c] + eps
                post = prior * likelihood
                post_sum = post.sum()
                if post_sum > 0:
                    self.card_ownership[c, :, agent] = post / post_sum
