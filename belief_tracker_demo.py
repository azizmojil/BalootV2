import numpy as np
from itertools import combinations

# Simplified constants for demonstration
SUITS = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
# Note: Baloot uses 7, 8, 9, 10, J, Q, K, A.
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
DECK = [(s, r) for s in SUITS for r in RANKS]

class BeliefTracker:
    """
    A probabilistic belief tracker for imperfect information card games.
    Maintains a 32x4 tensor representing the probability of each card 
    being in each player's hand.
    """
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.num_cards = len(DECK)
        
        # cards_ownership: (32 cards, 4 players)
        # Initialize with 0.33 probability for everyone except oneself (assuming we are player 0 for now)
        # In a real environment, you'd initialize based on who you are.
        self.cards_ownership = np.full((self.num_cards, self.num_players), 1/3, dtype=np.float32)
        
        # Keep track of possible Seras (3 consecutive cards of the same suit)
        # A Sera in Baloot is 3 cards. Let's define all possible 3-card sequences.
        self.all_possible_seras = self._generate_all_seras()
        
        # Dictionary mapping player_id to lists of possible Seras they might hold
        self.player_possible_seras = {i: [] for i in range(num_players)}

    def _generate_all_seras(self):
        """Generate all possible Sera combinations (3 consecutive cards same suit)."""
        seras = []
        for suit in SUITS:
            for i in range(len(RANKS) - 2):
                sera = [(suit, RANKS[i]), (suit, RANKS[i+1]), (suit, RANKS[i+2])]
                seras.append(sera)
        return seras

    def initialize_own_hand(self, player_id, hand):
        """Sets your own hand to 1.0 probability, and 0.0 for others."""
        for card in hand:
            card_idx = DECK.index(card)
            self.cards_ownership[card_idx, :] = 0.0
            self.cards_ownership[card_idx, player_id] = 1.0
            
            # Also zero out this card for everyone else.
            for p in range(self.num_players):
                if p != player_id:
                    self.cards_ownership[card_idx, p] = 0.0

    def declare_hidden_sera(self, player_id):
        """
        Player declares a Sera but does not reveal the cards.
        We find all Seras that are still mathematically possible for this player,
        and boost the probability of those specific cards.
        """
        print(f"\n[Event] Player {player_id} declares a hidden Sera!")
        possible_seras_for_player = []
        
        for sera in self.all_possible_seras:
            is_possible = True
            for card in sera:
                card_idx = DECK.index(card)
                # If we know this player mathematically CANNOT have this card 
                # (e.g., probability is 0.0 because another player has it or it was played)
                if self.cards_ownership[card_idx, player_id] == 0.0:
                    is_possible = False
                    break
            
            if is_possible:
                possible_seras_for_player.append(sera)
                
        self.player_possible_seras[player_id] = possible_seras_for_player
        
        print(f"-> Found {len(possible_seras_for_player)} possible Seras for Player {player_id}.")
        
        # Update probabilities based on these possible Seras
        # We increase the probability of cards that appear in these possible Seras.
        self._update_probabilities_from_possible_seras(player_id)

    def _update_probabilities_from_possible_seras(self, player_id):
        """Boosts the probability of cards that are part of potential Seras."""
        possible_seras = self.player_possible_seras[player_id]
        if not possible_seras:
            return
            
        # Count how many possible Seras each card belongs to
        card_counts = {card: 0 for card in DECK}
        for sera in possible_seras:
            for card in sera:
                card_counts[card] += 1
                
        # Boost probability (simplified heuristic: the more possible Seras a card is in, 
        # the higher its probability)
        for card, count in card_counts.items():
            if count > 0:
                card_idx = DECK.index(card)
                # If probability isn't already 1.0, boost it to a high confidence (e.g., 0.8)
                if self.cards_ownership[card_idx, player_id] < 1.0:
                    self.cards_ownership[card_idx, player_id] = 0.8

    def play_card(self, player_id, played_card, trick_suit=None):
        """
        Updates belief state when a card is played.
        If player fails to follow suit, we mathematically eliminate that suit from their hand.
        """
        print(f"\n[Event] Player {player_id} plays {played_card}")
        
        # 1. This card is now played, no one holds it anymore (except technically public knowledge)
        card_idx = DECK.index(played_card)
        self.cards_ownership[card_idx, :] = 0.0
        
        # 2. Check for failure to follow suit (The big "tell")
        played_suit = played_card[0]
        if trick_suit is not None and played_suit != trick_suit:
            print(f"-> Player {player_id} failed to follow trick suit ({trick_suit})! They are void in {trick_suit}.")
            # Eliminate all cards of the trick suit for this player
            for rank in RANKS:
                void_card = (trick_suit, rank)
                void_idx = DECK.index(void_card)
                self.cards_ownership[void_idx, player_id] = 0.0
                
            # 3. RE-EVALUATE possible Seras!
            # If they are void in a suit, any possible Seras in that suit are now impossible.
            if self.player_possible_seras[player_id]:
                self._reevaluate_seras_after_void(player_id, trick_suit)

    def _reevaluate_seras_after_void(self, player_id, void_suit):
        """Removes impossible Seras and updates probabilities."""
        original_count = len(self.player_possible_seras[player_id])
        
        # Keep only Seras that do NOT contain the void suit
        self.player_possible_seras[player_id] = [
            sera for sera in self.player_possible_seras[player_id] 
            if sera[0][0] != void_suit # check suit of the first card in the Sera
        ]
        
        new_count = len(self.player_possible_seras[player_id])
        print(f"-> Filtering possible Seras... eliminated {original_count - new_count} Seras.")
        print(f"-> {new_count} possible Seras remaining for Player {player_id}.")
        
        # If we narrowed it down perfectly to 1 Sera, we know EXACTLY what cards they have!
        if new_count == 1:
            exact_sera = self.player_possible_seras[player_id][0]
            print(f"-> [BINGO!] We deduced the exact hidden Sera: {exact_sera}")
            for card in exact_sera:
                card_idx = DECK.index(card)
                self.cards_ownership[card_idx, player_id] = 1.0
                # Zero out for other players
                for p in range(self.num_players):
                    if p != player_id:
                        self.cards_ownership[card_idx, p] = 0.0

    def print_top_probabilities(self, player_id, top_n=5):
        """Helper to print the most likely cards for a player."""
        probs = self.cards_ownership[:, player_id]
        top_indices = np.argsort(probs)[::-1][:top_n]
        print(f"\nTop {top_n} likely cards for Player {player_id}:")
        for idx in top_indices:
            if probs[idx] > 0.0:
                print(f"  {DECK[idx]}: {probs[idx]:.2f}")

if __name__ == "__main__":
    # --- DEMONSTRATION ---
    tracker = BeliefTracker()
    
    # 1. We are Player 0. We hold some cards.
    my_hand = [('Spades', '7'), ('Spades', '8'), ('Spades', '9'), ('Spades', '10'), ('Hearts', 'A')]
    tracker.initialize_own_hand(player_id=0, hand=my_hand)
    
    # 2. Player 1 declares a hidden Sera!
    # Because we hold the 7, 8, 9, 10 of Spades, we know Player 1 CANNOT have a Spades Sera 
    # involving those cards. The tracker will filter those out automatically.
    tracker.declare_hidden_sera(player_id=1)
    
    # Let's see what the tracker thinks Player 1 might have
    tracker.print_top_probabilities(player_id=1, top_n=5)
    
    # 3. Later in the game, Hearts are led. Player 1 plays a Diamond.
    # Player 1 failed to follow suit! They have no Hearts.
    tracker.play_card(player_id=1, played_card=('Diamonds', '7'), trick_suit='Hearts')
    
    # Let's see how the probabilities updated
    tracker.print_top_probabilities(player_id=1, top_n=5)
