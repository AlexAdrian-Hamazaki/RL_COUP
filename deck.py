import numpy as np                
from card import *
class Deck():
    def __init__(self):
        dukes = [Duke() for _ in range(3)]
        contessas = [Contessa() for _ in range(3)]
        assassins = [Assassin() for _ in range(3)]
        captains = [Captain() for _ in range(3)]
        ambos = [Ambassador() for _ in range(3)]
        
        self._deck = np.array(dukes + contessas + assassins + captains + ambos)
    
    @property
    def deck(self):
        return self._deck
    @deck.setter
    def deck(self, deck):
        self._deck = deck
        
    def __repr__(self):
        return f"Deck State : {self._deck.tolist()}"
        
    def remove_top_card(self):# top card gets drawn be player
        self.deck = self.deck[1:]
    def add_to_bottom(self, card):# player adds card to bottom
        self.deck = np.append(self.deck, card)
    def shuffle(self):
        np.random.shuffle(self._deck)  # Shuffle in place, no need to assign it back
        print("Deck Shuffled")