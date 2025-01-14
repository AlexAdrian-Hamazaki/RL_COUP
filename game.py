import numpy as np
from actions import Actions
from card import Assassin, Captain, Contessa, Ambassador, Duke
from player import Player

class Game:
    
    POSSIBLE_ACTIONS = {} # set of possible actions to take
    
    def __init__(self, n_players):
        self._n_players = n_players
        self._players = [Player(name=n) for n in range(n_players)]
        self._deck = Deck()    
        self._deck._shuffle() # shuffle deck
        self._bank = CoinBank()
        self._curr_turn = None
        
        # TODO ADD INITIALIZATION OF GAME FUNCTIONS HERE INSTEAD OF IN SCRIPT
    
    def __repr__(self):
        return f"""
              Game of COUP
              Number of Players = {self._n_players}
              Current Deck Order (top to bottom) = {self._deck}
              Current Coins in Bank = {self._bank}
              Current player's tern = {str(self._curr_turn)}
              """
    
    @property
    def deck(self):
        return self._deck
        
    @property
    def bank(self):
        return self._bank

    
    
    def _setup_deal(self): # initialize dealing of cards to players
        print("Dealing Setup Cards")
        n=0
        while n < 2 * self._n_players:
            for player in self._players:
                player.draw_card(self)
                n+=1
                
    
    def _setup_give_coins(self): # initialize giving of cards to players
        print("Giving Coins to players")
        n = 0
        while n < 2 * self._n_players:
            for player in self._players:
                player.take_coin(self)
                n+=1
            
        
    
        
    
        
    
    

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
        
    def __repr__(self):
        return f"Deck State : {self._deck.tolist()}"
        
        
    def remove_top_card(self):# top card gets drawn be player
        self._deck = self._deck[1:]
    def _add_to_bottom(self):# player adds card to bottom
        pass
    def _shuffle(self):
        np.random.shuffle(self._deck)  # Shuffle in place, no need to assign it back
        print("Deck Shuffled")

class CoinBank():
    def __init__(self):
        self._n = 50
    def remove(self):
        self._n -=1
    def add(self):
        self._n +=1        
        
    def __repr__(self):
        return str(self._n)
    
    @property
    def n(self):
        return self._n
    


        
        
