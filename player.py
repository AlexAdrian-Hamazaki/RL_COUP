
import numpy as np

class Player():
    def __init__(self, name:str):
        self._name = name
        self._known_cards = [] # cards we know and where they are (deck or face-up by another player)
        self._known_deck_order = [] # cards we know are in deck and where they are
        self._claimed_cards = set() # cards we claim
        self._others_claimed_cards = None # cards claimed by others # this is handeled by game object. kinda jank
        self._coins = 0
        self._cards = [] # current cards
        
    def __repr__(self):
        return f"""
                Player {self._name},
                Cards {self.cards}
                Coins {str(self.coins)}
                Claimed {str(self.claimed_cards)}
                """
        
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value: str):
        self._name = value
    
    @property
    def known_cards(self):
        return self._known_cards
    @known_cards.setter
    def known_cards(self, value: list):
        self._known_cards = value
    
    @property
    def known_deck_order(self):
        return self._known_deck_order
    @known_deck_order.setter
    def known_deck_order(self, value: list):
        self._known_deck_order = value
    
    @property
    def claimed_cards(self):
        return self._claimed_cards
    @claimed_cards.setter
    def claimed_cards(self, value: list):
        self._claimed_cards = value
    
    @property
    def others_claimed_cards(self):
        return self._others_claimed_cards
    @others_claimed_cards.setter
    def others_claimed_cards(self, value): #TODO Handle player-card relationships. Maybe make a dict here for player:claimed cards?
        self._others_claimed_cards = value
    
    @property
    def coins(self):
        return self._coins
    @coins.setter
    def coins(self, value: int):
        if isinstance(value, int) and value >= 0:
            self._coins = value
        else:
            raise ValueError("Coins must be a non-negative integer.")
        
    @property
    def cards(self):
        return self._cards
    @cards.setter
    def cards(self, value: list):
        self._cards = value
        
    
    def draw_card(self, game):
        self._cards.append(game.deck.deck[0])
        game.deck.remove_top_card()
        
    def take_coin(self, game):
        try:
            if game.bank.n <= 0:
                raise ValueError
            else:
                print(f"\t{self.name} takes 1 coin from bank")
                self.coins+=1
                game.bank.remove()
        except ValueError as e:
            print("No coins in bank, invalid option")
            return 0 #action failed flag.
        
    def discard_coin(self, game, n:int):
        try:
            if self.coins <= n:
                raise ValueError
            else:
                self.coins+-n
                [game.bank.add() for _ in range(n)]
        except ValueError as e:
            print(f"\t{self.name} cannot give enough coins to perform action")
            return 0 #action failed flag.
    
    
    def add_claimed_card(self, action):
        self.claimed_cards.add(action)
    def remove_claimed_card(self, action):
        self.claimed_cards.add(action)
    def update_others_curr_ac(self, action, player): 
        # updates the "other players" knowledge of what the current player
        # is claiming
        # self is the other player in this case
        dic_other_claimed_cards = self.others_claimed_cards # keys are playernames and values is a set of their claimed cards
        print(dic_other_claimed_cards)
        print(player.name)
        current_players_claimed_cards = dic_other_claimed_cards[player.name]
        current_players_claimed_cards.append(action)
        # update knowledge of players cards
        dic_other_claimed_cards[player.name] = current_players_claimed_cards
        self.others_claimed_cards = dic_other_claimed_cards
        print(self.others_claimed_cards)
        assert False
        
    def put_card_on_bottom(self, game):
        pass
    def lose_life(self): 
        pass
    def take_action(self, game, action):
        pass
    def challenge(self, game):
        pass






