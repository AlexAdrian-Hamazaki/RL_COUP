
import numpy as np
from card import Card
class Player():
    def __init__(self, name:str):
        self._name = name
        self._known_cards = [] # cards we know and where they are (deck or face-up by another player)
        self._known_deck_order = [] # cards we know are in deck and where they are
        self._claimed_cards = set() # cards we claim
        self._others_claimed_actions = [] # cards claimed by others # this is handeled by game object. kinda jank
        self._coins = 0
        self._cards = [] # current cards
        self._status = 'alive'
        
    def __repr__(self):
        return f"""Player {str(self.name)}, Cards {str(len(self.cards))}, Coins {str(self.coins)}"""
        
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
    def others_claimed_actions(self):
        return self._others_claimed_actions
    @others_claimed_actions.setter
    def others_claimed_actions(self, value): #TODO Handle player-card relationships. Maybe make a dict here for player:claimed cards?
        self._others_claimed_actions = value
    
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
    def cards(self, cards: list):
        self._cards = cards
        
    @property
    def status(self):
        return self._status
    @status.setter
    def status(self, status):
        self._status = status
        
    
    def draw_card(self, game):
        card = game.deck.deck[0]
        card.status = 'hand'
        self.cards.append(card)
        game.deck.remove_top_card()
        
        
    def take_coins(self, game, n):
        try:
            if n > game.bank.n :
                raise ValueError(f"\tCannot take {n} couns from bank")
            else:
                self.coins+=n
                game.bank.remove(n)
        except ValueError as e:
            return 0 #action failed flag. #TODO
        
    def discard_coin(self, game, n:int):
        try:
            if self.coins <= n:
                raise ValueError(f"\tNot enough coins to discard {n}")
            else:
                self.coins-=n
                # print(f"\tGave {n} couns to bank")
                game.bank.add(n)
        except ValueError as e:
            return 0 #action failed flag.
    
    
    def add_claimed_action(self, action):
        self.claimed_cards.add(action)
    def remove_claimed_action(self, action):
        self.claimed_cards.add(action)

        
        
    def update_others_curr_ac(self, action, player): 
        # updates the "other players" knowledge of what the current player
        # is claiming
        # self is the other player in this case
        dic_other_claimed_cards = self.others_claimed_actions # keys are playernames and values is a set of their claimed cards
        current_players_claimed_cards = dic_other_claimed_cards[player.name]
        current_players_claimed_cards.append(action)
        # update knowledge of players cards
        dic_other_claimed_cards[player.name] = current_players_claimed_cards
        self.others_claimed_actions = dic_other_claimed_cards
        
    def put_card_on_bottom(self, card, game):
        self.cards.remove(card)
        game.deck.add_to_bottom(card)
        
                
    
    def lose_life(self, game): 
        """Player loses a life
        Handles the following:
        removal of lost life from claimed cards
        removal of card from players cards
        addition of dead card into known pool of revealed cards
        checks to see if player is dead and handles turn order changes

        Args:
            game (_type_): _description_

        Returns:
            _type_: _description_
        """
        player_cards = self.cards
        lo_names = set([card.name for card in player_cards])
        card_name = input(f"\tPlayer {self.name} choose to lose one of {lo_names}").strip().lower()
        
        if Card.SHORT_KEYS.get(card_name): # if shorthand name was used, get full name
            card_name = Card.SHORT_KEYS.get(card_name)
        
        updated_list = []
        found = False
        for card in player_cards:
            if card.name.lower() == card_name and found==False:
                print(f"Player {self.name} reveals a {card_name}")
                found = True
                revealed_card = card
            else:
                updated_list.append(card)
        if found == False:
            print(f"\tInvalid card name selected, you dont have that card: {lo_names}")
            return self.lose_life(game)
                
        self.cards = updated_list
        revealed_card.state='revealed'
        [self.remove_claimed_action(ac) for ac in revealed_card.REAL_ACTIONS] # may fail here
        game.revealed_cards.append(revealed_card)
        self.check_death(game)
    
    def check_death(self, game):
        # if player is dead, update the turn order
        if len(self.cards)==0:
            print(f"Player {self.name} is out of influence")
            self.status = 'dead'
            game.update_order_after_death()




