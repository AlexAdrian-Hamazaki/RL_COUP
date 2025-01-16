
import numpy as np
from card import Card
from player_knowledge import Knowledge

class Player():
    def __init__(self, name:str):
        self._name = name
        self._claimed_cards = set() # cards we claim
        self._knowledge= Knowledge() # cards claimed by others # this is handeled by game object. kinda jank
        self._coins = 10
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
    def claimed_cards(self):
        return self._claimed_cards
    @claimed_cards.setter
    def claimed_cards(self, value: list):
        self._claimed_cards = value
    
    @property
    def knowledge(self):
        return self._knowledge
    @knowledge.setter
    def knowledge(self, knowledge): 
        self._knowledge = knowledge
    
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
        # draw card
        self.cards.append(card)
        # remove card from top of deck
        game.deck.remove_top_card()
        # add to player knowledge
        self.knowledge.add_to_cards(card)
        
        
        
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
        
    def update_other_p_c_action(self, other_player): 
        self.knowledge.update_other_p_c_action(other_player)
        
    def put_card_on_bottom(self, card, game):
        self.cards.remove(card)
        self.knowledge.remove_from_cards(card)
        game.deck.add_to_bottom(card)
        
    def check_challenge(self, game): # TODO ENVIRONTMENT
        
        knowledge = self.knowledge ### TODO this knowledge needs to be passed to environment
        
        # game asks player if player wants to contest the proposed action of the current player
        contest = input(f"\t\tDoes {self.name} want to contest current action: (y/n)") #O
        if contest=="y":
            return True
        elif contest=="n":
            return False
        else:
            print("enter valid option (y/n)")
            return self.check_challenge(game)
        
                
    
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
        
        # Hand now becomes n-1 size
        self.cards = updated_list
        
        # add card to revealed card list in game
        revealed_card.state='revealed'
        game.revealed_cards.append(revealed_card)
        game.update_revealed_knowledge_for_players()
        
        # Remove the claimed action from this player's claimed actions
        [self.remove_claimed_action(ac) for ac in revealed_card.REAL_ACTIONS] #removes
        # Update each player's knowledge of this player's claimed actions
        game.update_claims(player = game.turn.current_player, other_players = game.turn.other_players)
        
        # Check to see if player is dead
        self.check_death(game)
    
    def check_death(self, game):
        # if player is dead, update the turn order
        if len(self.cards)==0:
            print(f"~~~~~~~~~~Player {self.name} is out of influence~~~~~~~~~~")
            self.status = 'dead'
            game.update_order_after_death()




