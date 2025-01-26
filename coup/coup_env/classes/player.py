
import numpy as np
from .card import Card
from .player_knowledge import Knowledge

class Player():
    type = 'agent'
    def __init__(self, name:str):
        self._name = name
        self._claimed_cards = set() # cards we claim
        self._knowledge= Knowledge() # cards claimed by others # this is handeled by game object. kinda jank
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
        

    def draw_card(self, game):## UPDATE KNOWLEDGE OF PLAYER
        card = game.deck.deck[0]
        card.status = 'hand'
        # draw card
        self.cards.append(card)
        # remove top unknown card from deck knowledge
        self.knowledge.remove_top_card_from_deck_knowledge()
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
        
    def discard_coins(self, game, n:int):
        try:
            if self.coins <= n:
                raise ValueError(f"\tNot enough coins to discard {n}")
            else:
                self.coins-=n
                # print(f"\tGave {n} couns to bank")
                game.bank.add(n)
        except ValueError as e:
            return 0 #action failed flag.
    
    
    def add_claimed_card(self, card):
        self.claimed_cards.add(card)
    def remove_claimed_card(self, card):
        self.claimed_cards.add(card)
        
    def update_other_p_c_card(self, other_player): 
        self.knowledge.update_other_p_c_card(other_player)
        
    def update_other_p_n_cards(self, other_player): 
        self.knowledge.update_other_p_n_cards(other_player)
        
    def update_other_p_n_coins(self, other_player): 
        self.knowledge.update_other_p_n_coins(other_player)

    
        
    def put_card_on_bottom(self, card, game): 
        self.cards.remove(card)
        self.knowledge.remove_from_cards(card)
        self.knowledge.add_to_deck_knowledge(card)
        game.deck.add_to_bottom(card)
        # print(f"\t\t\tPlayer {self.name} put a card on bottom of deck")

        
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
        
    def check_block(self, block): ### UPDATE KNOWLEDGE OF PLAYER
        
        knowledge = self.knowledge ### TODO this knowledge needs to be passed to environment
        current_action = block.turn.current_action
        
        blockable_cards = block.BLOCKABLE_CARDS[current_action.name]
        
        # game asks player if player wants to contest the proposed action of the current player
        block_status = input(f"\t\tDoes {self.name} want to block the {current_action} action?: (y/n)") #O
        if block_status=="y":
            block.declared_blocker = input(f"With what card?: {blockable_cards}" )
            return True
        elif block_status=="n":
            return False
        else:
            print("enter valid option (y/n)")
            return self.check_block(block)
        
                
    
    def lose_life(self, game):  ## UPDATE KNOWLEDGE OF PLAYER
        """Player loses a life
        Args:
            game (_type_): _description_

        Returns:
            _type_: _description_
        """ 
        player_cards = self.cards
        
        if len(player_cards) ==0:
            # player has no more cards
            return
        
        print(f"Agent {self.name} loses 1 life")
        dead_int = np.random.choice(range(len(player_cards))) # TODO make this learnable

        dead_card = player_cards[dead_int]
        
        
        # Hand now becomes n-1 size
        self.cards = [val for i, val in enumerate(player_cards) if i != dead_int]

        # add card to revealed card list in game
        game.add_to_revealed_cards(dead_card)
        # update self knowledge
        self.knowledge.remove_from_cards(dead_card)
        
        # Remove the claimed action from this player's claimed actions
        [self.remove_claimed_card(ac) for ac in dead_card.REAL_ACTIONS] #removes
        
        # Check to see if player is dead
        self.check_death(game)
    
    def check_death(self, game):
        # if player is dead, update the turn order
        if len(self.cards)==0:
            print(f"~~~~~~~~~~Player {self.name} is out of influence~~~~~~~~~~")
            self.status = 'dead'
            




class Bot(Player):
    type = 'bot'
    
    def __init__(self, name):
        super().__init__(name)

    def choose_base_action(self, lo_actions):
        if not lo_actions:
            raise ValueError("The list of actions is empty. Cannot choose an action.")
        return np.random.choice(lo_actions)
    
    def choose_to_challenge(self):            
        # TODO go through the challenges of the opposing bots. They will choose to challenge at at 10% probability, 40% if they are targeted
        # and 100% if they know all 3 cards locations. 
        
        # for now 10% probability of challengiung
        return np.random.random() < 0
    
    def choose_to_block(self):            
        # go through the blocks of the opposing bots. They will choose to block if they have the card that can. Otherwise they will
        # bluff at a 50% probability. And this will change depending on the action and if its targeting you or not.
        # for now 50% prob of blocking
        return np.random.random() < 0.5