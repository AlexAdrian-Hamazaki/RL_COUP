import numpy as np
from actions import Actions, Income, Foreign_Aid, Coup, Tax, Assassinate, Steal, Exchange, B_Assassinate, B_Foreign_Aid, B_Steal
from card import Assassin, Captain, Contessa, Ambassador, Duke
from challenge import Challenge
from player import Player
from deck import Deck
from bank import CoinBank
from turn import Turn
import textwrap
class Game:
    
    
    def __init__(self, n_players):
        self._n_players = n_players
        self._action_n =0
        self._players = [Player(n) for n in range(n_players)]
        self._deck = Deck()    
        self._deck.shuffle() # shuffle deck
        self._revealed_cards = []
        self._bank = CoinBank()
        self._turn = Turn()
        
        self.on = True # true if game is live, False if game is over
        
        # setup game
        self._setup_deal()    
        self._setup_give_coins()
        [self._setup_other_claimed_action_dicts(player) for player in self.players]    
        # setup actions, which will hold all possible actions
        self._actions = Actions()
        self._action_map =  {"income":Income,
                            "foreign_aid":Foreign_Aid,
                            "coup":Coup,
                            "tax":Tax,
                            "assassinate":Assassinate,
                            "steal":Steal,
                            "exchange":Exchange}
        self._block_action_map =  {"Block_Foreign_Aid":B_Foreign_Aid,
                                   "Block_Assassinate":B_Assassinate,
                                   "Block_Steal":B_Steal}
        
    
        print(f"""Initialized game with {len(self.players)} players""")
        
        
    def __repr__(self):
        result = f"""
{"*"*40}
GAME STATUS
{"-"*40}
Turn: {self._action_n}
Revealed cards: {str(self.revealed_cards)}
Current Coins in Bank = {self._bank}
{"*"*40}"""
        
        # Dedent the entire block to avoid unnecessary leading spaces
        return result

        
    @property
    def deck(self):
        return self._deck
    @property
    def bank(self):
        return self._bank
    @property
    def players(self):
        return self._players
    @players.setter
    def players(self, players):
        self._players = players
    @property
    def n_players(self):
        return self._n_players
    @n_players.setter
    def n_players(self, n_players):
        self._n_players = n_players

    @property
    def turn(self):
        return self._turn
    @turn.setter
    def turn(self, turn):
        self._turn = turn

    @property
    def revealed_cards(self):
        return self._revealed_cards
  
    @property
    def actions(self):
        return self._actions
    @property
    def action_map(self):
        return self._action_map
    
    
    
    def add_to_revealed_cards(self, card):
        self._revealed_cards = self._revealed_cards.append(card)
    
    
    def _setup_deal(self): # initialize dealing of cards to players
        print("Dealing Setup Cards")
        n=0
        while n < 2 * self._n_players:
            for player in self.players:
                player.draw_card(self)
                n+=1        
    
    def _setup_give_coins(self): # initialize giving of cards to players
        print("Giving Coins to players")
        n = 0
        while n < self._n_players:
            for player in self._players:
                player.take_coins(self, 2)
                n+=1
                
    def _setup_other_claimed_action_dicts(self, player):
        player_int = player.name
        players = self.players
        other_players = players[player_int+1:] + players[:player_int]
        other_player_names = [ply.name for ply in other_players]
        player.others_claimed_actions =  {name : [] for name in other_player_names} # init dict for each other player to hold their claimed cards
    
    def get_player_by_name(self, player_name):
        # returns the actual player object given a string name
        for player in self.players:
            if str(player_name) == str(player.name):
                return player
        raise ValueError(f"Player name {player_name} not found")
    
    def next_turn(self):
        self.turn.next_turn(self)
        if self.n_players==1:
            self.on=False
            
    def update_claims(self, player, other_players):
        """
        After some action is made by player
        this function updates the knowledge of all other players that the current player made the action 
        """        
        # update the other player's knowledge if this players claimed actions
        [other_player.update_other_p_c_action(player) for other_player in other_players]


    def update_deck_knowledge(self):
        # updates deck knowledge
        pass

    def update_revealed_knowledge_for_players(self):
        for player in self.players:
            player.knowledge.revealed_knowledge = self.revealed_cards

    def update_order_after_death(self): # should go into game object
        players = self.players
        # first we find the index of the player that is dead
        i_dead = None
        for i, player in enumerate(players):
            if player.status == 'dead':
                i_dead = i
                break
        if i_dead==None:
            raise ValueError("i_dead should never be None when this function is called")
        
        self.n_players-=1
        self.players = players[i_dead+1:] + players[:i_dead]
        
        self.turn.update_after_death(i_dead)
    
            

    


        
        
