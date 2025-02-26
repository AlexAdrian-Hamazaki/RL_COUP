import numpy as np
from .actions import Actions, Income, Foreign_Aid, Coup, Tax, Assassinate, Steal, Exchange, B_Assassinate, B_Foreign_Aid, B_Steal_Ambassador, B_Steal_Captain
from .card import Assassin, Captain, Contessa, Ambassador, Duke
from .challenge import Challenge
from .player import Player, Bot
from .deck import Deck
from .bank import CoinBank
from .turn import Turn
import textwrap

class Game:
    
    def __init__(self, n_players):
        self._n_players = n_players
        self._players = [Player(n) for n in range(n_players)]
        self._deck = Deck()    
        self._deck.shuffle() # shuffle deck
        
        self._revealed_cards = {'assassin':0, 'captain':0, 'duke':0, 'contessa':0, 'ambassador':0}
        self._n_cards = {n:2 for n in range(n_players)}
        self._n_coins = {n:2 for n in range(n_players)}

        self._bank = CoinBank()
        self._turn = Turn(self.players, self)
        
        self._win = False # true if agent won
        self._lost = False # True if agent is dead
        np.random.seed(None)
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
                            "exchange":Exchange,
                            'pass':Actions}
        
        # init deck knowledge
        [player.knowledge.init_deck_knowledge(self.deck) for player in self.players]
        self.update_knowledge()

            
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
    def agent(self):
        return self._agent
    @agent.setter
    def agent(self, agent):
        self._agent = agent

    @property
    def bots(self):
        return self._bots
    @bots.setter
    def bots(self, bots):
        self._bots = bots
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
    @revealed_cards.setter
    def revealed_cards(self, revealed_cards):
        self._revealed_cards = revealed_cards
        
    @property
    def n_cards(self):
        return self._n_cards
    @n_cards.setter
    def n_cards(self, n_cards):
        self._n_cards = n_cards
  
    @property
    def n_coins(self):
        return self._n_coins
    @n_coins.setter
    def n_coins(self, n_coins):
        self._n_coins = n_coins
        
    @property
    def actions(self):
        return self._actions
    @property
    def action_map(self):
        return self._action_map
    
    @property
    def win(self):
        return self._win
    @win.setter
    def win(self, win):
        self._win = win
    @property
    def lost(self):
        return self._lost
    @lost.setter
    def lost(self, lost):
        self._lost = lost
    
    def add_to_revealed_cards(self, card):
        self.revealed_cards[card.name.lower()] +=1 
    
    def _setup_deal(self): # initialize dealing of cards to players
        #print("Dealing Setup Cards")
        n=0
        while n < 2 * self._n_players:
            for player in self.players:
                player.draw_card(self)
                n+=1        
    
    def _setup_give_coins(self): # initialize giving of cards to players
        #print("Giving Coins to players")
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
    
    # def next_turn(self):
    #     self.turn.next_turn(self)
    #     if self.n_players==1:
    #         #print(f"Player {self.players} wins")
    #         self.on=False
            
    def assess_game_win(self):
        """
        Assess if the agent has won the game or lost it
        """
        if self.agent.cards == 0:
            self.lost = True
        elif len(self.players) == 1:
            self.win = True
            
            
    def update_knowledge(self):
        """
        After some action is made by player, be it challenge, claim, or block
        
        This function updates all player's knowledge of the following
        
        Revealed cards
        all other players claimed cards
        all other players n coins
        all other players n cards
        """
        # self.update_revealed_knowledge_for_players()

        self.update_all_players_claimed_cards()
        self.update_all_players_n_coins()
        self.update_all_players_n_cards()

    def update_revealed_knowledge_for_players(self):
        lo_names = [name for name in list(self.revealed_cards.keys())]
        for player in self.players:
            player.knowledge.revealed_knowledge = lo_names
            
            
    def update_all_players_claimed_cards(self):
        self.claimed_cards = {player.name :player.claimed_cards for player in self.players}
        # print(self.claimed_cards)
                
    def update_all_players_n_coins(self):
        self.n_coins = {player.name: player.coins for player in self.players}

    def update_all_players_n_cards(self):
        self.n_cards = {player.name: len(player.cards) for player in self.players}
            


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
        
    # def step(self, action, action_map, agent):
    #     """
    #     Leverage the turn class to            
    #     make a step to the next game state
    #     """
    #     self.turn.step(action, action_map, self)
    #     self.assess_game_win()
        
    

    
            

    


        
        
