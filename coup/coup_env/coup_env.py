from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict, Tuple, MultiDiscrete, MultiBinary, OneOf
import random
import functools
from copy import copy
import numpy as np
from classes.game import Game
import itertools

class CoupEnv(gym.Env):
    metadata = {
        "name": "coup_env_v0",
    }

    def __init__(self, n_players):
        """
        Defines the following attributes
        
        A Game of coup object with N_players 
        
        Coins and cards will be dealt
        """
        self.n_players = n_players
        self.player_ints = range(n_players)
        self.game = Game(self.n_players)
        self.players = list(range(n_players))  # [0, 1, 2, ..., n_players-1]
        self.turn_order_permutations = [self.players[i:] + self.players[:i] for i in range(self.n_players)]
        self.turn_order_permutation_map = dict(zip([n for n in range(len(self.turn_order_permutations))], self.turn_order_permutations))
        
        
        
        
        # not sure if needed
        self._card_names = ['assassin','ambassador', 'duke', 'contessa', 'captain']
        self._card_names_ints = [0, 1, 2, 3, 4]
        self._card_name_map = dict(zip(self._card_names, self._card_names_ints))
        
        deck_size = len(self.game.deck.deck)
        
        
        # action types
        self.action_type_map = {0:'normal_action',
                                1:'block_action',
                                2:'challenge_action'}
        
        
        self.observation_space = Dict({
            'mycards': MultiDiscrete([5, 5]), # pairs of cards here
            "mymoney": Discrete(n=14), # can have 0-13 coins
            "myclaims": MultiBinary([5]),
            "deck_knowledge": MultiDiscrete([6] * deck_size, start=[-1]*deck_size), #Order o deck, -1 indicates we do not know what the card is. # TODO figure out how to limit this to only correct observations like (-1,-1,2,4
            # Maybe will need to throw out bad obs?
            "others_claims": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [MultiBinary([5]) for _ in range(self.n_players)]))), # Player_int: Text # dict of text spaces for what others are claiming,
            
            "others_n_cards": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [Discrete(2) for _ in range(self.n_players)]))),
            
            "others_money": Dict(dict(zip([f'player{n}' for n in range(self.n_players)],
                                          [Discrete(14) for _ in range(self.n_players)]))), # Player_int: Discrete
            
            "revealed": Dict(dict(zip(self._card_names_ints, 
                                      [Discrete(3) for _ in range(5)] ))), # Card name, number revealed)
                
            "turn_order": Discrete(len(self.turn_order_permutations)), # see turn_order_permutation map to look at what the turn order exactly is
            "action_player": Discrete(self.n_players), 
            "target_player": Discrete(self.n_players), # may not need this
            "action_type": Discrete(3), # for action masking
        })
        
        # this is the workhorse. this sample method.
        print(self.observation_space.sample())
        
        
        # self.action_space = gym.spaces.Discrete({
            
        # })
        
        
        

    def reset(self, seed=None, options=None):
        """Resets the game to a fresh game with freshly dealt cards

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        """
        
        self.game = Game(n_players=self.n_players)
        
        players = self.game.players
        
        observation = {player: player.knowledge for player in players}
        
        return observation
        
    
    def step(self, actions):
        """Takes an action by the current agent (specified by agent_selection)
        
        Actions are chosen via Q behavioir policy (or randomly depending on epsilon)
        
        Depending on the action, the following env parameters will need to be updated
        - player knowledge (obseration space)
            - turn order
            - n coins of players
            - revealed cards
            - player's cards
            - players life
            
        


        Args:
            actions (_type_): _description_
        """
        pass

    def render(self):
        pass
