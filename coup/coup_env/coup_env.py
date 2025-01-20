from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict
import random
import functools
from copy import copy

from classes.game import Game

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
        self.game = None
        
        self.observation_space = Dict({
            'mycards': Text(max_length=2, min_length=1),
            "mymoney": Discrete(),
            "deck_knowledge": Sequence(Text), # sequence of text of what our card knowledge is
            "others_claims": Dict() # Player_int: Text # dict of text spaces for what others are claiming,
            "others_money": Dict(), # Player_int: Discrete
            "revealed": Text(min_length=0),
            "turn_order": Discrete()
            "action_player": Discrete()
            "target_player": Discrete() # may not need this
            "action_type": Discrete() # for action masking
            
        })
        
        
        self.action_space = gym.spaces.Discrete({
            
        })
        
        
        

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
