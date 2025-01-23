from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict, Tuple, MultiDiscrete, MultiBinary, OneOf
import random
import functools
from copy import copy
import numpy as np
from classes.game import Game
import itertools
from pettingzoo.utils import agent_selector, wrappers

from typing import Any,  Generic, Iterable, Iterator, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")
ActionMask = TypeError("ActionMask")

class CoupEnv(AECEnv):
    metadata = {
        "name": "coup_env_v0",
    }

    def __init__(self, n_players: int) -> None:
        """
        Defines the following attributes
        
        A Game of coup object with N_players 
        
        Coins and cards will be dealt
        """
        super().__init__() # grab aecenv 
        
        ############################################################################
        ###### Vals to help define obs and action space ##############################
        ############################################################################
        
        self.n_players = n_players
        self.game = Game(self.n_players)
        deck_size = len(self.game.deck.deck)
        self.agents = [p.name for p in self.game.players] # integer
        
        ############################################################################
        ###### Mapping integers to names of things ##############################
        ############################################################################
        self._card_names = ['assassin','ambassador', 'duke', 'contessa', 'captain']
        self._card_names_ints = [0, 1, 2, 3, 4]
        self._card_name_map = dict(zip(self._card_names, self._card_names_ints))
        
        # action types
        self.action_type_map = {0:'normal_action',
                                1:'challenge_action',
                                2:'block_action'}
        
        
        
        ############################################################################
        ###### Observation spaces for each agent ##############################
        ############################################################################
        # init a gem instance to help define observation space

        self.observation_space_dict = Dict({
            'agent_cards': MultiDiscrete([5, 5]), # pairs of cards here
            "agent_money": Discrete(n=14), # can have 0-13 coins
            "agent_claims": MultiBinary([5]),
            "agent_deck_knowledge": MultiDiscrete([6] * deck_size, start=[-1]*deck_size), #Order o deck, -1 indicates we do not know what the card is. # TODO figure out how to limit this to only correct observations like (-1,-1,2,4
            # Maybe will need to throw out bad obs?
            "others_claims": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [MultiBinary([5]) for _ in range(self.n_players)]))), # Player_int: Text # dict of text spaces for what others are claiming,
            
            "others_n_cards": Dict(dict(zip([f'player{n}' for n in range(self.n_players)], 
                                      [Discrete(2) for _ in range(self.n_players)]))),
            
            "others_money": Dict(dict(zip([f'player{n}' for n in range(self.n_players)],
                                          [Discrete(14) for _ in range(self.n_players)]))), # Player_int: Discrete
            
            "revealed": Dict(dict(zip(self._card_names_ints, 
                                      [Discrete(3) for _ in range(5)] ))), # Card name, number revealed)
                
            "action_player": Discrete(self.n_players), 
            "target_player": Discrete(self.n_players), # may not need this
            "action_type": Discrete(3), # for action masking
        })
        
        self.observation_spaces = dict(zip([agent for agent in self.agents],
                                           [self.observation_space_dict for _ in self.agents]))
        
        
        ############################################################################
        ###### Action spaces for each agent ##############################
        ############################################################################
        
        # action space stays sime. Masking happens later
        self._base_actions = list(set(self.game.actions.ALLOWED_ACTIONS + list(self.game.actions.CHALLENGABLE_ACTIONS)))
        self._base_actions.remove("assassinate")
        self._base_actions.remove("steal")
        self._base_actions.remove("coup")
        self._base_actions.remove("block_assassinate")
        self._base_actions.remove("block_foreign_aid")
        self._base_actions.remove("block_steal_amb")
        self._base_actions.remove("block_steal_cap")
        self._steal_actions = [f"steal_from_{agent}" for agent in self.agents]
        self._base_actions = self._base_actions + self._steal_actions
        
        self._assassinate_actions = [f"assassinate_{agent}" for agent in self.agents]
        self._coup_actions = [f"coup_{agent}" for agent in self.agents]
        self._challenge_action = ["challenge"] 
        self._pass_action = ['pass']
        
        self._actions = self._base_actions + self._assassinate_actions + self._coup_actions + self._challenge_action + self._pass_action
        self._actions.sort()
        self._action_space_map = dict(zip([action for action in self._actions],
                                          [n for n in range(len(self._actions))]
                                          ))
                
        self.action_space_dict = Discrete(len(self._actions), start = 0) # first is action, second is target player, -1= No target
        self.action_spaces = dict(zip([agent for agent in self.agents],
                                           [self.action_space_dict for _ in self.agents]))
        
        
    def _get_obs(self, agent: AgentID) -> ObsType:
        """
        Returns the observation an agent currently can make.
        """
                #################################
        ##### THINGS ONLY THE AGENT KNOWS
        #################################
        agent_instance = self.game.players[agent] # grab the correct instance of the agent
        agent_cards = agent_instance.cards # need to 
        agent_money = agent_instance.coins
        agent_claims = agent_instance.claimed_cards
        agent_deck_knowledge = agent_instance.knowledge.deck_knowledge 
        others_claims = agent_instance.knowledge.other_player_claims # turn this into cards instead of actions
        others_n_cards = agent_instance.knowledge.other_player_n_cards # turn this into cards instead of actions
        others_money = agent_instance.knowledge.other_player_n_coins
        
        #################################
        ##### THINGS EVERYONE KNOWS
        #################################
        next_action_type = self.game.next_action_type
        revealed = self.game.revealed_cards
        turn_order = self.game.turn.turn_order
        others_claims = agent_instance.knowledge.other_player_claims # turn this into cards instead of actions
        others_n_cards = agent_instance.knowledge.other_player_n_cards # turn this into cards instead of actions
        others_money = agent_instance.knowledge.other_player_n_coins
        
        current_base_player= self.game.turn.current_base_player.name
        if self.game.turn.current_base_action_instance:
            current_claimed_card = self.game.turn.current_base_action_instance.card
        else:
            current_claimed_card = None
        base_action_target_player = -1 # no target # todo fix WHEN USING TARGET
        
        observation = {
            "next_action_type": next_action_type, #base_action, challenge_action, or block_action
            "agent_cards": agent_cards,
            "agent_money": agent_money,
            "agent_claims": agent_claims,
            "agent_deck_knowledge": agent_deck_knowledge,
            "others_claims": others_claims,
            "others_n_cards": others_n_cards,
            "others_money": others_money,
            "revealed": revealed,
            "turn_order": turn_order,
            "current_base_player": current_base_player,
            "current_claimed_card": current_claimed_card,
            "base_action_target_player": base_action_target_player,
            }
        
        return observation
        
    
    def observe(self, agent: AgentID) -> dict[ObsType, ActionMask]:
        """
        get the observation and action mask for an agent
        
        env.last() calls this
        """
        # observation
        observation = self._get_obs(agent)
                
        # Compute action mask
        action_mask = self._compute_action_mask(agent)
        
        return {'observation':observation, 'action_mask':action_mask}
    
    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]
    
    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        
        return self.observation_spaces[agent]
        
    
    def _compute_action_mask(self, agent:AgentID) -> ActionMask: # TODO    
        """Computes an action mask for the agent
        
        Will only be called if agent needs to make decision 
        
        

        Args:
            agent (AgentID): _description_

        Returns:
            ActionMask: _description_
        """
        
        # init action mask of 0s to represent valid actions
        a_mask = np.array([0] * len(self._actions), dtype=np.int8)

        next_action_type = self.game.next_action_type # what type of action is able to be selected here
        agent_instance = self.game.players[agent]
        agent_money = agent_instance.coins
        current_base_action_instance = self.game.turn.current_base_action_instance
        

        if next_action_type == 'base_action':
            if agent_money>10:
                lo_valid_actions = self._assassinate_actions + self._coup_actions
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            
            elif agent_money >=7:
                lo_valid_actions = self._assassinate_actions + self._coup_actions + self._base_actions
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask

            elif agent_money >=3:
                lo_valid_actions = self._assassinate_actions + self._base_actions
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            
            else:
                lo_valid_actions = self._base_actions
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
    
        elif next_action_type == "challenge_action":
            lo_valid_actions = self._challenge_action + self._pass_action
            lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
            a_mask[lo_valid_indexes] = 1
            return a_mask
        
        
        elif next_action_type == "block_action":
            pass
            if current_base_action_instance.name == 'foreign_aid':
                lo_valid_actions = self._block_fa + self._pass_action
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]

            elif current_base_action_instance.name == "assassinate":
                lo_valid_actions = self._block_ass + self._pass_action
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                return a_mask
            elif current_base_action_instance.name == "steal":
                lo_valid_actions = self._block_steal + self._pass_action
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                return a_mask
            
    def _is_action_valid(self, action):
        real_action = self._compute_action_mask()[0][action[0]] == 1
    
        real_target = self._compute_action_mask()[1][action[1]] == 1
        return real_action * real_target
    
    def sample_valid_action(self):
        mask = self._compute_action_mask()
        # Sample valid actions for each instance in the mask
        valid_actions = []
        for row in mask:
            # Get indices of valid actions for this row
            valid_indices = np.flatnonzero(row)
            if valid_indices.size == 0:
                raise ValueError("No valid actions available for some instance in the mask.")
            # Sample a valid action from the valid indices
            sampled_action = np.random.choice(valid_indices)
            valid_actions.append(sampled_action)
        return np.array(valid_actions)
    
    def reset(self, seed=None, options=None) -> None:
        """
        Resets the game to a fresh game with freshly dealt cards
        
        Reset needs to re-initialize the following attributes
        - game
            -agents
            -agent_selection generator
        - observations
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """   
        ############################################################################
        ###### self.game initialiation -- very important ##############################
        ############################################################################
        self.game = Game(n_players=self.n_players)
                
        ############################################################################
        ###### self.agents initialization -- very important ##############################
        ############################################################################
        self.possible_agents = ['player']
        self.agents = [p.name for p in self.game.players] # integer
        
        ############################################################################
        ###### basic RL outputs for each agent ##############################
        ############################################################################
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}      
        
        #### AGENT SELECTOR ALLOWS FOR STEPPING THROUGH AGENTS
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
 
    def state(self) -> dict[AgentID,ObsType]:
        return self.observations
        
        
    
    def step(self, action: ActionType) -> None: # TODO
        """Takes an action by the current agent
        
        Updates self.observations depending on a lot of things:
        
        if action type is base action
            agent's claims change
            action type changes from base action to challenge action
        if action type is challenge action and baseplayer not current agent:
            if agent chose to challenge, update game state
            otherwise, pass
        if action type is challenge and base player is current agent:
            agent does their base action on game
            agent changes action type to base action
        
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        print(f"Current Agent {agent}")
        
        
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        
        
        return 
        
        
        action = self.sample_valid_action()
        if self.game.turn.action_type == 'base_action':
            action = [12,-1]
            
        print(f"Selected Action {action}")
        
        # check if action is valid
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action {action} for the current state.\n {self.game.turn.action_type}\n{self.game.turn.current_base_player}")

        action_map = self._action_space_map
        #first make the step through the action
        
        self.last = self.observe().copy()

        self.game.step(action, action_map)

        
        # then get the observation of the new environment
        observation = self.observe()
        
        # then get info on if we've terminated
        terminated = self.game.win
        # truncated if agent is dead
        truncated = self.game.lost 
        
        # then get the reward for stepping/terminating
        reward = self.get_reward(terminated, truncated)
        # print(f"Reward {reward}, terminated {terminated}, truncated {truncated}")
        
        # then get any additional info and truncation flag
        return observation, reward, terminated, truncated, {}
        
    def get_reward(self, terminated, truncated):        
        last = self.last
        obs = self.observe()
        reward = 0    
    
        # +1 for every money we gained
        money_reward = obs["agentmoney"] - last["agentmoney"]
    
        # if another bot has 1 less life, this is +5    
        diff_lives = {bot: last['others_n_cards'][bot] - obs['others_n_cards'][bot] for bot in last['others_n_cards'].keys()}
        kills = sum(diff_lives.values())
        kills_reward = abs(kills*10)
    
        # if we've terminated and we have won this is +50
        if terminated: # agent won
            win_reward = 50
        elif truncated: #agent died
            win_reward = -50
        else:
            win_reward = 0
        
        # if we've died this is -50
        reward += money_reward + kills_reward + win_reward
        
        return reward


    

    def render(self):
        pass



