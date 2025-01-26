from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict, Tuple, MultiDiscrete, MultiBinary, OneOf
import random
import functools
from copy import copy
import numpy as np
from .classes.game import Game
import itertools
from pettingzoo.utils import agent_selector, wrappers
from collections import Counter
from itertools import combinations_with_replacement

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
        self.action = None
        self.possible_agents = self.agents.copy()
        self._nstep =0
        
        ############################################################################
        ###### Mapping integers to names of things ##############################
        ############################################################################
        self._card_names = ['unknown', 'assassin','ambassador', 'duke', 'contessa', 'captain']
        self._card_names_ints = [-1, 0, 1, 2, 3, 4]
        self._card_name_map = dict(zip(self._card_names, self._card_names_ints))
        
        
        # Generate all possible combinations of 2 integers (with repetition allowed)
        card_combinations = list(combinations_with_replacement(self._card_names_ints[:], r=2))
        self._card_combination_map = {index: combo for index, combo in enumerate(card_combinations)}
            
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
        
        self._exe = ['exe']
        self._steal_actions = [f"steal_{agent}" for agent in self.agents]
        self._base_actions = self._base_actions + self._steal_actions
        
        self._assassinate_actions = [f"assassinate_{agent}" for agent in self.agents]
        self._coup_actions = [f"coup_{agent}" for agent in self.agents]
        self._challenge_action = ["challenge"] 
        self._pass_action = ['pass']
        
        self._actions = self._base_actions + self._assassinate_actions + self._coup_actions + self._challenge_action + self._pass_action + self._exe
        self._actions.sort()
        self._action_space_map = dict(zip([action for action in self._actions],
                                          [n for n in range(len(self._actions))]
                                          ))
                
        self.action_space_dict = Discrete(len(self._actions), start = 0) # first is action, second is target player, -1= No target
        self.action_spaces = dict(zip([agent for agent in self.agents],
                                           [self.action_space_dict for _ in self.agents]))
        
        ############################################################################
        ###### Observation spaces for each agent ##############################
        ############################################################################
        # init a gem instance to help define observation space

        self.observation_space_dict = Dict({
            'observation': 
                Dict({
                    'agent_cards': Discrete(len(self._card_combination_map.keys())), # pairs of cards here
                    "agent_deck_knowledge": MultiDiscrete([6] * deck_size, start=[-1]*deck_size), #Order o deck, -1 indicates we do not know what the card is. # TODO figure out how to limit this to only correct observations like (-1,-1,2,4
                    # # Maybe will need to throw out bad obs?
                    "claims": Dict(dict(zip([n for n in range(self.n_players)], 
                                            [MultiBinary([6]) for _ in range(self.n_players)]))), # Player_int: Text # dict of text spaces for what others are claiming,
                    
                    "n_cards": Dict(dict(zip([n for n in range(self.n_players)],
                                            [Discrete(3, start = 0) for _ in range(self.n_players)]))),
                    
                    # "money": Dict(dict(zip([n for n in range(self.n_players)],
                    #                             [Discrete(14) for _ in range(self.n_players)]))), # Player_int: Discrete
                    
                    # "revealed": Dict(dict(zip(self._card_names_ints[1:], 
                    #                         [Discrete(3, start = 0) for _ in range(len(self._card_names_ints[1:]))]))), # Card name, number revealed)
                        
                    # "current_base_player": Discrete(len(self.agents), start = 0), 
                    # "current_claimed_card": Discrete(len(self._actions), start = -1), # may not need this
                }),
            'action_mask':
                MultiBinary(len(self._actions))
        })

        self.observation_spaces = dict(zip([agent for agent in self.agents],
                                           [self.observation_space_dict for _ in self.agents]))
        
        
        
        
    def _get_obs(self, agent: AgentID) -> dict[ObsType, ActionMask]:
        """
        Returns the observation an agent currently can make.
        """
        #################################
        ##### THINGS ONLY THE AGENT KNOWS
        #################################
        agent_instance = self.game.players[agent] # grab the correct instance of the agent
        agent_cards = agent_instance.cards # need to 
        agent_deck_knowledge = agent_instance.knowledge.deck_knowledge 
        
        
        
        # agent cards
        agent_cards = [self._card_name_map[card.name.lower()] for card in agent_cards]
        agent_cards = sorted(agent_cards)
        if len(agent_cards)==1:
            agent_cards = [-1] + agent_cards
        if len(agent_cards)==0:
            agent_cards = [-1,-1]
        # Convert the list to a tuple and find the corresponding key
        card_combo = [k for k, v in self._card_combination_map.items() if v == tuple(agent_cards)][0]  # Get the key


        #################################
        ##### THINGS EVERYONE KNOWS
        #################################
        revealed = self.game.revealed_cards
        # Create a dictionary where the key is the integer and the value is its count
        revealed  = dict(Counter(revealed))
        turn_order = self.game.turn.turn_order
        current_base_player= self.game.turn.current_base_player.name
        current_claimed_card = self.game.turn.current_base_action_instance.card
        
        money = self.game.n_coins
        n_cards = self.game.n_cards
        claims = self.game.claimed_cards
        





        
        ###############################################################################
        ######################## OBSERVATION #######################################################
        ##############################################################################################################
        
    
        observation = {
            "agent_cards": card_combo ,
            "agent_deck_knowledge": np.array(agent_deck_knowledge),
            "claims": claims,
            "n_cards": n_cards,
            # "money": money,
            # "revealed": revealed,
            # "current_base_player": current_base_player,
            # "current_claimed_card":current_claimed_card,
    
            }
        # Convert everything to the integers the space is expecting
        observation['agent_cards'] = observation['agent_cards'] 
        observation['agent_deck_knowledge'] = [self._card_name_map[card.lower()] for card in observation['agent_deck_knowledge']]
        observation['claims'] = {agent: self._convert_claims_to_multibinary(observation['claims'][agent]) for agent in list(observation['claims'].keys())}
        observation['n_cards'] = observation['n_cards']
        # observation['money'] = observation['money']
        # observation['revealed'] = {self._card_name_map.get(key): value for key, value in observation['revealed'].items()}        
        # observation['current_base_player'] = observation['current_base_player']
        # observation['current_claimed_card'] = self._card_name_map.get(observation['current_claimed_card'], -1)
        
        ##############################################################################################################
        ########## Action Mask #######################################################
        ##############################################################################################################
        
        action_mask = self._compute_action_mask(agent)

    
        return {"observation": observation, "action_mask":action_mask} # type: ignore}
        
    
    def observe(self, agent: AgentID) -> dict[ObsType, ActionMask]:
        """
        Observes the self.state to get an observation for an agent
        
        env.last() calls this
        """
        # observation
        observation = self.state[agent]
        return observation
    

    
    
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
        
    
    def _compute_action_mask(self, agent:AgentID) -> ActionMask:    
        """Computes an action mask for the agent
        
        Will only be called if agent needs to make decision 
        
        

        Args:
            agent (AgentID): _description_

        Returns:
            ActionMask: _description_
        """
        
        # init action mask of 0s to represent valid actions
        a_mask = np.array([0] * len(self._actions), dtype=np.int8)

        next_action_type = self.game.turn.next_action_type # what type of action is able to be selected here
        agent_instance = self.game.players[agent]
        agent_money = agent_instance.coins
        current_base_action_instance = self.game.turn.current_base_action_instance
        
        if len(self.agents) == 1: # if game end just pass
            lo_valid_actions = self._pass_action
            lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
            a_mask[lo_valid_indexes] = 1
            return a_mask
        
    
        if next_action_type == 'claim_base_action':
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
            # if the agent is the same as the base player, the challenge round has completed
            # so they actually do not get to make a choise. they will execute the action they claimed
            if agent == self.game.turn.current_base_player.name:
                lo_valid_actions = self._exe
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            
            # if action is not challengable, pass
            if not current_base_action_instance.challengable:
                lo_valid_actions = self._pass_action
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            else: # action is challengable
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
            
        if next_action_type == 'pass_action':
            lo_valid_actions = self._pass_action
            lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
            a_mask[lo_valid_indexes] = 1
            return a_mask
            
        else:
            raise LookupError(f"Somethingg is not implemented properly. next action = {next_action_type}")
            
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
    
    def reset(self, seed=None, options=None):
        """
        Resets the game to a fresh game with freshly dealt cards
        
        Reset needs to re-initialize the following attributes
        - game
            -agents
            -agent_selection generator
        - observations
        - _cumulative_rewards
        - rewards
        - terminations
        - truncations
        - infos
        - action
        
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
        self.agents = [p.name for p in self.game.players] # integer
        
        ############################################################################
        ###### basic RL outputs for each agent ##############################
        ############################################################################
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}


        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"next_action_type":self.game.turn.next_action_type} for agent in self.agents}
        self.state = {agent: self._get_obs(agent) for agent in self.agents}      
        
        #### AGENT SELECTOR ALLOWS FOR STEPPING THROUGH AGENTS
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        ### action will be reset
        self.action = None
        
        return 

    
    def step(self, action: ActionType) -> None:
        """Makes an action by the current agent
        
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
            
        Updates
        -rewards # reward for the previous step
        -terminations
        -_cumulative_rewards # rewards over the game
        -truncations
        -infos
        -agent_selection
        
        """
        self._nstep +=1
        print(self._nstep)
        print("\n\n")
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]): # for when you died not on your turn
            print(f'Current agent is dead, {self.agent_selection}, skipping action')
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            
            # Update mask for next agent
            self.infos[self.agent_selection] = {"next_action_type":self.game.turn.next_action_type}
            return # step does nothing because agent is dead 
        
        ################################################################# 
        ###################### IS the Game over ########################### 
        ################################################################# 
        
        
        if len(self.agents)==1:
            print("~~~~~~~~~~~~~~~~  Game over  ~~~~~~~~~~~~~~~~~")
            print(f"~~~~~~~~~~~~~~~~  Agent {self.agents[0]} has won  ~~~~~~~~~~~~~~~~~")
            
            # update game state
            self.state = {agent: self._get_obs(agent) for agent in self.agents} 
            # update rewards
            self._reset_rewards()
            self.rewards  = {agent: self._get_reward(agent) for agent in self.agents}
            self._cumulative_rewards[self.agent_selection]= 0
            self._accumulate_rewards()
            
            return 
            
            
        
        # Its not over
        self.action = action
        agent = self.agent_selection
    
        
        print(f"Current Agent {agent}")
        print(f"Current action type {self.game.turn.next_action_type}")
        ################################################################# 
        ###################### STEP BLOCK ########################### 
        ################################################################# 
        action = self._convert_action(action) # action[0] = target agent, # action[1] = action_str
        print(f"Selected Action: {action}")


        # If we are at a base action
        if self.game.turn.next_action_type == "claim_base_action":
            self.game.turn.claim_base_action(agent, action)
            self.game.turn.next_action_type = "challenge_action"
            

        elif self.game.turn.next_action_type == "challenge_action":
            # if our agent is the same as the base player
            if agent == self.game.turn.current_base_player.name:
                print("~~~Challlenge round has reached base player, \n ignoring challenge action and executing base action~~~")
                # challenge round completed with no one challenging
                self.game.turn.exe_base_action() 
                self.game.turn.next_action_type = 'claim_base_action'
            elif list(action.values())[0] == 'pass': # non-base player passed
                pass
            elif list(action.values())[0] == 'challenge': # challenged
                print("Executing challenge")
                self.game.turn.exe_challenge(agent) 
                self.game.turn.next_action_type = "pass_action"


        elif self.game.turn.next_action_type == "block_action":
            # future implementation TODO
            pass
        
        elif self.game.turn.next_action_type == "pass_action":
            if agent == self.game.turn.current_base_player.name: # if the current agent is the base acting player,
                if self.game.turn.challenge.status: # if challenge succeeded
                    pass
                else: # challenge failed. execute base action
                    self.game.turn.exe_base_action()
                # next, the next agent will make a base action
                self.game.turn.next_action_type = 'claim_base_action'
            else: # otherwise agents just pass
                pass
            
        print(f"Next action type: {self.game.turn.next_action_type}")
        # update game state
        self.state = {agent: self._get_obs(agent) for agent in self.agents} 
        # update truncations
        # never update truncations right now -> but if you need to now then do it here
        
        # update terminations
        self.terminations = {agent: self._get_termination(agent) for agent in self.agents}

        # update rewards
        # First reset the reward dic so we dont accumulate last step's rewards
        self._reset_rewards()
        self.rewards  = {agent: self._get_reward(agent) for agent in self.agents}
        print(self._cumulative_rewards)

        self._accumulate_rewards() 
        print(self.rewards)
        print(self._cumulative_rewards)
        

        
        self.agent_selection = self._agent_selector.next()
        # Update mask for next agent
        
        next_action_type = self.game.turn.next_action_type
        self.infos[self.agent_selection] = {"next_action_type": next_action_type}

        return 
    
    def _get_termination(self, agent):
        player = self.game.players[agent]
        return player.status == 'dead'
    
    def _was_dead_step(self, action: ActionType) -> None:
        """Helper function that performs step() for dead agents.

        Does the following:

        1. Removes dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is dead, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Examples:
            Highly recommended to use at the beginning of step as follows:

        def step(self, action):
            if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
                self._was_dead_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)
        

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order: # another agent is dead
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0] # make next turn be next dead agent
        else: # no agent is dead
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
                assert False
            self._skip_agent_selection = None
            # selects the next agent.
            self.agent_selection = self._agent_selector.next() # pull request this I think?
        self._clear_rewards()
        
    def _reset_rewards(self):
        self.rewards = {agent: 0 for agent in self.agents}
        
    
    def _get_reward(self, agent) -> int:
        """Get reward of current agent
        
        Note. i may make this more expressive later on

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.terminations[agent] == True: # if agent died this turn
            return -1
        
        if len(self.agents)==1:        
            return 1
        
        return 0
        

    def _convert_action(self, action: ActionType) -> dict[AgentID, str]:
        """Converts action type to dictionary where
        AgentId is the target of the action
        and str is the string representation of the action

        Args:
            action (ActionType): _description_

        Returns:
            str: _description_
        """
        action_map = {value: key for key, value in self._action_space_map.items()}
        action_str = action_map[action]
        try:
            agent_id = action_str.split("_")[-1]
            if type(int(agent_id)) == int:
                agent_id = int(agent_id)
                action_str = "_".join(action_str.split("_")[:-1])
        except ValueError:
            agent_id = -1
            action_str = action_str
        
        return {agent_id:action_str}
    
    def _convert_claims_to_multibinary(self, observation):
        
        # Define the full set of possible keys (from both maps)
        all_keys = list(self._card_name_map.keys())
        key_to_index = {key: idx for idx, key in enumerate(all_keys)}
        # Initialize a MultiBinary array
        multi_binary = np.zeros(len(all_keys), dtype=int)

        # Populate the binary array based on observation['agent_claims'] (set)
        
        for claim in list(observation):
            key = claim.lower()
            if key in key_to_index:  # Check if the key is in either map
                multi_binary[key_to_index[key]] = 1
        return multi_binary
        

    def last_(self):
        return self.last()[0]





