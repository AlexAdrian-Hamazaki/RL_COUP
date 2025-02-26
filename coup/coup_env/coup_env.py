from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Sequence, Dict, Tuple, MultiDiscrete, MultiBinary
from gymnasium.spaces.utils import flatten_space
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
import tqdm

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
        self._nstep = 0
        
        ############################################################################
        ###### Mapping integers to names of things ##############################
        ############################################################################
        self._card_names = ['unknown', 'assassin','ambassador', 'duke', 'contessa', 'captain']
        self._card_names_ints = [-1, 0, 1, 2, 3, 4]
        self._card_name_map = dict(zip(self._card_names, self._card_names_ints))
        
        
        # Generate all possible combinations of 2 integers (with repetition allowed)
        card_combinations = list(combinations_with_replacement(self._card_names_ints[:], r=2))
        self._card_combination_map = {index: combo for index, combo in enumerate(card_combinations)}
        
        
        # Possible next_Action
        self.NEXT_ACTION_TYPE_MAP = {"exe_action":0,
                                     "claim_base_action":1,
                                     "pass_action":2,
                                     "challenge_action":3,
                                     "block_action":4}
            
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
        self._no_action = ['none']
        self._steal_actions = [f"steal_{agent}" for agent in self.agents]
        self._base_actions = self._base_actions + self._steal_actions
        
        self._assassinate_actions = [f"assassinate_{agent}" for agent in self.agents]
        self._coup_actions = [f"coup_{agent}" for agent in self.agents]
        self._challenge_action = ["challenge"] 
        self._pass_action = ['pass']
        
        self._actions = self._base_actions + self._assassinate_actions + self._coup_actions + self._challenge_action + self._pass_action + self._exe + self._no_action
        self._actions.sort()
        self._action_space_map = dict(zip([action for action in self._actions],
                                          [n for n in range(len(self._actions))]
                                          ))
                
        self.action_space_dict = Discrete(len(self._actions), start = 0) # first is action, second is target player, -1= No target
        self.action_spaces = dict(zip([agent for agent in self.agents],
                                           [self.action_space_dict for _ in self.agents]))
        
        print("Action Map")
        print(self._action_space_map)
        
        ############################################################################
        ###### Observation spaces for each agent ##############################
        ############################################################################
        # init a gem instance to help define observation space

        self.observation_space_dict = Dict({
            'observation': 
                Dict({
                    'agent_cards': Discrete(len(self._card_combination_map.keys())), # pairs of cards here
                    # "agent_deck_knowledge": MultiDiscrete([6] * deck_size, start=[-1]*deck_size), #Order o deck, -1 indicates we do not know what the card is. # TODO figure out how to limit this to only correct observations like (-1,-1,2,4
                    # # Maybe will need to throw out bad obs?
                    "claims": Dict(dict(zip([n for n in range(self.n_players)], 
                                            [MultiBinary([6]) for _ in range(self.n_players)]))), # Player_int: Text # dict of text spaces for what others are claiming,
                    
                    "n_cards": Dict(dict(zip([n for n in range(self.n_players)],
                                            [Discrete(3, start = 0) for _ in range(self.n_players)]))),
                    
                    "money": Dict(dict(zip([n for n in range(self.n_players)],
                                                [Discrete(14) for _ in range(self.n_players)]))), # Player_int: Discrete
                    
                    "revealed": Dict(dict(zip(self._card_names_ints[1:], 
                                            [Discrete(4, start = 0) for _ in range(len(self._card_names_ints[1:]))]))), # Card name, number revealed)
                        
                    "current_base_player": Discrete(len(self.agents), start = 0), 
                    "current_claimed_action": Discrete(len(self._actions), start = 0), # may not need this
                    'current_acting_player': Discrete(len(self.agents), start = 0),
                    "next_action_type": Discrete(len(list(self.NEXT_ACTION_TYPE_MAP.keys()))),
                    
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
        # Ensure all keys are strings and all values are integers
        revealed = {str(k): int(v) for k, v in revealed.items()}
        turn_order = self.game.turn.turn_order
        current_base_player= self.game.turn.current_base_player.name
        current_claimed_card = self.game.turn.current_base_action_instance.card
        
        if self.game.turn.current_base_action_instance.has_target:

            current_claimed_action = f"{self.game.turn.current_base_action_instance.name}_{self.game.turn.current_base_action_instance.target_player.name}"
        else:
            current_claimed_action = self.game.turn.current_base_action_instance.name
        
        money = self.game.n_coins
        money = {int(k): int(v) for k, v in money.items()}

        n_cards = self.game.n_cards
        n_cards = {int(k): int(v) for k, v in n_cards.items()}
        
        claims = self.game.claimed_cards
        claims = {int(k): set(str(i) for i in v) for k,v in claims.items()}

        ###############################################################################
        ######################## OBSERVATION #######################################################
        ##############################################################################################################
        
    
        observation = {
            "agent_cards": int(card_combo) ,
            # "agent_deck_knowledge": np.array(agent_deck_knowledge),
            "claims": claims,
            "n_cards": n_cards,
            "money": money,
            "revealed": revealed,
            "current_base_player": int(current_base_player),
            "current_claimed_action": str(current_claimed_action),
            'next_action_type': str(self.game.turn.next_action_type),
            'current_acting_player': int(self.agent_selection)
    
            }
        # Convert everything to the integers the space is expecting
        observation['agent_cards'] = observation['agent_cards'] 
        # observation['agent_deck_knowledge'] = [self._card_name_map[card.lower()] for card in observation['agent_deck_knowledge']]
        
        # print(observation['claims'][0])
        
        observation['claims'] = {agent: self._convert_claims_to_multibinary(observation['claims'][agent]) for agent in list(observation['claims'].keys())}
        observation['n_cards'] = observation['n_cards']
        observation['money'] = observation['money']
        observation['revealed'] = {self._card_name_map.get(key): value for key, value in observation['revealed'].items()}        
        observation['current_base_player'] = observation['current_base_player']
        observation['current_claimed_action'] = self._action_space_map.get(observation['current_claimed_action'], -1)
        observation['next_action_type'] = self.NEXT_ACTION_TYPE_MAP[self.game.turn.next_action_type]
        observation['current_acting_player'] =  self.agent_selection
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
                lo_valid_actions = self._assassinate_actions + self._coup_actions # enough money for all actions
                lo_valid_actions = self._remove_self_target(lo_valid_actions, agent) # remove ability to coup, assassinate, or steal from self
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            
            elif agent_money >=7:
                lo_valid_actions = self._assassinate_actions + self._coup_actions + self._base_actions
                lo_valid_actions = self._remove_self_target(lo_valid_actions, agent) # remove ability to coup, assassinate, or steal from self

                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask

            elif agent_money >=3:
                lo_valid_actions = self._assassinate_actions + self._base_actions
                lo_valid_actions = self._remove_self_target(lo_valid_actions, agent) # remove ability to coup, assassinate, or steal from self
                lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
                a_mask[lo_valid_indexes] = 1
                return a_mask
            
            else:
                lo_valid_actions = self._base_actions
                lo_valid_actions = self._remove_self_target(lo_valid_actions, agent) # remove ability to coup, assassinate, or steal from self
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
        
        if next_action_type == 'exe_action':
            lo_valid_actions = self._exe
            lo_valid_indexes = [self._action_space_map[action] for action in lo_valid_actions]
            a_mask[lo_valid_indexes] = 1
            return a_mask
            
        else:
            raise LookupError(f"Somethingg is not implemented properly. next action = {next_action_type}")
        
    def _remove_self_target(self, lo_valid_actions, agentID):
        lo_valid_actions = [action for action in lo_valid_actions if str(agentID) not in action]
        return lo_valid_actions
        
            
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
        
        Returns state, info
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
        
        #### AGENT SELECTOR ALLOWS FOR STEPPING THROUGH AGENTS
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"next_action_type":self.game.turn.next_action_type} for agent in self.agents}
        self.state = {agent: self._get_obs(agent) for agent in self.agents}      
    
        
        ### action will be reset
        self.action = None
        
        return self.state, self.infos
    
    def check_game_over(self, prev_state):
        """ 
        Game ends when there is only 1 agent left
        
        If game is over, dish out rewards
        
        If game ends, infos[agent] next action will be "win"
        """
        if len(self.agents)==1:
            #print("~~~~~~~~~~~~~~~~  Game over  ~~~~~~~~~~~~~~~~~")
            #print(f"~~~~~~~~~~~~~~~~  Agent {self.agents[0]} has won  ~~~~~~~~~~~~~~~~~")
            # update game state
            self.state = {agent: self._get_obs(agent) for agent in self.agents} 
            # update rewards. This time game_win should be accurate as should game_end
            self.rewards  = {agent: self._get_reward(agent, prev_state) for agent in self.agents}
            # self._cumulative_rewards[self.agent_selection]= 0
            self._accumulate_rewards()
            self.infos[self.agent_selection] = {'next_action_type':"win"}

            return True
            
            
        

    
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
        #print(self._nstep)
        #print("\n\n")

        prev_state = self.state.copy()


        # Game continues, select agent and make a turn
        self.action = action
        agent = self.agent_selection
        
        # print(f"Found acting agent {agent}")
    
        
        #print(f"Current Agent {agent}")
        #print(f"Current action type {self.game.turn.next_action_type}")
        ################################################################# 
        ###################### STEP BLOCK ########################### 
        ################################################################# 
        action = self._convert_action(action) # action[0] = target agent, # action[1] = action_str
        #print(f"Selected Action: {action}")
        
        # if we are at an execute action
        if self.game.turn.next_action_type == "exe_action":
            self.game.turn.exe_base_action() 
            self.game.turn.next_action_type = 'claim_base_action'
            
            
        
        # If we are at a base action
        elif self.game.turn.next_action_type == "claim_base_action":
            
            self.game.turn.claim_base_action(agent, action)
            
            # SET NEXT ACTION TYPE TO BE CHALLENGE
            self.game.turn.next_action_type = "challenge_action"
            
        # If we are at a challenge action
        elif self.game.turn.next_action_type == "challenge_action":
            # We go around the table and people decide if they want to pass or challenge
            if list(action.values())[0] == 'pass': # non-base player passed
                pass
            elif list(action.values())[0] == 'challenge': # challenged
                # print("Executing challenge")
                self.game.turn.exe_challenge(agent) # this will lose a live from someone, and set challenge.status to T or F
            
            # print(f"Now we are in a challenge action, where acting agent is {agent}")
            # print(f"Agent has taken action {list(action.values())[0]}")
            
            # print("Now we go to next agent")
            # If the NEXT agent is the base_player, the next action type becomes exe_action
            next_index = (self.agent_selection + 1) % len(self.agents)
            if self.agents[next_index] == self.game.turn.current_base_player.name:
                # print(f"next agent {self.agents[next_index]}")
                # print(f"We found that the next agent is the same as the base_acting agent, setting action to exe")
                
                self.game.turn.next_action_type = "exe_action" # Everyone else passes once someone has challenged
                
            else:
                self.game.turn.next_action_type = "pass_action"  # Else, next action becomes pass because someone's already challenged
                
            
        # elif self.game.turn.next_action_type == "block_action":
        #     # future implementation TODO

        #     # If the NEXT agent is the base_player, the next action type becomes exe_action
        #     if self._agent_selector.next() == self.game.turn.current_base_player.name:
        #         self.game.turn.next_action_type = "exe_action" # Everyone else passes once someone has challenged
        #     pass
        
        # elif self.game.turn.next_action_type == "pass_action":
        #     if self._agent_selector.next()  == self.game.turn.current_base_player.name: # if the current agent is the base acting player
        #         self.game.turn.next_action_type = "exe_action" 
        #     else:
        #         self.game.turn.next_action_type = "pass_action" 
            
        #print(f"Next action type: {self.game.turn.next_action_type}")
        # update game state
        # update truncations
        # never update truncations right now -> but if you need to now then do it here
        
        # print(f"Next acting agent should be {self.agent_selection}")
        # print(f"Next action type should be {self.game.turn.next_action_type}")
        # print('\n')
        # Update mask for next agent
        
        ################################################################# 
        ###################### Update Next Agent ########################### 
        ################################################################# 
        
        self.state = {agent: self._get_obs(agent) for agent in self.agents} 
        
        next_action_type = self.game.turn.next_action_type
        self.infos[self.agent_selection] = {"next_action_type": next_action_type}
        
        ######################  ########################################### 
        ###################### handeling agent death ########################### 
        ################################################################# 
        
        # update terminations
        self.terminations = {agent: self._get_termination(agent) for agent in self.agents}
        
        ######################  ########################################### 
        ###################### Updating rewards ########################### 
        ################################################################# 
        # First reset the reward dic so we dont accumulate last step's rewards
        self._reset_rewards()
        self.rewards  = {agent: self._get_reward(agent, prev_state) for agent in self.agents}
        self._accumulate_rewards() # add rewards for this agent
        


        #print(self.rewards)

        # Remove Dead Agents
        [self.remove_dead_agents(agent) for agent in self.agents]
        
        self.agent_selection = self._agent_selector.next()

        
        ################################################################# 
        ###################### IS the Game over ########################### 
        ################################################################# 
        
        if self.check_game_over(prev_state): # raise termination flag in infos
            # print(self.rewards)
            # print(self._cumulative_rewards)
            return # game is over
        return 
    
    def _get_termination(self, agent):
        player = self.game.players[agent]
        if len(player.cards) == 0:
            return True
        else:
            return False
    
    def _was_dead_step(self, agent) -> None:
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
        # if action is not None:
        #     raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        # del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)
        
                
        # finds next dead agent
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order: # another agent is dead ### THIS WILL NEVER HAPPEN WITH COUP BECAUSE ONLY 1 PERSON CAN DIE AT A TIME
            assert False # This is put here rn to see if we ever enter here which i dont think we ever do
            if getattr(self, "_skip_agent_selection", None) is None: # if this is NONE, then we bypass normal agent iter to mak
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0] # make next turn be next dead agent so we can keep removing them
            
        else: # no more agents are dead, load live agent
            self.agent_selection = self._agent_selector.next()
            
        self._clear_rewards()
        
    def _reset_rewards(self):
        self.rewards = {agent: 0 for agent in self.agents}
        
    
    def _agent_won(self, agent) -> bool:
        """check if agent won

        """
        if len(self.agents) == 1 and agent in self.agents:
            return True
        
        else:
            return False
                    
    def _agent_lost(self,agent) -> bool:
        """check if agent lost

        """
        if self.terminations[agent] == True: # if agent died this turn
            return True
        else:
            return False
        
    def _agent_gained_coins(self, agent, prev_state) -> int:
        """Return a number of coins equal to how many the agent just got

        Args:
            agent (_type_): _description_

        Returns:
            int: _description_
        """
        prev_coins = prev_state[agent]['observation']['money'][agent]
        coins = self.state[agent]['observation']['money'][agent]
        diff_coins =coins-prev_coins
        return diff_coins
        
    def _agent_killed(self, agent, prev_state) -> bool:
        """Return a bool indicating if current agen't step killed another agent's card

        Args:
            agent (_type_): _description_

        Returns:
            int: _description_
        """
        prev_agent_cards = prev_state[agent]['observation']['n_cards'].copy()
        agent_cards = self.state[agent]['observation']['n_cards'].copy()
        
        
        prev_agent_cards.pop(agent)
        agent_cards.pop(agent)
        
        for iter_agent in list(prev_agent_cards.keys() & agent_cards.keys()):  # Find common keys
            difference_in_cards = int(agent_cards[iter_agent]) - int(prev_agent_cards[iter_agent])  # Subtract values
            
            if difference_in_cards < 0:

                return True
        return False
    
    def _agent_lost_life(self, agent, prev_state) -> bool:
        """Return a bool indicating if current agent lost a card

        Args:
            agent (_type_): _description_

        Returns:
            int: _description_
        """
        prev_agent_cards = prev_state[agent]['observation']['n_cards']
        agent_cards = self.state[agent]['observation']['n_cards']
        difference_in_cards = agent_cards[agent] - prev_agent_cards[agent]  # Subtract values
        
        if difference_in_cards <0:
            return True
        return False

    
    
    def _get_reward(self, agent, prev_state) -> int:
        """Get reward of current agent
    
        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get rewards for each game state change
        win  = 1
        lose = -1
        coins = 0.1
        kill = 0.5
        lose_life = -0.5
        
        
        # init reward for this agent
        reward = 0
        
        # check if agent just won game
        if self._agent_won(agent):
            reward+=win
        
        # check if agent just lost game
        if self._agent_lost(agent):
            reward+=lose
        
        # check if agent got coins. Get coins value reward for each coin you get
        if self._agent_gained_coins(agent, prev_state)>0:
            reward+=coins*self._agent_gained_coins(agent,prev_state)
        
        # check if agent killed someone
        if self._agent_killed(agent, prev_state):
            reward+=kill
            
        # check if agent lost life
        if self._agent_lost_life(agent, prev_state):
            reward+=lose_life
        
        return reward
        

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


    def last(
        self, observe: bool = True
    ) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection

        assert agent is not None
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )
        
    def remove_dead_agents(self, agent):
        if (self.terminations[agent] or self.truncations[agent]): # for when you died not on your turn
            # print(f'Current agent is dead, {self.agent_selection}, skipping action')
        
            self._was_dead_step(agent)
            
            # Handle the case where the agent dies mid round or something. I'm pretty sure I need to set the next action type to be a base action
            self.infos[agent] = {"next_action_type":self.game.turn.next_action_type}
        
    def last_all(self):
        
        observations = {agent : self.observe(agent) for agent in self.agents}
        
        
        rewards = {agent:self._cumulative_rewards[agent] for agent in self.agents}
        terminations = {agent:self.terminations[agent] for agent in self.agents}
        truncations = {agent:self.truncations[agent] for agent in self.agents}
        infos = {agent:self.infos[agent] for agent in self.agents}
        
        return (observations, rewards, terminations, truncations, infos)
        
            