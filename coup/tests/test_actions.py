

# Importing Coup Env
from coup_env.coup_env import CoupEnv
from coup_env.coup_player import CoupPlayer
from opponent import Opponent, RandomOpponent

import copy
import os
import random
from collections import deque
from datetime import datetime

# Utility Imports
import json
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import torch
import sys
import copy
import os
import random
from collections import deque
from datetime import datetime
from gymnasium.spaces.utils import flatten, unflatten

import pytest
# Agile imports
from agilerl.algorithms.dqn import DQN

def translate_dict(data_dict, _card_combination_map, card_name_map, NEXT_ACTION_TYPE_MAP, ACTION_SPACE_MAP):
    """
    Translates values in a dictionary based on specified mapping dictionaries.

    Args:
    - data_dict (dict): The dictionary containing the data to be translated.
    - _card_combination_map (dict): A dictionary to map `agent_cards` values.
    - card_name_map (dict): A dictionary to map `revealed` and `current_claimed_card` values.
    - NEXT_ACTION_TYPE_MAP (dict): A dictionary to map `next_action_type` values.

    Returns:
    - dict: The translated dictionary with updated values.
    """
    # Translate `agent_cards`
    data_dict['agent_cards'] = _card_combination_map.get(data_dict['agent_cards'], 'Unknown')
    data_dict['agent_cards'] = [card_name_map.get(card) for card in data_dict['agent_cards']]

    # Translate `revealed` keys
    data_dict['revealed'] = {card_name_map.get(int(k)): v for k, v in data_dict['revealed'].items()}
    data_dict['claims'] = {k: [card_name_map.get(i) for i,card in enumerate(v) if card==1] for k,v in data_dict['claims'].items()}

    # Translate `current_claimed_card`
    data_dict['current_claimed_action'] = ACTION_SPACE_MAP.get(data_dict['current_claimed_action'], 'ERROR')

    # Translate `next_action_type`
    data_dict['next_action_type'] = NEXT_ACTION_TYPE_MAP.get(data_dict['next_action_type'], 'ERROR')
    return data_dict

def translate_action_column(df, action_map):
    df['action'] = df['action'].apply(lambda x: action_map.get(x, 'Unknown'))
    return df

def get_maps():
    coup_env = CoupEnv(2)
    # coup_env._action_space_map
    ACTION_SPACE_MAP = {v: k for k, v in coup_env._action_space_map.items()}
    CARD_NAME_MAP = {v: k for k, v in coup_env._card_name_map.items()}
    NEXT_ACTION_TYPE_MAP = {v: k for k, v in coup_env.NEXT_ACTION_TYPE_MAP.items()}
    CARD_COMBINATION_MAP = coup_env._card_combination_map
    
    return ACTION_SPACE_MAP, CARD_NAME_MAP, NEXT_ACTION_TYPE_MAP, CARD_COMBINATION_MAP

@pytest.fixture
def path_to_actions():
    return "/home/aadrian/Documents/RL_projects/RL_COUP/coup/metrics/actions/2048_lesson1_actions.jsonl"

import yaml 
@pytest.fixture
def lesson():
    with open("/home/aadrian/Documents/RL_projects/RL_COUP/curriculums/lesson1.yaml") as file:
        LESSON = yaml.safe_load(file)
    return LESSON

@pytest.fixture
def load_actions(path_to_actions):
    df = pd.read_json(path_to_actions,   lines=True)
    ACTION_SPACE_MAP, CARD_NAME_MAP, NEXT_ACTION_TYPE_MAP, CARD_COMBINATION_MAP = get_maps()
    n_actions = len(df['action'].unique())
    
    df = translate_action_column(df, action_map=ACTION_SPACE_MAP)
    assert n_actions == len(df['action'].unique())
    
    df['state'] = df['state'].apply(lambda x: translate_dict(x, CARD_COMBINATION_MAP, CARD_NAME_MAP, NEXT_ACTION_TYPE_MAP, ACTION_SPACE_MAP))
    df['next_state'] = df['next_state'].apply(lambda x: translate_dict(x, CARD_COMBINATION_MAP, CARD_NAME_MAP, NEXT_ACTION_TYPE_MAP, ACTION_SPACE_MAP))
    
    # unpack states
    state = pd.DataFrame(df['state'].apply(pd.Series))
    state.columns = [f"ORIGONAL_{col}" for col in state.columns]
    next_state = pd.DataFrame(df['next_state'].apply(pd.Series))
    next_state.columns = [f"STEPPED_{col}" for col in next_state.columns]

    
    df = pd.concat([df, state, next_state], axis = 1)

    assert 'action_mask' in df.keys()
    df['action_mask'] = df.apply(lambda x: translate_action_mask(x, ACTION_SPACE_MAP), axis=1)

    return df

def translate_action_mask(row, ACTION_SPACE_MAP):
    # Extract the original action mask from the row
    action_mask = row['action_mask']
    # Ensure the action_mask is a list or array
    if not isinstance(action_mask, (list, tuple)):  
        return None  # or handle the error as needed
    # Translate action mask using ACTION_SPACE_MAP
    translated_mask = [ACTION_SPACE_MAP[index] for index, action_bool in enumerate(action_mask) if action_bool == 1]
    return translated_mask  # Returning a list (not modifying row directly)


# Test to show that income action increases current base player's money
def test_load_actions(load_actions):
    assert type(load_actions) ==pd.DataFrame
    
    
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@pytest.fixture
def states(load_actions):
    states = pd.DataFrame(load_actions.state.tolist())
    return states

@pytest.fixture
def next_states(load_actions):
    next_states = pd.DataFrame(load_actions.next_state.tolist())
    return next_states

def test_claim_actions1(load_actions):
    # Test to make sure that when we claim a new action, the current claimed action is "none"
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type'] == "claim_base_action"]
    assert (actions['ORIGONAL_current_claimed_action'] == 'none').all()

def test_claim_action2(load_actions):
    pd.set_option('display.max_columns', 10)
    # test to make sure that when we claim a new action, the current_claimed_action for next step (challenge step) is the claimed action
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type'] == "claim_base_action"]
    actions =  actions.loc[:,['action', 'ORIGONAL_next_action_type', 'STEPPED_next_action_type', 'ORIGONAL_current_claimed_action', 'STEPPED_current_claimed_action']]
    assert (actions.loc[:,'STEPPED_current_claimed_action'] == actions['action']).all()

    # check to make sure that the situations where challenge action -> challenge action occurs, base actin stays same
    actions = load_actions[(load_actions.loc[:,'ORIGONAL_next_action_type'] == "challenge_action") & (load_actions.loc[:,'STEPPED_next_action_type'] == "challenge_action")]
    if not actions.empty:
        assert (actions.loc[:,'ORIGONAL_current_claimed_action'] == actions.loc[:,'STEPPED_current_claimed_action']).all()
        
    # check to make sure that the situations where challenge_action -> block action occurs, base action stays same
    # TODO
    
    # checkt to make sure situation where challenge action -> exe action occurs, base action moves to 'none' # TODO WHEN BLOCK IS IMPLEMETED CHANGE THIS
    actions = load_actions[(load_actions.loc[:,'ORIGONAL_next_action_type'] == "challenge_action") & (load_actions.loc[:,'STEPPED_next_action_type'] == "exe_action")]
    if not actions.empty:
        assert (actions.loc[:,'ORIGONAL_current_claimed_action'] == actions.loc[:,'STEPPED_current_claimed_action']).all()
        
def test_claim_action3(load_actions):
    # Test to make sure that when the next_action_type is challenge, action mask is challenge or pass and so is action
    actions = load_actions[(load_actions.loc[:,'ORIGONAL_next_action_type'] == "challenge_action")]
    assert (actions['action_mask'].apply(lambda x: set(x) <= {"challenge", "pass"} and len(x) <= 2)).all()
    
    # Test to make sure action is either pass or challenge
    assert (actions['action'].isin(['pass','challenge'])).all()
    
    
    
    
def test_prev_and_next_same(load_actions):
    # check that the next_state is the same as the next line's state
    grouped = load_actions.groupby('game_id')
    for id, grouped_df in grouped:
        for i in range(len(grouped_df)-1):
            grouped_df.reset_index(drop = True, inplace = True)
            assert grouped_df.loc[i, 'next_state'] == grouped_df.loc[i+1, 'state'], f"Mismatch at row {i}: next_state ({grouped_df.loc[i, 'next_state']}) != state ({grouped_df.loc[i + 1, 'state']})"
            
            
def test_base_player_swapping(load_actions):
    # check to make sure base player is swapping @ claimed actions and exe so everyones getting aturn
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="claim_base_action"]
    
    actions = actions.loc[:,['game_id','ORIGONAL_current_base_player', 'STEPPED_current_base_player']]
    
    for id, game_df in actions.groupby('game_id'):
        print(game_df['ORIGONAL_current_base_player'])
        print(game_df['ORIGONAL_current_base_player'].shift(1))
        print(game_df['STEPPED_current_base_player'])
        assert (game_df['ORIGONAL_current_base_player'] != game_df['ORIGONAL_current_base_player'].shift(1)).fillna(True).all()
        assert (game_df['ORIGONAL_current_base_player'] == game_df['STEPPED_current_base_player']).all()


def test_income_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action']=='income']
    actions = actions.loc[:,['action',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward","ORIGONAL_money", "STEPPED_money"]]
    print(actions)
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        state_money = row['ORIGONAL_money'][current_base_player]
        next_state_money = row['STEPPED_money'][current_base_player]
        assert next_state_money == state_money+1 # checks To make sure money upticks
        assert float(row['reward']) == float(lesson['rewards']['coins']) # check to make sure correct reward is received
        
def test_foreign_aid_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action']=='foreign_aid']
    actions = actions.loc[:,['action',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward","ORIGONAL_money", "STEPPED_money"]]
    print(actions)
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        state_money = row['ORIGONAL_money'][current_base_player]
        next_state_money = row['STEPPED_money'][current_base_player]
        assert next_state_money == state_money+2 # checks To make sure money upticks
        assert round(float(row['reward']),2) == round(float(lesson['rewards']['coins']*2),2) # check to make sure correct reward is received
        
def test_tax_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action']=='tax']
    actions = actions.loc[:,['action',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward","ORIGONAL_money", "STEPPED_money"]]
    print(actions)
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        state_money = row['ORIGONAL_money'][current_base_player]
        next_state_money = row['STEPPED_money'][current_base_player]
        assert next_state_money == state_money+3 # checks To make sure money upticks
        assert round(float(row['reward']),2) == round(float(lesson['rewards']['coins']*3),2) # check to make sure correct reward is received
        
def test_steal_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action'].str.contains(r'^steal_\d$')]
    actions['target'] = actions['ORIGONAL_current_claimed_action'].str.split("_").str[-1]
    
    actions = actions.loc[:,['action', 'target',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward","ORIGONAL_money", "STEPPED_money"]]
    

    print(actions.head())
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        player_state_money = row['ORIGONAL_money'][current_base_player]
        player_next_state_money = row['STEPPED_money'][current_base_player]
        
        current_target_player = str(row['target'])
        target_state_money = row['ORIGONAL_money'][current_target_player]
        target_next_state_money = row['STEPPED_money'][current_target_player]
        
        stolen_money = target_state_money - target_next_state_money
        print(f"Stolen money {stolen_money}")
        print(f"reward {round(float(row['reward']),2)}")
        print(f"expected reward {round(float(lesson['rewards']['coins']*stolen_money),2)}")
        assert current_base_player != current_target_player # make sure player is not stealing from self
    
        assert round(float(row['reward']),2) == round(float(lesson['rewards']['coins']*stolen_money),2) # check to make sure correct reward is received
        assert True # TODO check that cumulative rewards give -0.2 to whoever was stolen from
        
        assert player_next_state_money == player_state_money+stolen_money # checks To make sure money upticks
        assert target_next_state_money <= target_state_money # checks To make sure money downticks for target
        
        
def test_assassinate_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action'].str.contains(r'^assassinate_\d$')]
    actions['target'] = actions['ORIGONAL_current_claimed_action'].str.split("_").str[-1]
    
    actions = actions.loc[:,['action', 'target',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward", "ORIGONAL_money", "STEPPED_money", "ORIGONAL_n_cards", "STEPPED_n_cards"]]
    print(actions.head())
    expected_reward = round(float(lesson['rewards']['kill']), 2)
    print(f"expected reward {expected_reward}")
    
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        player_state_money = row['ORIGONAL_money'][current_base_player]
        player_next_state_money = row['STEPPED_money'][current_base_player]
    
        current_target_player = str(row['target'])
        origonal_target_lives = row['ORIGONAL_n_cards'][current_target_player]
        stepped_target_lives = row['STEPPED_n_cards'][current_target_player]
        
        assert current_base_player != current_target_player # make sure player is not stealing from self
        
        assert origonal_target_lives-1 == stepped_target_lives # make sure target player lost a life
        print(f"seen reward {round(float(row['reward']),2)}")
        assert round(float(row['reward']), 2) == round(float(lesson['rewards']['kill']), 2) or round(float(lesson['rewards']['kill']), 2) + round(float(lesson['rewards']['win']), 2), "Check to make sure the correct reward is received"

def test_claim_assassinate(load_actions, lesson):
    # test that you lose 3 money and reward when you claim assassinate
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type'] == "claim_base_action"]
    actions = actions[actions.loc[:,'action'].str.contains(r'^assassinate_\d$')]
    actions['target'] = actions['action'].str.split("_").str[-1]

    actions = actions.loc[:,['action', 'target','ORIGONAL_next_action_type','ORIGONAL_current_base_player', "reward", "ORIGONAL_money", "STEPPED_money", "ORIGONAL_n_cards", "STEPPED_n_cards"]]

    print(actions)
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        # Use current_base_player to index the corresponding rows
        state_money = row.loc['ORIGONAL_money'][current_base_player]
        next_state_money = row.loc['STEPPED_money'][current_base_player]
        assert state_money- 3 == next_state_money
        assert (round(float(row['reward']),2) == -round(float(lesson['rewards']['coins']*3),2))

def test_coup_actions(load_actions, lesson):
    pd.set_option('display.max_columns', 10)
    actions = load_actions[load_actions.loc[:,'ORIGONAL_next_action_type']=="exe_action"]
    actions = actions[actions.loc[:,'ORIGONAL_current_claimed_action'].str.contains(r'^coup_\d$')]
    actions['target'] = actions['ORIGONAL_current_claimed_action'].str.split("_").str[-1]
    
    actions = actions.loc[:,['action', 'target',"ORIGONAL_current_claimed_action",'ORIGONAL_current_base_player', "reward", "ORIGONAL_money", "STEPPED_money", "ORIGONAL_n_cards", "STEPPED_n_cards"]]
    print(actions.head())
    expected_reward = round(float(lesson['rewards']['kill']), 2)
    print(f"expected reward {expected_reward}")
    
    for i, row in actions.iterrows():
        current_base_player = str(row['ORIGONAL_current_base_player'])
        player_state_money = row['ORIGONAL_money'][current_base_player]
        player_next_state_money = row['STEPPED_money'][current_base_player]
    
        current_target_player = str(row['target'])
        origonal_target_lives = row['ORIGONAL_n_cards'][current_target_player]
        stepped_target_lives = row['STEPPED_n_cards'][current_target_player]
        
        assert current_base_player != current_target_player # make sure player is not couping self
        assert player_next_state_money == player_state_money-7 # checks To make sure money downticks
        
        assert origonal_target_lives-1 == stepped_target_lives # make sure target player lost a life
        print(f"seen reward {round(float(row['reward']),2)}")
        assert round(float(row['reward']), 2) == round(float(lesson['rewards']['kill']), 2) or round(float(lesson['rewards']['kill']), 2) + round(float(lesson['rewards']['win']), 2), "Check to make sure the correct reward is received"


        
        
        
def test_cumulative_rewards(load_actions, lesson):
    # check that cumulative reward tracking
    assert True




# Test to show that rewards combine with cumulative rewards properly

# test to show that all actions work as intended

# test to show that next_state is equal to the state of the next action