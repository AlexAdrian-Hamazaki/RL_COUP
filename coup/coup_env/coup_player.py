from coup_env import CoupEnv
from tqdm import tqdm, trange
from gymnasium.spaces.utils import flatten

class CoupPlayer:
    """This is a helper class to help run COUP in different ways.
    """
    
    @classmethod    
    def fill_replay_buffer(cls, memory, n_players, obs_space):
        """Fill the replay buffer with experiences collected by taking random actions in the environment.

        :param memory: Experience replay buffer
        :type memory: AgileRL experience replay buffer
        
        Returns memory with full replay buffer
        """

        env = CoupEnv(n_players = n_players)
        env.reset()
        
        print("Filling replay buffer ...")
        pbar = tqdm(total=memory.memory_size)
        turn_counter = 0
        
        for agent in env.agent_iter():
            turn_counter+=1
            # Observation is the previous state/observation whatever
            # And everything else is what the consequence was of the PREVIOUS agents action on the state
            
            # current observation is the agent's current view of the env
            # trans_reward is the previous agent's reward for transitioning into this state
            # trans_termionation is previous agent's terminating flag for transitioning into this state
            
            observation, reward, termination, _, info = env.last() # reads the observation of the last state from current agent's POV
            state = observation['observation']
            
            ### If game ended reset it
            if info['next_action_type']=="win":
                
                env.reset()

                observation, reward, termination, _, info = env.last() # reads the observation of the last state from current agent's POV
                state = observation['observation']
                # print(f"Game ended after {turn_counter} turns, reset env")
                turn_counter = 0
            
            
            ###### SELECT ACTION #######
            if termination:
                action = None

            else:
                # Randomly sample an action from the action mask
                action_mask = observation['action_mask']
                action = env.action_space(agent).sample(action_mask) 
        
            ######### STEP ############
            env.step(action) # current agent steps
            
            
            ######## SEE CONSEQUENCE OF STEP ##########
            next_observation, reward, termination, _, info = env.last() # reads the observation of the last state from current agent's POV
            next_state = next_observation['observation']
            
            ########### FLATTEN STATES #############
            try:
                assert (next_state in obs_space)
            except AssertionError:
                print(next_state)
                assert False
            try:
                state = flatten(obs_space, state)
                next_state = flatten(obs_space, next_state)
            except IndexError as e:
                assert False

            ########### FILL NONE with Pass action #############
            try:
                assert state is not None, "Error: state is None!"
                assert action is not None, "Error: action is None!"
                assert reward is not None, "Error: reward is None!"
                assert next_state is not None, "Error: next_state is None!"
                assert termination is not None, "Error: termination is None!"
                assert info is not None, "Error: info is None!"
            except AssertionError as e:
                print(e)
                print("state:", state)
                print("action:", action)
                print("reward:", reward)
                print("next_state:", next_state)
                print("termination:", termination)
                print("info:", info)
                raise  # Re-raise the exception to catch the issue

            
            # Save experiences to replay buffer
            memory.save_to_memory(
                state,
                action,
                reward,
                next_state,
                termination,
                is_vectorised=False,
            )

            
            # print(mem_full - len(memory))
            if len(memory) % 100 ==0: 
                pbar.update(100)
            

            if int(len(memory)) == int(memory.memory_size):
                pbar.close()
                break
        
        print("Replay buffer warmed up")
        return memory
    
    @classmethod
    def flatten_obs(cls, obs_space, observation):
        
        flat = flatten(obs_space, observation)
        
        return flat
        
