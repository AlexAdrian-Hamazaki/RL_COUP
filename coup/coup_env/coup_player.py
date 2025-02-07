from coup_env import CoupEnv
from tqdm import tqdm, trange

class CoupPlayer:
    """This is a helper class to help run COUP in different ways.
    """
    
    @classmethod    
    def fill_replay_buffer(cls, memory, n_players):
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
        mem_full = memory.memory_size
        
        for agent in env.agent_iter():
            turn_counter+=1
            # Observation is the previous state/observation whatever
            # And everything else is what the consequence was of the PREVIOUS agents action on the state
            
            # current observation is the agent's current view of the env
            # trans_reward is the previous agent's reward for transitioning into this state
            # trans_termionation is previous agent's terminating flag for transitioning into this state
            
            observation, reward, termination, _, info = env.last() # reads the observation of the last state from current agent's POV
            
        
            ###### SELECT ACTION #######
            if termination:
                # agent died at some point
                action = None
            else:
                # Randomly sample an action from the action mask
                action_mask = observation['action_mask']
                action = env.action_space(agent).sample(action_mask) 
        
    
            ######### STEP ############
            env.step(action) # current agent steps
            
            ######## SEE CONSEQUENCE OF STEP ##########
            next_observation, next_reward, next_termination,_ , info = env.last() # return the observation of the NEXT agent for the CURRENT agent's chosen action
            
            ######## Select things to put in memory buffer ########
            state = observation['observation'] # the state observed by our current agent
            next_state = next_observation['observation']
            
            # Save experiences to replay buffer
            memory.save_to_memory(
                state,
                action,
                reward,
                next_state,
                termination,
                is_vectorised=False,
            )
            ### If game ended reset it
            if info['next_action_type']=="win":
                env.reset()
                # print(f"Game ended after {turn_counter} turns, reset env")
                turn_counter = 0
            
            # print(mem_full - len(memory))
            if len(memory) % 100 ==0: 
                pbar.update(100)
            

            if int(len(memory)) == int(memory.memory_size):
                pbar.close()
                break
        
        print("Replay buffer warmed up")
        return memory
        
