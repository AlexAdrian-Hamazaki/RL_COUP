from coup_env import CoupEnv
import tqdm

class CoupPlayer:
    """This is a helper class to help run COUP in different ways.
    """
    
    @classmethod    
    def fill_replay_buffer(self, memory, opponent, n_players):
        """Fill the replay buffer with experiences collected by taking random actions in the environment.

        :param memory: Experience replay buffer
        :type memory: AgileRL experience replay buffer
        """
        print("Filling replay buffer ...")
        pbar = tqdm(total=memory.memory_size)
        
        env = CoupEnv(n_players = n_players)
        env.reset()
        
        while len(memory) < memory.memory_size:

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last() # return the last state 
                
                # Record state in memory buffer
                memory.save_to_memory_vect_envs() # TODO
                
                if termination or truncation:
                    action = None
                else:
                    # this is where you would insert your policy
                    action_mask = observation['action_mask']
                    action = env.action_space(agent).sample(action_mask) 
                env.step(action)
                
        env.close()