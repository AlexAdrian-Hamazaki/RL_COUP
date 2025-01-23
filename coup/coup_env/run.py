#!/usr/bin/env python3

from coup_env import CoupEnv
from collections import defaultdict
import gymnasium as gym
import numpy as np

class SimpleAgent():
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, env) -> int:
        """
        agent gets valid action according to action mask
        """
        mask = env._compute_action_mask()
        # Sample valid actions for each instance in the mask
        valid_actions = []
        for row in mask:
            # Get indices of valid actions for this row
            valid_indices = np.flatnonzero(row)
            if valid_indices.size == 0:
                raise ValueError("No valid actions available for some instance in the mask.")
            print(valid_indices)
            # Sample a valid action from the valid indices
            sampled_action = np.random.choice(valid_indices)
            valid_actions.append(sampled_action)
        return np.array(valid_actions)
 

        
def main():
    # hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1
    env = CoupEnv(4)
    
    agent = SimpleAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    
    for _ in range(10): # for each episode
        
        observation = env.reset()
        done = False
        # action_type = observation['action_type']
        
        while not done:
            print("AGENT STEP")
            observation, reward, terminated, truncated, info = env.step()
            done = terminated or truncated
            # update the agent
            # agent.update(obs, action, reward, terminated, agents_action_observation)
            
            print("BOT STEP")
            # each bot make their random steps and affects the action_type parameter
            observation, reward, terminated, truncated, info = env.bot_step() # bot makes random action based on previous action. Lets assume the bots just do normal actions

        # reduce exploration level each episode
        agent.decay_epsilon()
        
    
    
    
    
    

    
    
    
if __name__ == "__main__":
    main()