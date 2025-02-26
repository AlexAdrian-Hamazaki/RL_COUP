from agilerl.algorithms.dqn import DQN

import numpy as np

class Opponent:
    """Coup opponent to train and/or evaluate against.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param difficulty: Difficulty level of opponent, 'random', 'weak' or 'strong'
    :type difficulty: str
    """
    def __init__(self, dqn_path, device):
        self.dqn_path= dqn_path
        self.device = device
        
        self.load_model()
        
    def __str__(self):
        return f"Model from {self.dqn_path}"
    
        
    def load_model(self):
        self.model =  DQN.load(self.dqn_path, self.device)
            
    def get_action(self, state, action_mask):
        epsilon = 0  # model is not exploring/learning so eps = 0 
        action = self.model.get_action(
                                    state, epsilon, action_mask
                                )[0] 
        return action
    
import numpy as np

class RandomOpponent:
    def __init__(self):
        pass  # No initialization needed for this example

    def get_action(self, state,  eps, action_mask):
        """Return a random valid action from the action_mask."""
        valid_actions = np.where(action_mask == 1)[0]  # Get indices where action_mask is 1
        return [np.random.choice(valid_actions) if valid_actions.size > 0 else None]  # Random valid choice or None if no valid actions

        
        