from agilerl.algorithms.dqn import DQN


class Opponent:
    """Coup opponent to train and/or evaluate against.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param difficulty: Difficulty level of opponent, 'random', 'weak' or 'strong'
    :type difficulty: str
    """
    def __init__(self, difficulty, device):
        self.difficulty = difficulty
        self.device = device
        
        self.load_model()
        
    def __str__(self):
        return f"Model difficulty {self.difficulty} from path {self.path}"
    
        
    def load_model(self):
        if self.difficulty == "random":
            self.path = "models/DQN/lesson1_trained_agent.pt"
            self.model =  DQN.load(self.path, self.device)
            
        elif self.difficulty == "weak":
            self.path = "models/DQN/lesson2_trained_agent.pt"
            self.model =  DQN.load(self.path, self.device)
            
        elif self.difficulty == self.strong_rule_based_opponent:
            self.path = "models/DQN/lesson3_trained_agent.pt"
            self.model =  DQN.load(self.path, self.device)
            
    def get_action(self, state, action_mask):
        epsilon = 0  # model is not exploring/learning so eps = 0 
        action = self.model.get_action(
                                    state, epsilon, action_mask
                                )[0] 
        return action
    
    