class Action:
    # each person will have an action doing class?
    ALLOWED_ACTIONS = {"income", "foreign_aid", "block_foreigh_aid", "coup",
                       "tax", "assassinate", "block_assassinate",
                       "steal", "block_steal", "exchange"}
    
    def __init__(self, game, action):
        self._game = game
        self._action = action
    
    @property
    def game(self):
        return self._game
    @property
    def action(self):
        return self._action
    @action.setter
    def action(self, action):
        """Setter to validate that action is in allowed actions."""
        if  action not in Action.ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Allowed actions are {', '.join(Action.ALLOWED_ACTIONS)}.")
        self._action = action

    
    @classmethod
    def income(cls):
        pass
    
    


class Contest(Actions):
    pass
    
    
        
