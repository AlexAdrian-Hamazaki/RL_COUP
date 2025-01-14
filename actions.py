class Actions:
    # each person will have an action doing class?
    ALLOWED_ACTIONS = [
        "income", "foreign_aid", "coup", "tax", 
        "assassinate", "steal", "exchange"
    ]
    
    def income(self, game):
        player = game.current_player
        player.take_coin(game)
    
    def foreign_aid(self, game):
        player = game.current_player
        player.take_coin(game)
        player.take_coin(game)
    
    def coup(self, game):
        player = game.current_player
        player.remove
    
    def tax(self, game):
        pass
    
    def assassinate(self, game):
        pass
    
    def steal(self, game):
        pass
    
    def exchange(self, game):
        pass
    

class Contest:
    pass

    # @property
    # def action(self):
    #     return self._action
    
    # @action.setter
    # def action(self, action):
    #     """Setter to validate that action is in allowed actions."""
    #     if  action not in Actions.ALLOWED_ACTIONS:
    #         raise ValueError(f"Invalid action '{action}'. Allowed actions are {', '.join(Actions.ALLOWED_ACTIONS)}.")
    #     self._action = action