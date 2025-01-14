class Actions:
    # each person will have an action doing class?
    ALLOWED_ACTIONS = [
        "income", "foreign_aid", "coup", "tax", 
        "assassinate", "steal", "exchange"
    ]
    CONTESTABLE_ACTIONS = {'tax','assassinate','steal','exchange',
                           'block_assassinate','block_foreign_aid','block_steal'}
    
    ACTION_COST = {"coup":7, 'assassinate':3}
    
    ACTIONS_WITH_TARGET = {'coup', 'assassinate', 'steal'}
  
# Subclasses for each action
class Income(Actions):
    def __init__(self, name='income'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name
    
    def do(self, game):
        player = game.current_player
        player.take_coin(game)
        print(f"\tPlayer {player.name} takes income!")

class Foreign_Aid(Actions):
    def __init__(self, name='foreign_aid'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name    
    def do(self, game):
        player = game.current_player
        player.take_coin(game)
        player.take_coin(game)
        print(f"{player.name} takes foreign aid!")

class Coup(Actions):
    def __init__(self, name='coup'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        player = game.current_player
        other_player = game.target_player
        player.discard_coin(game, 7)  # Cost of coup
        other_player.lose_life(game)
        print(f"{player.name} performs a coup!")

class Tax(Actions):
    def __init__(self, name='Tax'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        player = game.current_player
        player.take_coin(game)
        print(f"{player.name} collects tax!")

class Assassinate(Actions):
    def __init__(self, name='Assassinate'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Assassination action performed!")

class Steal(Actions):
    def __init__(self, name='Steal'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Steal action performed!")

class Exchange(Actions):
    def __init__(self, name='Exchange'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Exchange action performed!")


# class Contest:
#     pass

    # @property
    # def action(self):
    #     return self._action
    
    # @action.setter
    # def action(self, action):
    #     """Setter to validate that action is in allowed actions."""
    #     if  action not in Actions.ALLOWED_ACTIONS:
    #         raise ValueError(f"Invalid action '{action}'. Allowed actions are {', '.join(Actions.ALLOWED_ACTIONS)}.")
    #     self._action = action