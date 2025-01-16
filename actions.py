from player import Player
from card import Card
class Actions:
    # each person will have an action doing class?
    ALLOWED_ACTIONS = [
        "income", "foreign_aid", "coup", "tax", 
        "assassinate", "steal", "exchange"
    ]
    CHALLENGABLE_ACTIONS = {'tax','assassinate','steal','exchange',
                           'block_assassinate','block_foreign_aid','block_steal'}
    
    ACTION_COST = {"coup":7, 'assassinate':3}
    
    BLOCKABLE_ACTIONS = {'foreign_aid','assassinate','steal'}
    
    ACTIONS_WITH_TARGET = {'coup', 'assassinate', 'steal'}
    
    def __init__(self, name="Action"):
        self._name = name    
        self._challengable = self._name in self.CHALLENGABLE_ACTIONS
        self._blockable = self._name in self.BLOCKABLE_ACTIONS  
        self._cost = self.ACTION_COST.get(self._name, 0)
        self._has_target = self._name in self.ACTIONS_WITH_TARGET
        
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value  # Update the _name attribute

    @property
    def challengable(self):
        return self._challengable

    @challengable.setter
    def challengable(self, value):
        self._challengable = value  # Update the _challengable attribute

    @property
    def blockable(self):
        return self._blockable

    @blockable.setter
    def blockable(self, value):
        self._blockable = value  # Update the _blockable attribute

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value  # Update the _cost attribute

    @property
    def has_target(self):
        return self._has_target

    @has_target.setter
    def has_target(self, value):
        self._has_target = value  # Update the _has_target attribute



    def perform_action(self):
        print("Performing a general action!")
        
    def check_coins(self, coins):
        # returns true if you have enough coins for action
        # or action has no coin cost
        if self.cost > coins:
            return False
        else:
            return True #player has sufficient coins
  
# Subclasses for each action
class Income(Actions):
    def __init__(self, name='income'):
        super().__init__(name)  # Initialize the parent class (Actions)
        
    def do(self, turn, game):
        player = turn.current_player
        player.take_coins(game, 1)
        print(f"\tPlayer {player.name} takes income!")


class Foreign_Aid(Actions):
    def __init__(self, name='foreign_aid'):
        super().__init__(name)  # Initialize the parent class (Actions)

    def do(self, turn, game):
        player = turn.current_player
        player.take_coins(game, 2)
        print(f"{player.name} takes foreign aid!")


class ActionsWTarget(Actions):
    def __init__(self, name='ActionWTarget'):
        super().__init__(name)
        self._target_player = None
        
    @property
    def target_player(self):
        return self._target_player
    target_player.setter
    def target_player(self,target_player):
        self._target_player=target_player
    

class Coup(ActionsWTarget):
    def __init__(self, name='coup'):
        super().__init__(name)  # Initialize the parent class (Actions)
    def __repr__(self):
        return self.name

    def do(self, turn, game):
        player = turn.current_player
        target_player = self.target_player
        if target_player is None:
            raise ValueError("Something went wrong in selecting target player")
        player.discard_coin(game, 7)  # Cost of coup
        target_player.lose_life(game)
        print(f"Player {player.name} performs a coup against {target_player.name}!")


class Tax(Actions):
    card = {"duke"}
    def __init__(self, name='tax'):
        super().__init__(name)  # Initialize the parent class (Actions)
    
    def do(self, turn, game):

        player = turn.current_player
        player.take_coins(game,3)
        print(f"\tPlayer {player.name} collects tax!")


class Assassinate(ActionsWTarget):
    card = {"assassin"}
    def __init__(self, name='assassinate'):
        super().__init__(name)  # Initialize the parent class (Actions)

    def do(self, turn, game):
        print("Assassination action performed!")

class Steal(ActionsWTarget):
    card = {"captain"}
    def __init__(self, name='steal'):
        super().__init__(name)  # Initialize the parent class (Actions)
    
    def do(self, turn, game):
        print("Steal action performed!")


class Exchange(Actions):
    card = {"ambassador"}
    def __init__(self, name='exchange'):
        super().__init__(name)  # Initialize the parent class (Actions)
        
    def do(self, turn, game):
        print("Exchange action performed!")



class BlockAction(Actions):
    card = {}
    def __init__(self, name="BlockAction"):
        super().__init__(name)
        self._blocks = None
        
    def __do(self):
        return
    
    @property
    def blocks(self):
        return self._blocks

class B_Foreign_Aid(BlockAction):
    card = {'duke'}
    def __init__(self, name="block_foreign_aid"):
        super().__init__(name)
        self._blocks = "foreign_aid"
        
class B_Assassinate(BlockAction):
    card = {"contessa"}
    def __init__(self, name="block_assassinate"):
        super().__init__(name)
        self._blocks = 'assassinate'
        
class B_Steal(BlockAction):
    card = {'captain','ambassador'}
    def __init__(self, name="block_steal"):
        super().__init__(name)
        self._blocks = 'steal'
        

    
    