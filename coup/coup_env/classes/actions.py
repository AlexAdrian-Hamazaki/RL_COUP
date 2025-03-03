from .player import Player
from .card import Card
import numpy as np
class Actions:
    # each person will have an action doing class?
    ALLOWED_ACTIONS = [
        "income", "foreign_aid", "coup", "tax", 
        "assassinate", "steal", "exchange"
    ]
    CHALLENGABLE_ACTIONS = {'tax','assassinate','steal','exchange',
                           'block_assassinate','block_foreign_aid','block_steal_cap',
                           "block_steal_amb"}
    
    ACTION_COST = {"coup":7, 'assassinate':3}
    
    BLOCKABLE_ACTIONS = {'foreign_aid','assassinate','steal'}
    
    ACTIONS_WITH_TARGET = {'coup', 'assassinate', 'steal'}
    
    CARDS_ACTIONS_MAP = {
        "income": None,
        "foreign_aid": None,
        "coup": None,
        "tax": "duke",
        "assassinate": "assassin",
        "steal": "captain",
        "exchange": "ambassador",
        "block_foreign_aid": "duke",
        "block_assassinate": "contessa",
        "block_steal_amb": "ambassador",
        "block_steal_cap": "captain"
    }
    
    
    
    card = None
    
    def __init__(self, target_player = None, name="pass"):
        self._name = name    
        self._challengable = self._name in self.CHALLENGABLE_ACTIONS
        self._blockable = self._name in self.BLOCKABLE_ACTIONS  
        self._cost = self.ACTION_COST.get(self._name, 0)
        self.has_target = None
        self._target_player = target_player 
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
        
    @property
    def target_player(self):
        return self._target_player
    @target_player.setter
    def target_player(self, target_player):
        self._target_player = target_player

    def do(self, player, game):
        # print("Passing!")
        pass
        
    def check_coins(self, coins):
        # returns true if you have enough coins for action
        # or action has no coin cost
        if self.cost > coins:
            return False
        else:
            return True #player has sufficient coins
  
# Subclasses for each action
class Income(Actions):
    card = None
    def __init__(self, name='income'):
        super().__init__(name=name)  # Initialize the parent class (Actions)
        
    def do(self, player, game):
        #print(f"\tPlayer {player.name} takes income!")
        player = player
        player.take_coins(game, 1)



class Foreign_Aid(Actions):
    card = None
    def __init__(self, name='foreign_aid'):
        super().__init__(name=name)  # Initialize the parent class (Actions)

    def do(self, player, game):
        #print(f"{player.name} takes foreign aid!")

        player = player
        player.take_coins(game, 2)


class ActionsWTarget(Actions):
    def __init__(self, target_player: int, name: str = 'ActionWTarget'):
        super().__init__(target_player=target_player, name=name) 
        self.has_target = True


class Coup(ActionsWTarget):
    card = None

    def __init__(self, target_player: int, name='coup'):
        super().__init__(target_player=target_player, name=name) 
        self.cost = 7
    def __repr__(self):
        return self.name

    def do(self, player, game):
        #print(f"Player {player.name} performs a coup!")
    
        player = player
        target_player = self.target_player
        if target_player is None:
            raise ValueError("Something went wrong in selecting target player")
        player.discard_coins(game, 7)  # Cost of coup
        target_player.lose_life(game)

class NoAction(Actions):
    card = ""
    def __init__(self, target_player=None, name="none"):
        super().__init__(target_player=target_player, name=name)


class Tax(Actions):
    card = "duke"
    def __init__(self, name='tax'):
        super().__init__(name=name)  # Initialize the parent class (Actions)
    
    def do(self, player, game):
        #print(f"\t\tPlayer {player.name} collects tax!")

        player = player
        player.take_coins(game,3)


class Assassinate(ActionsWTarget):
    card = "assassin"
    def __init__(self, target_player: int, name='assassinate'):
        super().__init__(target_player=target_player, name=name) 
        self.cost = 3
        
    def claim(self, player, game):
        assert self.check_coins(player.coins)
        player.discard_coins(game, 3)    
            
        
    def do(self, player, game):
        player = player
        target = self.target_player
        target.lose_life(game)

class Steal(ActionsWTarget):
    card = "captain"
    def __init__(self, target_player: int, name: str = 'steal'):
        super().__init__(target_player=target_player, name=name) 
    
    def do(self, player, game):
        #print("\t\tSteal action performed!")

        player = player
        target = self.target_player
        
        if target.coins >=2:
            target.discard_coins(game, 2)
            player.take_coins(game, 2)
        else:
            coins = target.coins
            target.discard_coins(game, coins)
            player.take_coins(game, coins)
            


class Exchange(Actions):
    card = "ambassador"
    def __init__(self, name='exchange'):
        super().__init__(name=name)  # Initialize the parent class (Actions)
        
    def do(self, player, game):
        #print("\t\tExchange action performed!")
        player = player
    
        player.draw_card(game)
        player.draw_card(game)
        
        self.select_bottom(player, game)
        self.select_bottom(player, game)
        
        
        

    def select_bottom(self, player, game): 

        cards = player.cards
        card = np.random.choice(cards)
        card_name = card.name
        
        lo_names = [card.name for card in cards]
        if not card_name in lo_names:
            #print("Invalid Selection")
            return self.select_bottom(player,game)
        
        card_index = lo_names.index(card_name)        
        player.put_card_on_bottom(cards[card_index], game)


class BlockAction(Actions):
    card = None
    def __init__(self, name="BlockAction"):
        super().__init__(name=name)
        self._blocks = None
        
    def __do(self):
        return
    
    @property
    def blocks(self):
        return self._blocks

class B_Foreign_Aid(BlockAction):
    card = 'duke'
    def __init__(self, name="block_foreign_aid"):
        super().__init__(name=name)
        self._blocks = "foreign_aid"
        
class B_Assassinate(BlockAction):
    card = 'contessa'
    def __init__(self, name="block_assassinate"):
        super().__init__(name=name)
        self._blocks = 'assassinate'
        
class B_Steal_Ambassador(BlockAction):
    card = 'ambassador'
    def __init__(self, name="block_steal_amb"):
        super().__init__(name=name)
        self._blocks = 'steal'
        
class B_Steal_Captain(BlockAction):
    card = 'captain'
    def __init__(self, name="block_steal_cap"):
        super().__init__(name=name)
        self._blocks = 'steal'
        

    
    