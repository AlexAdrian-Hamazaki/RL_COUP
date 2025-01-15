from player import Player
from card import Card
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
        player.take_coins(game, 1)
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
        player.take_coins(game, 2)
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
        # player.discard_coin(game, 7)  # Cost of coup
        other_player.lose_life(game)
        print(f"Player {player.name} performs a coup against {other_player.name}!")

class Tax(Actions):
    card = {"duke"}
    def __init__(self, name='tax'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        player = game.current_player
        player.take_coins(game,3)
        print(f"\tPlayer {player.name} collects tax!")

class Assassinate(Actions):
    card = {"assassin"}
    def __init__(self, name='assassinate'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Assassination action performed!")

class Steal(Actions):
    card = {"captain"}
    def __init__(self, name='steal'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Steal action performed!")

class Exchange(Actions):
    card = {"ambassador"}
    def __init__(self, name='exchange'):
        self._name = name        
    def __repr__(self):
        return self.name
    @property
    def name(self):
        return self._name  
    def do(self, game):
        print("Exchange action performed!")


class Challenge:
    def __init__(self, game, player:Player, challenging_player:Player):
        self._game = game
        self._current_action = game.current_action # in the future this will need to turn into "challenged action" when blocks are included
        self._current_player = player
        self._challenging_player = challenging_player
        
    
    @property
    def game(self):
        return self._game
        
    @property
    def current_action(self):
        return self._current_action
    @property
    def current_player(self):
        return self._current_player
    @property
    def challenging_player(self):
        return self._challenging_player
    
    def can_rev_cc(self)-> bool:
        """
        can player reveal their claimed card?
        returns true if the player has the card for the action they claim they do
        otherwise false
        """
        player_cards = self.current_player.cards
        action = self.current_action
        
        if any(action.name in c.REAL_ACTIONS for c in player_cards):
            print("\t\tChallenge failed")
            return True
        else:
            print("\t\tChallenge successful")
            return False
        
    def challenge_fails(self):
        # Actions for player whoo was challenged 
        success = False
        for card in self.current_player.cards:
            if self.current_action.name.lower() in card.REAL_ACTIONS:
                self.current_player.put_card_on_bottom(card, self.game)
                self.current_player.remove_claimed_action(self.current_action)
                success = True
                break
        if not success:
            raise ValueError("Error: unable to reveal card but player should have it in hand")    
        self.game.deck.shuffle()
        
        #### Actions done by player whose challenge failed
        self.challenging_player.lose_life(self.game) # reveals this card, which handels it going in the dead pile as well    
        
    def challenge_succeeds(self):
        # if challenge succeds, active player needs to reveal a card of their choice
        self.current_player.lose_life(self.game) # reveals this card, which handels it going in the dead pile as well

