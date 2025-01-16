from challenge import Challenge
from actions import B_Steal, B_Foreign_Aid, B_Assassinate
class Block:
    
    # keys are the .name of blockable actions
    BLOCK_ACTION_MAP = {"foreign_aid":B_Foreign_Aid,
                        "assassinate":B_Assassinate,
                        "steal":B_Steal}
    
    def __init__(self, game, turn):
        self._game = game
        self._turn = turn
        self._status = None
        
    @property
    def game(self):
        """Getter for _game"""
        return self._game

    @game.setter
    def game(self, game):
        """Setter for _game"""
        self._game = game

    @property
    def turn(self):
        """Getter for _turn"""
        return self._turn

    @turn.setter
    def turn(self, turn):
        """Setter for _turn"""
        self._turn = turn
        
    @property
    def status(self):
        """Getter for _status"""
        return self._status

    @status.setter
    def status(self, status):
        """Setter for _status"""
        self._status = status
        
    
    def is_action_blockable(self):
        return self.turn.current_action.blockable

    def block_round(self):
        other_players = self.game.turn.other_players
        for other_player in other_players:
            if other_player.check_block(self): # compares player knowledge to game state and asks if they want to block most recent action #TODO
                print(f"\tPlayer {other_player.name} blocks action")
                
                block_action = self.BLOCK_ACTION_MAP[self.turn.current_action.name]() # get an instance of the blocking action required
                
                #make a new challenge where the current player is the player that just blocked
                challenge = Challenge(game = self.game, 
                                      current_action=block_action, 
                                      current_player=other_player,
                                      challenging_player=None)
                
                if challenge.is_action_challengable(): # all other players  can challenge 
                    challenge.challenge_round()
                    
                ###### RESULT OF CHALLENGE
                if challenge.status == 1:
                    self.status = 0 # block action got challenged successfully. Block action does not go through
                    print("\t\tBlock Success, action does not go through")
                    return 
                elif challenge.status == 0:  # Block action was challenged, challenge failed. Block goes through
                    print("\t\tBlock Failed, action goes through")
                    self.status = 1 #
                    return 
                    # and the action still goes through
                elif challenge.status is None: #No one challenged the block. Block goes through
                    print("\t\tBlock Success, action does not go through")
                    self.status = 1
                    return 
            