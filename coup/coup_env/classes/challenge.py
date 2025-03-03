class Challenge:
    def __init__(self, game, current_base_action, current_base_player: "Player"):
        self._game = game
        self._current_base_action = current_base_action
        self._current_base_player = current_base_player
        self._status = None # None if no challenge is done, False if challenge fails, True if succeeds
        
    @property
    def game(self):
        return self._game
        
    @property
    def current_base_action(self):
        return self._current_base_action
    
    @property
    def current_base_player(self):
        return self._current_base_player
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status):
        self._status = status
    
    def execute_challenge(self) -> bool:
        """
        can player reveal their claimed card?
        returns true if the player DOES NOT have the card for the action they claim they do -> meaning the challenge was a success
        otherwise return false indicating that the challenge was NOT a success
        """
        player_cards = self.current_base_player.cards
        action = self.current_base_action
        

        lo_dooable_actions = [action for card in player_cards for action in card.REAL_ACTIONS]
        # if this is true, the player HAS the card they claim, meaning the challenge FAILS
        player_has_card = action.name in lo_dooable_actions 
        if player_has_card: # if player has card, challenge fails
            self.status = False # challenge fails, player has card, Challenging player must lose card
        else:
            self.status = True  # challenge succeeds, player has card, Challenged player must lose card


    def is_action_challengable(self):
        return self.current_base_action.challengable 
    

    def get_other_players(self):
        players = self.game.players
        current_base_player = self.current_base_player
        lo_names = [player.name for player in players]
        
        player_int = lo_names.index(current_base_player.name)
    
        other_players = players[player_int+1:] + players[:player_int]

        return other_players