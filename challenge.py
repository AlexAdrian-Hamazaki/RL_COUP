
class Challenge:
    def __init__(self, game, current_action, current_player: "Player", challenging_player:"Player"):
        self._game = game
        self._current_action = current_action
        self._current_player = current_player
        self._challenging_player = challenging_player
        self._status = None #None if no challenge is done, 0 if challenge fails, 1 if succeeds
        
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
    @challenging_player.setter
    def challenging_player(self, challenging_player):
        self._challenging_player = challenging_player
    
    @property
    def status(self):
        return self._status
    @status.setter
    def status(self, status):
        self._status = status
    
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
        current_player = self.current_player
        game = self.game
        current_action = self.current_action
        success = False
        for card in current_player.cards:
            if current_action.name.lower() in card.REAL_ACTIONS: # if the action is allowed by the cards the current player has
                current_player.put_card_on_bottom(card, game)
                current_player.remove_claimed_action(current_action)
                game.deck.shuffle()
                current_player.draw_card(game)
                success = True # the challenge 
                break
        if not success:
            raise ValueError("Error: unable to reveal card but player should have it in hand")    
        
        #### Actions done by player whose challenge failed
        self.challenging_player.lose_life(self.game) # reveals this card, which handels it going in the dead pile as well    
        
    def challenge_succeeds(self):
        # if challenge succeds, active player needs to reveal a card of their choice
        self.current_player.lose_life(self.game) # reveals this card, which handels it going in the dead pile as well


    def is_action_challengable(self):
        return self.current_action.challengable 
    
            
    def challenge_round(self):
        other_players = self.game.turn.other_players
        for other_player in other_players:
            if other_player.check_challenge(self): # compares player knowledge to game state and asks if they want to challenge most recent action
                print(f"\tPlayer {other_player.name} contests action")
                self.challenging_player = other_player
                self.execute_challenge() # cards are checkd to see if contest is successfull
                break
                
    
    def execute_challenge(self):
        #other player contests the action of current player (in self.current_player)
        #first the current player reveals if they have the card required to do the action
        if self.can_rev_cc(): # if they have the card they claim this will be true
            self.challenge_fails()
            self.challenge_status = 'failed'
        else: # the current player does not have the card they claim to have the power for
            self.challenge_succeeds()
            self.challenge_status = 'successful'
            