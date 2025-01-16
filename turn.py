import textwrap
from challenge import Challenge
from block import Block

class Turn:
    def __init__(self):
        self._current_player = None
        self._turn_order_index = -1
        self._current_action = None 
        self._other_players = None
        
    def __repr__(self):
        # Indent each player properly by prepending "   " to each player's name
        players_list = "\n   ".join([str(player) for player in self.other_players])
        players_list = "   " + players_list  # Ensure the first player is indented too
        result= f"""
Current player: {str(self.current_player)}
Turn order after player:
{players_list}
"""
        return result

    # Getter and Setter for _current_player
    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    # Getter and Setter for _turn_order_index
    @property
    def turn_order_index(self):
        return self._turn_order_index

    @turn_order_index.setter
    def turn_order_index(self, value):
        self._turn_order_index = value

    # Getter and Setter for _current_action
    @property
    def current_action(self):
        return self._current_action

    @current_action.setter
    def current_action(self, value):
        self._current_action = value

    # Getter and Setter for _other_players
    @property
    def other_players(self):
        return self._other_players

    @other_players.setter
    def other_players(self, value):
        self._other_players = value
        
    #####################
    #####  CLASS FUNCTIONS
    ####################
    
    def next_turn(self, game): 
        # upticks turn index, and updates game object accordingly
        self.update_player_turns(game)
        
        # print message indicating status of game
        print(game)
        print(self)
        
        current_player = self.current_player
        
        # print current players knowledge
        print(current_player.knowledge)
        # current player claims a certian action        
        self.claim_action(current_player, game)
        
        #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
        challenge = Challenge(game=game, 
                              current_action = self.current_action,
                              current_player = current_player,
                              challenging_player = None)

        if challenge.is_action_challengable(): # if action can be challenged 
            challenge.challenge_round()
            print(f'Challenge status {challenge.status}')
            
        ###### RESULT OF CHALLENGE
        if challenge.status == 1:
            return # nothing hpapens if the contest was successfull. Action does not go through.
            # handeling of lost life is handled in challenge_round
        elif challenge.status == 0:  # if challenge failed. This means that the player that challenged lost a life
            # and the action still goes through
            self.do(game)
            return
            
    
        #### BLOCKING OPTION
        elif challenge.status is None:  # no one challenged:

            # then players can choose to block
            # block will need information about whose is making the action, what the action is (if the action is blockable), and Will eventually need to check their knowledge
            block = Block(game=game,
                          turn=self)
            
            if block.is_action_blockable():
                if self.current_action.name == "foreign_aid":
                    block.block_round()
                else:
                    block.block_duel() # blocking for asssinate/steal
                
            if block.status == 1: #block was a success so the active player will not do the action
                return
            elif block.status == 0: #block failed, action happens anyways
                self.do(game)
                return
            elif block.status == None: # no one chose to block
                self.do(game)
                return
            return
            


    def claim_action(self, player, game):
        """
        Player claims an action and turn object is updated
        Turn object is updated to update each player's claimed actions
        """
        # Create the prompt text
        prompt_text = f"""
        {"="*40}
        Select an action out of the following:
        {"-"*40}
        {', '.join(game.actions.ALLOWED_ACTIONS)}
        {"="*40}
        """
        
        # Use textwrap.dedent to clean up any unwanted indentation in the prompt
        # Then, add one tab at the beginning of the entire prompt string
        action_input = input(
            textwrap.dedent(prompt_text).replace("\n", "\n\t")  # Add tab before every new line
        )
        if action_input not in game.actions.ALLOWED_ACTIONS:
            print("\t\tChose invalid action")
            return self.claim_action(player, game)
        

        action_instance = game.action_map.get(action_input)()

        if not action_instance.check_coins(player.coins): # player does not have enough coins
            print("\t\tInsufficient coins")
            return self.claim_action(player, game)
        
        if action_instance.has_target: # if action has a target
            input_target_player = input(f"\n\tWhat player do you want to target?: {[player.name for player in self.other_players]}")
            try:

                if input_target_player in [str(player.name) for player in self.other_players]: # ensure valid player was targeted and get player object
                    action_instance.target_player = game.get_player_by_name(input_target_player) # set target in the action instance
                else:
                    raise ValueError
            except ValueError as e:
                print("\t\tInvalid player selection, try again")
                return self.claim_action(player, game)

        # update turn to store claimed action
        self.current_action = action_instance
        
        # update this player's claimed actions
        player.add_claimed_action(action_instance.name)
        # Update game to update every player's knowledge of what every player is claiming
        game.update_claims(player, self.other_players)

    def update_player_turns(self, game):
        """Update the turn knowledge of the game object
        Updates self.current_player
        self.other_players is an ordered list indicating next players to go    
        """
        players = game.players
        n_players = game.n_players
        
        # uptick turn order index
        self.turn_order_index +=1
        # handle turn order
        if self.turn_order_index == n_players:
            self.turn_order_index = 0 # reset order
        self.current_player = players[self.turn_order_index]
        # print(f"Player {self.current_player.name}'s turn")
        self.other_players  = players[self.turn_order_index+1:] + players[:self.turn_order_index]
        

    def update_after_death(self, i_dead):
        # Case when player who dies its their turn
        if i_dead == self.turn_order_index:
            # down tirk turn int so that when it gets upticked later its right
            self.turn_order_index-=1
        elif i_dead > self._turn_order_index: 
            # player did when it was not their turn.
            return 
        # case where dead player dies and is behind them in turn order # note that sthis ends up being the same case as case 1
        elif i_dead < self._turn_order_index:
            self._turn_order_index-=1
            
    def do(self, game):
        # does action
        self.current_action.do(self, game)
                