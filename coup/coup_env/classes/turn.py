import textwrap
from .challenge import Challenge
from .block import Block

class Turn:
    def __init__(self, players):
        self._turn_order_index = 0
        self._current_base_player = players[self.turn_order_index]
        self._current_base_action = None 
        self._current_other_players = players[self.turn_order_index+1:] + players[:self.turn_order_index]
        self._turn_order = [self.current_base_player.name] + [player.name for player in self.current_other_players]
    
        self._action_type = 'base_action'
        self._current_chooser = self.current_base_player
    
    # Getter and Setter for _turn_order_index
    @property
    def turn_order_index(self):
        return self._turn_order_index

    @turn_order_index.setter
    def turn_order_index(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("turn_order_index must be a non-negative integer")
        self._turn_order_index = value

    # Getter and Setter for _current_base_player
    @property
    def current_base_player(self):
        return self._current_base_player

    @current_base_player.setter
    def current_base_player(self, value):
        self._current_base_player = value  # Add validation if needed (e.g., check if `value` is in players)

    # Getter and Setter for _current_base_action
    @property
    def current_base_action(self):
        return self._current_base_action

    @current_base_action.setter
    def current_base_action(self, value):
        self._current_base_action = value

    # Getter and Setter for _current_other_players
    @property
    def current_other_players(self):
        return self._current_other_players

    @current_other_players.setter
    def current_other_players(self, value):
        if not isinstance(value, list):
            raise ValueError("current_other_players must be a list")
        self._current_other_players = value

    # Getter and Setter for _turn_order
    @property
    def turn_order(self):
        return self._turn_order

    @turn_order.setter
    def turn_order(self, value):
        if not isinstance(value, list):
            raise ValueError("turn_order must be a list")
        self._turn_order = value

    # Getter and Setter for _action_type
    @property
    def action_type(self):
        return self._action_type

    @action_type.setter
    def action_type(self, value):
        self._action_type = value  # Add validation if you expect specific types or values

    # Getter and Setter for _current_chooser
    @property
    def current_chooser(self):
        return self._current_chooser

    @current_chooser.setter
    def current_chooser(self, value):
        self._current_chooser = value 
        
    
    #####################
    #####  CLASS FUNCTIONS
    ####################
    
    def step(self, game):
        """Depending on the game state defined in turn,
        take 1 step to the next game state
        
        Sometimes game may receive updated values

        Args:
            game (_type_): game object
        """
        pass
        
    
    def next_turn(self, game): 
        # upticks turn index, and updates game object accordingly
        self.update_player_turns(game)

        
        current_player = self.current_player
        
        # print current players knowledge
        print(current_player.knowledge)
        # current player claims a certian action        
        self.claim_action(current_player, game)
        self.action_type="base_action"
        
        #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
        challenge = Challenge(game=game, 
                              current_action = self.current_action,
                              current_player = current_player,
                              challenging_player = None)
        self.challenge = challenge
        
        if challenge.is_action_challengable(): # if action can be challenged 
            self.action_type="challenge_action"
            challenge.challenge_round()
            
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
            self.block = block
            
            if block.is_action_blockable():
                self.action_type="block_action"

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
        player.add_claimed_card(action_instance.card)
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
                