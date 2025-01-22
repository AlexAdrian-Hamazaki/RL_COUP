import textwrap
from .challenge import Challenge
from .block import Block

class Turn:
    def __init__(self, players):
        self._turn_order_index = 0
        self._current_base_player = players[self.turn_order_index]
        self._current_base_action = None 
        self._current_base_action_target_int = None
        self._current_base_action_instance = None
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
        
    @property
    def current_base_action_target_int(self):
        return self._current_base_action_target_int

    @current_base_action_target_int.setter
    def current_base_action_target_int(self, value):
        self._current_base_action_target_int = value 
        
    @property
    def current_base_action_instance(self):
        return self._current_base_action_instance

    @current_base_action_instance.setter
    def current_base_action_instance(self, value):
        self._current_base_action_instance = value 
    
    #####################
    #####  CLASS FUNCTIONS
    ####################
    
    def step(self, action, action_map, game): # i need to think about how exactly this is going to work. I need to step to the next state of the game
        """Depending on the game state defined in turn,
        
        take 1 step to the next game state such that it is the current AGENTS TURN ONCE AGAIN
            this means I have to go through all of the bot's actions within one call of the step function
        
        Sometimes game may receive updated values
        
        States can be 3 fundamental types.
        A base action state moves to a challenge state. but its seen by a differnt agent. So I think I do need multiple agents here
        Challenge state then moves to base action state, or to a block action state depending on the end of that state
        Block action state always moves back to base action state but.

        Args:
            game (_type_): game object
        """
        self.game = game
        self.current_base_player = game.agent # I need to re-think this. This is always going to be player 0
        self.current_base_action = action_map[action[0]]
        self.current_base_action_target_int = action[1]
        
        if self.action_type == "base_action": #
            self.exe_base_action()
        
        elif self.action_type == "challenge_action":
            # if the agent is at a chellgen action. They may choose to challenge the action
                    #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
                    
            # TODO, modify this to just be challenge of current player
            # observation change is just that a player's claims have been updated and 
            # that the action may be targeting someone
            # and the action type. 
            
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
            
            # After the result of the challenge. If it was successfull then all bots can choose to block.

        elif self.action_type == "block_action":
            # if the action type is block. the agent can choose to block the action
            pass
        
        
        
        
        game.assess_game_win()
        
        
    def claim_action(self):
        """
        Player claims an action and turn object is updated
        Turn object is updated to update each player's claimed actions
        """
        current_player = self.current_base_player
        current_action_target_int = self.current_base_action_target_int
        action_str = self.current_base_action
        
        if action_str not in self.game.actions.ALLOWED_ACTIONS:
            raise ValueError("\t\tChose invalid action")
        
        action_instance = self.game.action_map.get(action_str)()
        
        # should never be hit. handeled by coup_env
        if not action_instance.check_coins(current_player.coins): # player does not have enough coins
            print("\t\tInsufficient coins")
            raise ValueError("Insufficient Coins")
        
        # update action instance with target if there is one
        if current_action_target_int > -1: # if there is a target 
            action_instance.target = current_action_target_int
    
        # update turn to store claimed action instance
        self.current_base_action_instance = action_instance
        
        # update this player's claimed actions
        current_player.add_claimed_card(action_instance.card)
        
        # Update game to update every player's knowledge of what every player is claiming
        self.game.update_knowledge()
    
    def exe_base_action(self):
        """
        Executes a base action made by the agent
        """
        
        self.claim_action() # adds current base action to players claimed cards.
        
        # Do bot challenges
        for bot in self.game.bots:
            chal = bot.choose_to_challenge()
            if chal: 
                # run through challenge. Depending on outcome, this brings game object to next
                self.exe_challenge(bot, self.game)
                return
            
        assert False
        # do bot blockings (only done if no challenges occur)
        for bot in self.game.bots:
            block = bot.choose_to_block()
            if block:
                self.exe_block(bot, self.game)
                # run through block Depending on outcome, this brings game object to next 
                return
        
        # If we reach here, no blocks or challenges were done so we can just do action
        self.current_base_action.do(self, self.game)            

        
    def exe_challenge(self, bot, game)-> None:
        """A challengs has happened
        
        Determines if the challenge is successful
        Executes the effect on the game of if the challenges was successful or not

        Returns:
            _type_: _description_
        """
        game = game
        current_action = self.current_base_action_instance
        current_player = self.current_base_player
        challenging_player = bot
        
        #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
        challenge = Challenge(game=game, 
                              current_action = current_action,
                              current_player = current_player,
                              challenging_player = challenging_player)
        
        if challenge.is_action_challengable(): # if action can be challenged 
            challenge.duel(bot) 
        else:
            print("Action is not challengable. doing action")
            current_action.do(current_player, game)
            return 

            
            
    def exe_block(self):
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
                