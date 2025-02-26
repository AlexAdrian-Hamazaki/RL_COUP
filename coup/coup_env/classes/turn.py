import textwrap
from .challenge import Challenge
from .block import Block
from .actions import NoAction

class Turn:
    def __init__(self, players, game):
        self._turn_order_index = 0
        self._current_base_player = players[self._turn_order_index]
        self._current_base_action_str = None 
        self._current_base_action_target_int = None
        self._current_base_action_challenger_int = None
        self._current_base_action_instance = NoAction()
        self._current_other_players = players[self._turn_order_index+1:] + players[:self._turn_order_index]
        self._turn_order = [self.current_base_player.name] + [player.name for player in self.current_other_players]
    
        self._next_action_type = 'claim_base_action'
        self._current_chooser = self.current_base_player
        self.game = game
    
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
    
    @property
    def current_base_action_challenger_int(self):
        return self._current_base_action_challenger_int

    @current_base_action_challenger_int.setter
    def current_base_action_challenger_int(self, value):
        self._current_base_action_challenger_int = value 
    

    # Getter and Setter for _current_base_action
    @property
    def current_base_action(self):
        return self._current_base_action

    @current_base_action.setter
    def current_base_action(self, value):
        self._current_base_action = value

    @property
    def current_base_action_str(self):
        return self._current_base_action_str

    @current_base_action_str.setter
    def current_base_action_str(self, value):
        self._current_base_action_str = value

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

    # Getter and Setter for _next_action_type
    @property
    def next_action_type(self):
        return self._next_action_type

    @next_action_type.setter
    def next_action_type(self, value):
        self._next_action_type = value  # Add validation if you expect specific types or values

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
    
    
    def claim_base_action(self, agent, action):
        """
        Player claims an action and turn object is updated
        Turn object is updated to update each player's claimed actions
        """
    
        self.current_base_player_int = agent
        self.current_base_player = self.game.players[agent]
        
        # get target player
        self.current_target = self.game.players[list(action.keys())[0]]
        # get str of current action
        self.current_base_action_str = list(action.values())[0]
        print(self.current_base_action_str)
    
        # get action instance
        if self.current_base_action_str in self.game.actions.ACTIONS_WITH_TARGET:
            action_instance = self.game.action_map.get(self.current_base_action_str)(self.current_target)
        else:
            action_instance = self.game.action_map.get(self.current_base_action_str)()
            
        if self.current_base_action_str == "assassinate":      
            action_instance.claim(self.current_base_player, self.game)
        
        
        # update turn to store claimed action instance
        self.current_base_action_instance = action_instance
        print(self.current_base_action_instance)
        
        
        # update this player's claimed actions # TEST THIS TODO
        if action_instance.card:
            self.current_base_player.add_claimed_card(action_instance.card)

        # Update game to update every player's knowledge of what every player is claiming
        
        self.game.update_knowledge() # TEST THIS TODO

    
    def exe_base_action(self) -> None:
        """
        Executes a base action made by the agent
        """
        # print(self.current_base_player)
        self.current_base_action_instance.do(self.current_base_player, self.game)   
        self.game.update_knowledge()
        # reset base action
        self.reset()
        # make it so next base player is next
        self.current_base_player_int, self.current_base_player  = self.get_next_base_player_int()
    
        return
    
    def get_next_base_player_int(self):
        game = self.game
        players = game.players
        n_players = game.n_players
        
        # uptick turn order index
        self.turn_order_index +=1
        
        # handle turn order
        if self.turn_order_index == n_players:
            self.turn_order_index = 0 # reset order
        
        return self.turn_order_index, players[self.turn_order_index]
        
    
    def exe_challenge(self, agent:int) -> None:
        """
        Executes a challenge made by the agent
        """
        game = self.game
        agent = game.players[agent]
        

        challenging_player = agent
        current_player = self.current_base_player
        current_action = self.current_base_action_instance
        
    
        #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
        self.challenge = Challenge(game=game, 
                              current_action = current_action,
                              current_player = current_player,
                              challenging_player = challenging_player)
        
        self.challenge.duel(challenging_player) 
        
        if self.challenge.status == True: # challenge succeeded
            current_player.lose_life(game) # current player loses life
            # turn the current base action into a PASS action because it just got challenged and is no longer valid
            
        elif self.challenge.status == False: # challenge failed
            # challenging player loses life
            challenging_player.lose_life(game)
            # action goes through. it will go through at the .exe action
        self.game.update_knowledge()
        return 
        

        # Do bot challenges
        for bot in self.game.turn.current_other_players:
            chal = bot.choose_to_challenge()
            if chal: 
                # run through challenge. Depending on outcome, this brings game object to next state
                # because if an action is challenged than it cant be blocked. and if it is not challengable its not blockable
                self.exe_challenge_action(bot, self.game)
                self.update_player_turns()

                return
            
        if not self.current_base_action_instance.blockable:
            #print("Action is not blockable, doing action")
            self.current_base_action_instance.do(self.current_base_player, self.game)   
            return
            
        # do bot blockings (only done if no challenges occur)
        for bot in self.game.bots:
            block = bot.choose_to_block()
            if block:
                self.exe_block(bot, self.game)
                # run through block Depending on outcome, this brings game object to next 
                self.update_player_turns()

                return
            


           

        
    def exe_challenge_action(self, challenging_player, game)-> None:
        """
        note. if action is actually not challengable, then this dones nothing
        
        Determines if the challenge is successful
        Executes the effect on the game of if the challenges was successful or not

        Returns:
            _type_: _description_
        """
        game = game
        current_action = self.current_base_action_instance
        current_player = self.current_base_player
        challenging_player = challenging_player
        
        #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
        challenge = Challenge(game=game, 
                              current_action = current_action,
                              current_player = current_player,
                              challenging_player = challenging_player)
        
        challenge.duel(challenging_player) 
  

    def exe_block(self, bot, game):
        """
        Note, if action is not blockable then nothing happens
        
        A block has happened
        Update game statuses. But because agent has to choose if they challenge or not, just returns the next step

        Args:
            bot (_type_): _description_
            game (_type_): _description_
        """
        game = game
        current_action = self.current_base_action_instance
        current_player = self.current_base_player
        blocking_player = bot

        # then players can choose to block
        # block will need information about whose is making the action, what the action is (if the action is blockable), and Will eventually need to check their knowledge
        game.next_action_type = 'challenge_action'
        game.challenging_player = bot.name

    
    # def step(self, game): 
    #     # upticks turn index, and updates game object accordingly
    #     self.update_player_turns(game)
    #     current_player = self.current_player
        
    #     # #print current players knowledge
    #     # #print(current_player.knowledge)
    #     # current player claims a certian action        
    #     self.claim_action(current_player, game)
    #     self.next_action_type="base_action"
        
    #     #### CHALLENGE BLOCK game, current_action, current_player: "Player", challenging_player:"Player")
    #     challenge = Challenge(game=game, 
    #                           current_action = self.current_action,
    #                           current_player = current_player,
    #                           challenging_player = None)
    #     self.challenge = challenge
        
    #     if challenge.is_action_challengable(): # if action can be challenged 
    #         self.next_action_type="challenge_action"
    #         challenge.challenge_round()
            
    #     ###### RESULT OF CHALLENGE
    #     if challenge.status == 1:
    #         return # nothing hpapens if the contest was successfull. Action does not go through.
    #         # handeling of lost life is handled in challenge_round
    #     elif challenge.status == 0:  # if challenge failed. This means that the player that challenged lost a life
    #         # and the action still goes through
    #         self.do(game)
    #         return
            
    
    #     #### BLOCKING OPTION
    #     elif challenge.status is None:  # no one challenged:

    #         # then players can choose to block
    #         # block will need information about whose is making the action, what the action is (if the action is blockable), and Will eventually need to check their knowledge
    #         block = Block(game=game,
    #                       turn=self)
    #         self.block = block
            
    #         if block.is_action_blockable():
    #             self.next_action_type="block_action"

    #             if self.current_action.name == "foreign_aid":
    #                 block.block_round()
    #             else:
    #                 block.block_duel() # blocking for asssinate/steal
                
    #         if block.status == 1: #block was a success so the active player will not do the action
    #             return
    #         elif block.status == 0: #block failed, action happens anyways
    #             self.do(game)
    #             return
    #         elif block.status == None: # no one chose to block
    #             self.do(game)
    #             return
    #         return
            
    def update_player_turns(self):
        """Update the turn knowledge of the game object
        Updates self.current_player
        self.other_players is an ordered list indicating next players to go    
        """
        game = self.game
        players = game.players
        n_players = game.n_players
        
        # uptick turn order index
        self.turn_order_index +=1
        # handle turn order
        if self.turn_order_index == n_players:
            self.turn_order_index = 0 # reset order
            
        self.current_base_player = players[self.turn_order_index]

        # #print(f"Player {self.current_player.name}'s turn")
        self.current_other_players  = players[self.turn_order_index+1:] + players[:self.turn_order_index]
        self.turn_order = [self.current_base_player.name] + [player.name for player in self.current_other_players]

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
            
    def reset_for_base_action(self):
        
        self.current_base_action = None 
        self.current_base_action_target_int = None
        self.current_base_action_challenger_int = None
        self.current_base_action_instance = None
        self.next_action_type = 'base_action'
        self.current_chooser = self.current_base_player
        
    def do(self, game):
        # does action
        self.current_action.do(self, game)
                
                
    def reset(self):
        self._current_base_action_str = None 
        self._current_base_action_target_int = None
        self._current_base_action_challenger_int = None
        self._current_base_action_instance = NoAction()
                