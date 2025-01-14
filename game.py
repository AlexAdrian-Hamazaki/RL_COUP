import numpy as np
from actions import Actions
from card import Assassin, Captain, Contessa, Ambassador, Duke
from player import Player

class Game:
    
    
    def __init__(self, n_players):
        self._n_players = n_players
        self._players = [Player(name=n) for n in range(n_players)]
        self._deck = Deck()    
        self._deck._shuffle() # shuffle deck
        self._bank = CoinBank()
        self._curr_turn = 0
        self._current_player = self._players[self._curr_turn]
        self._other_players = self._players[self._curr_turn+1] + self._players[:self._curr_turn]
        self._current_action = None 
        self._action_target = None
        
        self.on = True # true if game is live, False if game is over
        
        # setup game
        self._setup_deal()    
        self._setup_give_coins()    
        
        
        print(f"""
              Initializing game {self} 
              With players {self._players}
        """)
        
        # init action helper
        self.actions = Actions()
        
        
        
    def __repr__(self):
        return f"""
              Game of COUP
              Number of Players = {self._n_players}
              Current Deck Order (top to bottom) = {self._deck}
              Current Coins in Bank = {self._bank}
              Current player = {str(self.current_player)}
              Current claimed action = {str(self.current_action)}
              """
    
    @property
    def deck(self):
        return self._deck
        
    @property
    def bank(self):
        return self._bank
    
    @property
    def current_player(self):
        return self._current_player
    
    @property
    def other_players(self):
        return self._other_players
    
    @property
    def players(self):
        return self._players
    
    @property
    def current_action(self):
        return self._current_action
    
    @property
    def action_target(self):
        return self._action_target
    
    def _setup_deal(self): # initialize dealing of cards to players
        print("Dealing Setup Cards")
        n=0
        while n < 2 * self._n_players:
            for player in self._players:
                player.draw_card(self)
                n+=1        
    
    def _setup_give_coins(self): # initialize giving of cards to players
        print("Giving Coins to players")
        n = 0
        while n < 2 * self._n_players:
            for player in self._players:
                player.take_coin(self)
                n+=1
                
    def next_players_turn(self):
        # handle turn order
        if self._curr_turn == self._n_players-1:
            self._curr_turn = 0
        self._current_player = self.current_player
        
        # current player claims a certian action
        self.claim_action()
        # update claimed action knowledge
        self.update_claims()
        # all others players are allowed to contest
        self.round_of_contests()
        # execute an action
        self.do_action()
        
        self._curr_turn +=1 # uptick current turn index
        
    def claim_action(self):
        # prompt for action
        action_input = input(f"""
            {"="*40}
            Select an action out of the following:
            {"-"*40}
            {', '.join(self.actions.ALLOWED_ACTIONS)}
            {"="*40}
            """).strip().lower()
        
        if action_input not in self.actions.ALLOWED_ACTIONS:
            print("Chose invalid action")
            return self.claim_action()
        elif self.not_enough_couns(): # for the action you choose, you need to have enough coins
            print("Insufficient coins")
            return self.claim_action()
        else:
            # actual action function            
            action = getattr(self.actions, action_input, None)
            # update game to store claimed action
            self._curr_action = action
            
        
        
    def update_claims_add(self):
        # after each claimed action by a player, each other players gets a sequential chance to contest
        player =  self.current_player
        other_players = self.other_players
        
        # Claimed action currently being done by
        current_action = self.current_action
        
        # update this player's claimed actions
        player.add_claimed_action(current_action)
        # update the other player's knowledge if this players claimed actions
        [other_player.update_others_curr_ac(current_action, player) for other_player in other_players]
        
        
    def round_of_contests(self):
        for other_player in self.other_players:
            if self.check_contest(other_player):
                print(f"Player {other_player.name} contests action")
                
                # given game state, and thing just happened. a player can contest the proposed action of the previous player
                self.contest_action(other_player)
                
                break
            
    def check_contest(self, other_player):
        # game asks player if player wants to contest the proposed action of the current player
        contest = input(f"Does {other_player.name} want to contest current action: {self}")
        if contest:
            return True
        else:
            return False


    def do_action(self):
        # if no one contests the action it goes through
        action = self.current_action
        
        action(self)
        
        pass
    
    def contest_action(self, other_player):
        # other player contests the action of current player (in self.current_player)
        
        # first the current player reveals if they have the card required to do the action
        
            # if they do, 
                # other player reveals the card
                # current player places card in deck, shuffles, and draws 1 card
                # the card they they were claiming is removed from their claimed cards, and others knowledge of the current players
                # claimed cards are updated as well
            # if they do not
                # lying player reveals one card of their choice
                # updates the pool of shared card knowledge based on what is flipped up
                
        pass
    

        
                
class Deck():
    def __init__(self):
        dukes = [Duke() for _ in range(3)]
        contessas = [Contessa() for _ in range(3)]
        assassins = [Assassin() for _ in range(3)]
        captains = [Captain() for _ in range(3)]
        ambos = [Ambassador() for _ in range(3)]
        
        self._deck = np.array(dukes + contessas + assassins + captains + ambos)
    
    @property
    def deck(self):
        return self._deck
        
    def __repr__(self):
        return f"Deck State : {self._deck.tolist()}"
        
    def remove_top_card(self):# top card gets drawn be player
        self._deck = self._deck[1:]
    def _add_to_bottom(self):# player adds card to bottom
        pass
    def _shuffle(self):
        np.random.shuffle(self._deck)  # Shuffle in place, no need to assign it back
        print("Deck Shuffled")

class CoinBank():
    def __init__(self):
        self._n = 50      
    def __repr__(self):
        return str(self._n)
    
    def remove(self):
        self._n -=1
    def add(self):
        self._n +=1  
    
    @property
    def n(self):
        return self._n
    


        
        
