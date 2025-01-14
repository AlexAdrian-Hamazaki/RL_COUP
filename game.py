import numpy as np
from actions import Actions, Income, Foreign_Aid, Coup, Tax, Assassinate, Steal, Exchange
from card import Assassin, Captain, Contessa, Ambassador, Duke
from player import Player

class Game:
    
    
    def __init__(self, n_players):
        self._n_players = n_players
        self._players = [Player(n) for n in range(n_players)]
        self._deck = Deck()    
        self._deck._shuffle() # shuffle deck
        self._bank = CoinBank()
        self._curr_turn = 0
        self._current_player = self._players[self._curr_turn]
        self._other_players = self._players[self._curr_turn+1:] + self._players[:self._curr_turn]
        self._current_action = None 
        self._action_target = None
        self._contest_status = None
        
        self.on = True # true if game is live, False if game is over
        
        # setup game
        self._setup_deal()    
        self._setup_give_coins()
        [self._setup_other_claimed_card_dicts(player) for player in self.players]    
        # setup actions, which will hold all possible actions
        self.actions = Actions()
        self._action_map =  {"income":Income(),
                            "foreign_aid":Foreign_Aid(),
                            "coup":Coup(),
                            "tax":Tax(),
                            "assassinate":Assassinate(),
                            "steal":Steal(),
                            "exchange":Exchange()}
        
    
        print(f"""
              Initializing game {self} 
              With players {self._players}
        """)
        

        
        
        
    def __repr__(self):
        return f"""
              Game of COUP
              Number of Players = {self._n_players}
              Current Deck Order (top to bottom) = {self._deck}
              Current Coins in Bank = {self._bank}
              Current player = {str(self.current_player)}
              Current claimed action = {str(self.current_action)}
              Turn order after player: {str(self._other_players)}
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
    @current_player.setter
    def current_player(self, player):
        self._current_player = player
    
    @property
    def other_players(self):
        return self._other_players
    @other_players.setter
    def other_players(self, other_players):
        self._other_players = other_players
    
    @property
    def players(self):
        return self._players
    
    @property
    def current_action(self):
        return self._current_action
    
    @property
    def action_target(self):
        return self._action_target
    
    @property
    def contest_status(self):
        return self._contest_status
    @contest_status.setter
    def contest_status(self, status):
        if status not in {'failed', 'successful', None}:
            raise ValueError("invalid contest_status set")
        self._contest_status = status
    
    
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
                
    def _setup_other_claimed_card_dicts(self, player):
        player_int = player.name
        players = self.players
        other_players = players[player_int+1:] + players[:player_int]
        other_player_names = [ply.name for ply in other_players]
        player.others_claimed_cards =  {name : [] for name in other_player_names} # init dict for each other player to hold their claimed cards
    
    def update_player_turns(self):
        # handle turn order
        if self._curr_turn == self._n_players:
            self._curr_turn = 0
        self.current_player = self.players[self._curr_turn]
        print(f"Player {self.current_player.name}'s turn")
        
        self.other_players  = self._players[self._curr_turn+1:] + self._players[:self._curr_turn]
        
    
    def next_players_turn(self):
        self.update_player_turns()
        # current player claims a certian action
        self.claim_action()
        # update claimed action knowledge
        self.update_claims()
        
        if self.current_action in self.actions.CONTESTABLE_ACTIONS:
            # all others players are allowed to contest if the action is one that is contestable
            self.round_of_contests() # contest status is updated to see what happens to the action.
            # round of contest handels the actual checking to see who wins the contesting.
        
        if self._contest_status == 'success':
            pass # nothing hpapens if the contest was successfull. Action does not go through
        elif self._contest_status == "failed":
            # execute an action
            self.do_action()
        else:
            self.do_action()
        
        # reset contest status to None
        self.contest_status = None
        
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
        elif not self.check_coins(action_input): # for the action you choose, you need to have enough coins
            print("Insufficient coins")
            return self.claim_action()
        else:
            action_instance = self._action_map[action_input]
            # update game to store claimed action
            self._current_action = action_instance
            
    def check_coins(self, action_input):
        # returns true if you have enough coins for action
        # or action has no coin cost
        action_cost =self.actions.ACTION_COST.get(action_input)
        if not action_cost:
            return True 
        elif action_cost>self.current_player.coins:
            return False
        else:
            return True
        
            
    def update_claims(self):
        # after each claimed action by a player, each other players gets a sequential chance to contest
        player =  self.current_player
        other_players = self.other_players
                
        # Claimed action currently being done by
        current_action = self.current_action
        
        # update this player's claimed actions
        player.add_claimed_card(current_action)
        # update the other player's knowledge if this players claimed actions
        [other_player.update_others_curr_ac(current_action, player) for other_player in other_players]
        
        
    def round_of_contests(self):
        for other_player in self.other_players:
            if self.want_contest(other_player): # game asks if other player wants to contest prev action
                print(f"Player {other_player.name} contests action")
                contest_status = self._check_contest(other_player) # cards are checkd to see if contest is successfull
                self.contest_status = contest_status
                break # no other players need to be checked
            
    def want_contest(self, other_player):
        # game asks player if player wants to contest the proposed action of the current player
        contest = input(f"Does {other_player.name} want to contest current action: (y/n)")
        if contest=="y":
            return True
        elif contest=="n":
            return False
        else:
            print("enter valid option (y/n)")
            return self.want_contest(other_player)
        
    def check_contest(self, other_player):
        #other player contests the action of current player (in self.current_player)
        
        #first the current player reveals if they have the card required to do the action
        if self.current_player.can_rev_cc(): # if they have the card they claim
            
            #### Actions done by current player
            # reveal claimed card
            self.current_player.rev_claim_card() #
            # put in deck, shufffle and draw
            self.current_player.put_card_on_bottom()
            self.deck._shuffle()
            self.current_player.draw_card() 
            self.current_player.remove_claimed_card() # remove it as a card person is claiming because its shuffled in deck
            
            #### Actions done by player whose guess was wrong
            choice_card = other_player.rev_choice_card() # reveals this card, which handels it going in the dead pile as well
            other_player.remove_choice_card(choice_card)
            return 'failed'
            
        else: # the current player does not have the card they claim to have the power for
            choice_card = other_player.rev_choice_card() # reveals this card, which handels it going in the dead pile as well
            other_player.remove_choice_card(choice_card)
            return 'success'

    def do_action(self):
        # if no one contests the action it goes through
        action = self.current_action
        action.do(self)
        

                

    

        
                
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
    


        
        
