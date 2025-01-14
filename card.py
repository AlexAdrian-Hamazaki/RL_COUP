

class Card:
    # Sets of allowed things
    ALLOWED_STATES = {"deck", "face_up", "face_down"}  
    ALLOWED_NAMES = {"assassin", "captain", "ambassador, contessa, duke"}


    def __init__(self, name):
        self._name = name
        self._state = 'deck'
    
    @property # this is pretty much a gtter
    def state(self):
        return self._state
    @state.setter # this is a setter, lastly there is deleter
    def state(self, state:str):
        if state not in Card.ALLOWED_STATES:
            raise ValueError(f"Invalid state '{state}'. Allowed states are {', '.join(Card.ALLOWED_STATES)}.")
        self._state = state
        
    @property
    def name(self):
        #note: name has no _ because name is what is public, but self._name is private, only accessible via THIS public method
        return self._name
    @name.setter
    def name(self, name:str):
        if name not in Card.ALLOWED_NAMES:
            raise ValueError(f"Invalid cardname '{name}'. Allowed cards are {', '.join(Card.ALLOWED_NAMES)}.")
        
    def change_state(self, state):
        self._state = state
        

class Assassin(Card):
    REAL_ACTIONS = {"assassinate"}
    
    def __init__(self):
        self._name = "Assassin"
        self._state = 'deck'
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name    
        
    def assassinate(self, game, myplayer, otherplayer):
        # add coins to gamebank (in game)
        # remove coins from myplayer
        # Attempt to kill otherplayer
        pass
    

class Ambassador(Card):
    REAL_ACTIONS = {"exchange", 'block_steal'}
    
    def __init__(self):
        self._name = "Ambassador"
        self._state = 'deck'
        
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name
        
    def exchange(self, game, myplayer):
        pass
        

class Contessa(Card):
    REAL_ACTIONS = {"block_assassinate"}
    
    def __init__(self):
        self._name = "Contessa"
        self._state = 'deck'
        
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name
        
        
    def block_assassinate(self, game, myplayer, otherplayer):
        pass
        

class Duke(Card):
    REAL_ACTIONS = {"tax", 'block_foreign_aid'}
    
    def __init__(self):
        self._name = "Duke"
        self._state = 'deck'
        
        
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name
    
    def tax(self, game, myplayer):
        pass
        

class Captain(Card):
    REAL_ACTIONS = {"steal", "block_steal"}
    
    def __init__(self):
        self._name = "Captain"
        self._state = 'deck'
        
    def __repr__(self):
        return self.name
    
    @property
    def name(self):
        return self._name
        
        
        
    def steal(self, game, myplayer, otherplayer):

        pass

        
    
    
    

    
    
    
    
        
# class CardState(Card):
#     allowed_states = {"deck", "face_up", "face_down"}    
    
#     def __init__(self, state="deck"):
#         # Use the setter method to set the initial state
#         self.state = state

#     @property
#     def state(self):
#         # Getter method to retrieve the current state
#         return self._state
    
#     @state.setter
#     def state(self, value):
#         # Setter method to validate and set the state
#         if value not in CardState.ALLOWED_STATES:
#             raise ValueError(f"Invalid state '{value}'. Allowed states are {', '.join(CardState.ALLOWED_STATES)}.")
#         self._state = value
