# import numpy as np

class Game:
    def init_players():
        player = Player()
        pass
    pass

# class Game:
#     def __init__(self, n_players):
#         self.n_players = n_players
#         self.players = [Player("P{i}") for i in range(self.n_players)] # list of Player classes
#         self.turn_order = np.random.shuffle(self.players)
        
#         self.game_over = False
#         self.player_turns = np.zeros(self.n_players)
        
#         # stateSpace. How we will define the state of the game
#         self.stateSpace = [self.turn_order, self.deck, *self.me, *self.other_players]
#         self.stateSpacePlus = None #TODO
        
#         # possible actions to take
#         self.possibleActions = []
        
#         # actionSpace is the 
        
#     def __repr__(self):
#         print(f"""
#               Game of COUP
#               Players: {self.players},
#               Cards: {self.cards}
#               Coins: {self.coins}
#               """)
#     def draw_card(self, deck):
#         pass
#     def reveal_card(self, hand):
#         pass
#     def discard_card(self, deck):
#         pass

#     def set_turn_order(self):
#         self.turn_order = np.random.shuffle(self.n_players)
        


# class Player:
#     """
#     A person playing COUP
#     """
#     def __init__(self, name):
#         self.name = name
#         self.cards = []
#         self.coins = 2  # Start with 2 coins
#         self.is_eliminated = False

#     def __repr__(self):
#         return f"Player({self.name}, Cards: {self.cards}, Coins: {self.coins})"

#     def change_coin_amount(self, amount:int):
#         """add or decrease the maount of coins the player has

#         Args:
#             amount (int): + or - amount to change
#         """
#         self.coins += amount

#     def lose_card(self):
#         if self.cards:
#             return self.cards.pop(random.randint(0, len(self.cards) - 1))
#         return None

#     def perform_action(self, action, game):
#         """Perform an action using a card."""
#         action.action(self, game)

#     def has_card(self, card_name):
#         """Check if the player has a card with the given name."""
#         return any(card.name == card_name for card in self.cards)

#     def is_alive(self):
#         """Check if the player is still in the game."""
#         return not self.is_eliminated

#     def exchange_cards(self, game):
#         """Exchange cards with the deck."""
#         if len(self.cards) == 2:
#             # Player discards one card and draws two new ones
#             game.exchange_cards(self)
#         else:
#             raise ValueError("Player must have two cards to exchange.")

#     def perform_coup(self, target_player, game):
#         """Perform a Coup action."""
#         if self.coins >= 7:
#             self.coins -= 7
#             game.coup(target_player)
#         else:
#             raise ValueError("Not enough coins for Coup action.")
