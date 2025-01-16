
class Knowledge:
    """
    Represents a player's knowledge of the game state
    Knowledge of what is in deck
    knowledge of what is in cards
    knowledge of what other players are claiming
    """
    def __init__(self, deck_knowledge=None, cards=None, other_player_claims=None,
                    revealed_knowledge=None):
        # Initialize lists/dicts only if the arguments are None
        self._deck_knowledge = deck_knowledge if deck_knowledge is not None else []
        self._cards = cards if cards is not None else []
        self._other_player_claims = other_player_claims if other_player_claims is not None else {}
        self._revealed_knowledge = revealed_knowledge if revealed_knowledge is not None else []

        
    def __repr__(self):
        result = f"""~~~~~~~ Player Knowledge ~~~~~~~
Deck Knowledge: {self.deck_knowledge} 
Cards Knowledge: {self._cards}
Other Claims: {self._other_player_claims}
Revealed Cards: {self._revealed_knowledge}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
        return result
        
        # Getter for deck_knowledge
    @property
    def deck_knowledge(self):
        return self._deck_knowledge

    # Setter for deck_knowledge
    @deck_knowledge.setter
    def deck_knowledge(self, value):
        # Optional: Add validation or processing logic before setting
        self._deck_knowledge = value

    # Getter for cards
    @property
    def cards(self):
        return self._cards

    # Setter for cards
    @cards.setter
    def cards(self, value):
        # Optional: Add validation or processing logic before setting
        self._cards = value

    # Getter for other_player_claims
    @property
    def other_player_claims(self):
        return self._other_player_claims

    # Setter for other_player_claims
    @other_player_claims.setter
    def other_player_claims(self, value):
        # Optional: Add validation or processing logic before setting
        self._other_player_claims = value
        
    # Getter for revealed_knowledge
    @property
    def revealed_knowledge(self):
        return self._revealed_knowledge

    # Setter for revealed_knowledge
    @revealed_knowledge.setter
    def revealed_knowledge(self, value):
        self._revealed_knowledge = value
        
    # Add a card to deck_knowledge
    def add_to_deck_knowledge(self, card):
        if card.name.lower() not in self._deck_knowledge:
            self.deck_knowledge.append(card.name.lower())

    # Remove a card from deck_knowledge
    def reset_deck_knowledge(self): # May be wrong because there is still knowledge after a shuffle but model will hopefully learn
        self.deck_knowledge = []

    # Add a card to cards
    def add_to_cards(self, card):
        self.cards.append(card.name.lower())

    # Remove a card from cards
    def remove_from_cards(self, action_str):
        if action_str.lower() in self.cards:
            print(self.cards)
            self.cards.remove(action_str.lower())
            print(self.cards)

    # Add a claim for a player in other_player_claims (adding a new claim or adding to an existing claim)
    def add_to_other_player_claims(self, player, claim):
        if player not in self._other_player_claims:
            self._other_player_claims[player] = set()
        self._other_player_claims[player].add(claim)

    # Remove a claim for a player in other_player_claims
    def remove_from_other_player_claims(self, player, claim):
        if player in self._other_player_claims and claim in self._other_player_claims[player]:
            self._other_player_claims[player].remove(claim)

    # Add a card to revealed_knowledge
    def add_to_revealed_knowledge(self, card):
        self.revealed_knowledge.append(card.name.lower())
        
    def update_other_p_c_action(self, other_player): 
        """
        updates the current player's knowledge
        knowledge of what the other player
        is claiming
        """
        other_player_cc = other_player.claimed_cards
        
        player_cc_knowledge_dic = self.other_player_claims
        
        player_cc_knowledge_dic[other_player.name] = other_player_cc
                
            
        # dic_other_claimed_cards = self.others_claimed_actions # keys are playernames and values is a set of their claimed cards
        # current_players_claimed_cards = dic_other_claimed_cards[player.name]
        # current_players_claimed_cards.append(action)
        # # update knowledge of players cards
        # dic_other_claimed_cards[player.name] = current_players_claimed_cards
        # self.others_claimed_actions = dic_other_claimed_cards