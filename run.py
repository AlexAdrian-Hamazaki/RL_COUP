#!/usr/bin/env python3

from game import Game
from actions import Actions
from card import Assassin, Captain, Contessa, Ambassador, Duke
from player import Player
import numpy as np


def main():
    print("Game Begin")

    game = Game(4)

    
    while game.on:
        
        game.next_turn()
    
    
if __name__ == "__main__":
    main()