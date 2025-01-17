# Coup

"In the not too distant future, the government is run for profit by a new
“royal class” of multinational CEOs. Their greed and absolute control of
the economy has reduced all but a privileged few to lives of poverty and
desperation. Out of the oppressed masses rose The Resistance, an underground
organization focused on overthrowing these powerful rulers. The valiant
efforts of The Resistance have created discord, intrigue and weakness in the
political courts of the noveau royal, bringing the government to brink of
collapse. But for you, a powerful government official, this is your opportunity
to manipulate, bribe and bluff your way into absolute power. To be successful,
you must destroy the influence of your rivals and drive them into exile.
In these turbulent times there is only room for one to survive"

-- Coup rule book (https://www.qugs.org/rules/r131357.pdf)


### Game overview

There are 5 card types (and 3 of each)
    - Assassin
    - Captain
    - Duke
    - Ambassador
    - Contessa

Players are dealt 2 cards at the beginning of the game. 

Each card type has unique abilities that allow players to gain coins and kill other player's cards.

When a player has no more cards, they are no longer in the game.

The interesting part of the game is that players can bluff, using any ability regardless of what cards they have. 

Other players can call out their bluffs.

### Setup

With conda/mamba

```
mamba env create -f coup.yaml
mamba activate coup
```

```
./run.py
```


### A bit about the coding of this game

I coded this game to get some practice with OOP. Here is a map of my brainstorming of the rules of the game. This directly impacted how I chose my objects

[[image.png]]


I also coded this to eventually train a RL model on the game.
The progress (or completion depending on when this is read) can be found here [[]]

