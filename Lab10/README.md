# LAB 10
## Problem description
Use reinforcement learning to devise a tic-tac-toe player.

## My solution
I tried a very simple solution: every player stores all the states of a match. In training function, if the player wins I give 1 point to all states and 1 extra point to the final state because it is the state that gives the win to the player; viceversa i give -1 point to all states and -1 extra point to the final state. No points if the match ends in a draw.
When the smart player play a game, starting from the current state and from the list of all possible moves, it builds all the possible states that it can reach and check the values of them from the rewarded states list created during training. Then it simply choose the state with the maximum value and do the move in order to reach the chosen state.
After 50.000 matches played between 2 random players in order to train them, I played other 50.000 matches between one random player and one smart player and the smart player wins in ~85% of cases and lost in ~8% of matches.