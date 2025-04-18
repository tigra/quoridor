# Quoridor game AI self-play using Monte Carlo Tree Search

[Quoridor](https://en.wikipedia.org/wiki/Quoridor) game is played on 9x9 board. Players have 1 pawn each placed in the middles of boards sides. 2-player version is supported now (4-player - not yet).
On each turn, players can move their pawns, or place the walls between cells. Each wall is 2 cells long. Each player has 10 walls. In case opponent's pawn is blocking the way, player can jump over it.
The goal of the game is to reach the opposite side of the board with one's pawn. Completely blocking the path to victory for any pawn with walls is prohibited.

## New version vibecoded together with Claude

Full support of wall placements, pawn moves and jums, checks for
wall intersections and walls not blocking the path. 
Working plain MCTS two-player game implementation with text based visualization
and game history saving.

* (activate your virtual env)
* `pip install -r requirements.txt`
* `python quoridor.py`

Contains plain MCTS implementation, and (commented out) random game implementation, unfinished NN policy training 
procedure.

### Known issues

* Sometimes tries to place the wall in nonsensical place (ignored during turn execution) - seems fixed, but will keep an eye
* Sometimes blocks someone's pawn completely
* Sometimes allows jumping over the wall (probably connected with previous issue)
* Need to check if all allowed jumpover situations are supported
* NN implementation does not invert the player/board, so play is nonsensical for second player
* Requires refactoring and further testing
* Would be great to use Numpy more
* Would be great to parallelize simulations to have players think faster

## Older version
(Very) simplified :)
Two players start on different ends on 9*9 grid. 
They can move up, down, left, right inside the grid.
Player who first reaches the opposite end of a grid wins.

Here, each player uses own mcst searcher. First, it searches for 15 seconds.
For subsequent moves players have 1 second each to extend their trees and decide for move.

```sh
# But better inside virtualenv...
pip install numpy
pip install matplotlib
pip install mcts

python quorridor_old.py
```

### How it looks

```
Turn 77 (B):
      A   B   C   D   E   F   G   H   I  
    +---+---+---+---+---+---+---+---+---+
 1  |   .   .   .   .   .   .   .   |   |
    + . + . + . +-------+ . + . + . | . +
 2  |   . B .   .   |   |   .   .   |   |
    + . + . +-------| . | . + . + . + . +
 3  |   .   .   .   |   |   .   .   .   |
    +-------+ . + . + . + . + . +-------+
 4  |   .   .   .   .   .   .   .   .   |
    + . + . +-------+ . +---------------+
 5  |   |   |   .   .   |   .   .   .   |
    + . | . | . + . + . | . + . + . + . +
 6  |   |   |   .   .   |   .   .   .   |
    + . + . + . +-------+ . +-------+ . +
 7  |   .   .   .   .   .   .   .   .   |
    + . +-------+ . + . +-------+ . + . +
 8  |   .   .   .   .   .   |   .   |   |
    + . +-------+ . + . + . + . + . + . +
 9  |   .   .   . W .   .   |   .   |   |
    +---+---+---+---+---+---+---+---+---+
Walls remaining - White: 0, Black: 0


Game over! Player White wins!
Game completed in 205.64 seconds, 77 turns
Game history saved to game_histories/quorridor_random_game_1.txt
```
