# A simplified version of Quoridor game AI self-play using Monte Carlo Tree Search

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

* Sometimes hangs during thinking through the step
* Sometimes tries to place the wall in nonsensical place (ignored during turn execution)
* NN implementation does not invert the player/board, so play is nonsensical for second player
* Requires refactoring and further testing

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
