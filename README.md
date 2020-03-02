# A simplified version of Quoridor game using Monte Carlo Tree Search

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
pip install mcst

python quoridor.py
```