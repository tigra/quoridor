import traceback
from collections import namedtuple
from typing import NamedTuple

import mcts
import numpy as np

Coords = namedtuple('Coords', ['x', 'y'])

SHIFT_X = dict(up=0, down=0, left=-1, right=+1)
SHIFT_Y = dict(up=1, down=-1, left=0, right=0)

class QuridorState:
    def __init__(self, p1=Coords(5, 1), p2=Coords(5, 9), current_player=1, play_for=None, step=0):
        self.h_wall = np.zeros((11, 10), dtype=np.bool)
        self.h_wall[0,:] = True
        self.h_wall[-1,:] = True
        self.v_wall = np.zeros((10, 11), dtype=np.bool)
        self.v_wall[:,0] = True
        self.v_wall[:,-1] = True
        self.p1 = p1
        self.p2 = p2
        self.current_player = current_player
        self.play_for=play_for
        if play_for is None:
            raise ValueError("play_for==None")
        self.step = step

    def getPossibleActions(self):
        try:
            actions = []
            if self.current_player == 1:
                actions = self.possible_pawn_moves(self.p1, self.p2)
            elif self.current_player == 2:
                actions = self.possible_pawn_moves(self.p2, self.p1)
        except IndexError as e:
            print(e)
            traceback.print_exc()
        return actions


    def possible_pawn_moves(self, one, other):
        # actions = [a for a in ['up', 'down', 'left', 'right'] if self.can_move(one, SHIFT_X[a], SHIFT_Y[a])]
        actions = []
        if not self.h_wall[one.y + 1, one.x]:
            actions.append('up')
        if not self.h_wall[one.y - 1, one.x]:
            actions.append('down')
        if not self.v_wall[one.y, one.x - 1]:
            actions.append('left')
        if not self.v_wall[one.y, one.x + 1]:
            actions.append('right')
        return actions

    def move(self, coords, action):
        return Coords(coords.x + SHIFT_X[action], coords.y + SHIFT_Y[action])

    def takeAction(self, action):
        if self.current_player == 1:
            return QuridorState(p1=self.move(self.p1, action), p2=self.p2, current_player=2,
                                play_for=self.play_for, step=self.step+1)
        elif self.current_player == 2:
            return QuridorState(p1=self.p1, p2=self.move(self.p2, action), current_player=1,
                                play_for=self.play_for, step=self.step+1)

    def isTerminal(self):
        if self.current_player == 1:
            return self.p1.y == 9
        elif self.current_player == 2:
            return self.p2.y == 1

    def getReward(self):
        return 1 if self.current_player == self.play_for and self.isTerminal() else 0

    def __repr__(self):
        return f"QuoridorState(p1={self.p1}, p2={self.p2}, current_player={self.current_player}"

# s = QuridorState()
# for a in ['up','up', 'up']:
#     print(f"{s} -> {s.getPossibleActions()}")
#     s=s.takeAction(a)

state1 = QuridorState(play_for=1)
state2 = QuridorState(play_for=2)

DUMB_TIME=1000
WISE_TIME=15000
searcher1_wise = mcts.mcts(timeLimit=WISE_TIME)
searcher2_wise = mcts.mcts(timeLimit=WISE_TIME)
searcher1_dumb = mcts.mcts(timeLimit=DUMB_TIME)
searcher2_dumb = mcts.mcts(timeLimit=DUMB_TIME)

searcher1 = searcher1_wise
searcher2 = searcher2_wise

print(state1)


while not state1.isTerminal():
    action = searcher1.search(initialState=state1)
    searcher1 = searcher1_dumb
    print(f"Player {state1.current_player} moves {action}")
    state1 = state1.takeAction(action)
    state2 = state2.takeAction(action)
    print(f"step:{state1.step} state: {state1}    reward: {state1.getReward()}")

    if state1.isTerminal():
        break
    action = searcher2.search(initialState=state2)
    searcher2 = searcher2_dumb
    print(f"Player {state1.current_player} moves {action}")
    state1 = state1.takeAction(action)
    state2 = state2.takeAction(action)
    print(f"step:{state2.step} state: {state2}    reward: {state2.getReward()}")


print(f"Player {state1.current_player} won!")

