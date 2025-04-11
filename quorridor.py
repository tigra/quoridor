import math
import random
import time
import os
import copy
from collections import deque

import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QuorridorState:
    def __init__(self):
        # Board size
        self.size = 9

        # Player positions (column, row), 0-indexed internally
        self.player_positions = [(4, 0), (4, 8)]  # W starts at E1, B starts at E9

        # Remaining walls for each player
        self.remaining_walls = [10, 10]

        # Horizontal walls - True if wall exists at position (i, j)
        # Indexed as horizontal_walls[row][column]
        # self.horizontal_walls = np.array([[False for _ in range(9)] for _ in range(9)])
        self.horizontal_walls = [[False for _ in range(9)] for _ in range(9)]
        # self.horizontal_walls = np.zeros((9, 9), dtype=np.bool)

        # Vertical walls - True if wall exists at position (i, j)
        # Indexed as vertical_walls[row][column]
        # self.vertical_walls = np.array([[False for _ in range(9)] for _ in range(8)])
        self.vertical_walls = [[False for _ in range(9)] for _ in range(8)]
        # self.vertical_walls = np.zeros((8, 9), dtype=np.bool)

        # Current player (0 for White, 1 for Black)
        self.current_player = 0

        # Game history
        self.history = []

    def clone(self):
        """Create a deep copy of the current state"""
        new_state = QuorridorState()
        new_state.player_positions = list(self.player_positions)
        new_state.remaining_walls = list(self.remaining_walls)
        new_state.horizontal_walls = [row[:] for row in self.horizontal_walls]
        new_state.vertical_walls = [row[:] for row in self.vertical_walls]
        # new_state.vertical_walls = np.copy(self.vertical_walls)
        # new_state.horizontal_walls = np.copy(self.horizontal_walls)
        new_state.current_player = self.current_player
        new_state.history = list(self.history)
        return new_state

    def get_winner(self):
        """Return the winner of the game if any"""
        # White (player 0) wins by reaching the bottom row (row 8)
        if self.player_positions[0][1] == 8:
            return 0
        # Black (player 1) wins by reaching the top row (row 0)
        if self.player_positions[1][1] == 0:
            return 1
        return None

    def is_game_over(self):
        """Check if the game is over"""
        return self.get_winner() is not None


class QuorridorGame:
    def __init__(self):
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

    def get_legal_actions(self, state):
        """Get all legal actions for the current player"""
        actions = []

        # Pawn movements
        moves = self._get_legal_pawn_moves(state)
        for move in moves:
            actions.append(('move', move))

        # Wall placements
        if state.remaining_walls[state.current_player] > 0:
            # Horizontal walls (A2h through H9h)
            for row in range(1, 9):
                for col in range(8):
                    if self._is_valid_horizontal_wall(state, col, row):
                        actions.append(('wall_h', (col, row)))

            # Vertical walls (B1v through I8v)
            for row in range(8):
                for col in range(1, 9):
                    if self._is_valid_vertical_wall(state, col, row):
                        actions.append(('wall_v', (col, row)))

        return actions

    def _get_legal_pawn_moves(self, state):
        """Get all legal pawn moves for the current player"""
        legal_moves = []
        curr_pos = state.player_positions[state.current_player]
        opponent_pos = state.player_positions[1 - state.current_player]

        # Check regular moves in all four directions
        for dx, dy in self.directions:
            cur_row = curr_pos[1]
            cur_col = curr_pos[0]
            new_x, new_y = cur_col + dx, cur_row + dy

            # Check if the move is within bounds
            if not (0 <= new_x < state.size and 0 <= new_y < state.size):
                continue

            # Check if there's a wall blocking the move
            if dx == -1 and cur_col > 0 and cur_row < 8 and state.vertical_walls[cur_row][cur_col]:  # Left
                continue
            if dx == 1 and cur_col < 8 and cur_row < 8 and state.vertical_walls[cur_row][cur_col + 1]:  # Right
                continue
            if dy == -1 and cur_row > 0 and state.horizontal_walls[cur_row][cur_col]:  # Up
                continue
            if dy == 1 and cur_row < 8 and state.horizontal_walls[cur_row + 1][cur_col]:  # Down
                continue

            # If opponent is in this spot, check for jumps
            if (new_x, new_y) == opponent_pos:
                # Try to jump over the opponent
                jump_x, jump_y = new_x + dx, new_y + dy

                # Check if the jump is within bounds
                if not (0 <= jump_x < state.size and 0 <= jump_y < state.size):
                    continue

                # Check if there's a wall blocking the jump
                if dx == -1 and new_x > 0 and state.vertical_walls[new_y][new_x]:  # Left
                    # Check for diagonal jumps if direct jump is blocked
                    if dy == 0:  # Only try diagonal jumps for horizontal movement
                        # Try jumping up-diagonal
                        if new_y > 0 and not state.horizontal_walls[new_y][new_x]:
                            legal_moves.append((new_x, new_y - 1))
                        # Try jumping down-diagonal
                        if new_y < 8 and not state.horizontal_walls[new_y + 1][new_x]:
                            legal_moves.append((new_x, new_y + 1))
                    continue
                if dx == 1 and new_x < 8 and state.vertical_walls[new_y][new_x + 1]:  # Right
                    # Check for diagonal jumps if direct jump is blocked
                    if dy == 0:  # Only try diagonal jumps for horizontal movement
                        # Try jumping up-diagonal
                        if new_y > 0 and not state.horizontal_walls[new_y][new_x]:
                            legal_moves.append((new_x, new_y - 1))
                        # Try jumping down-diagonal
                        if new_y < 8 and not state.horizontal_walls[new_y + 1][new_x]:
                            legal_moves.append((new_x, new_y + 1))
                    continue
                if dy == -1 and new_y > 0 and state.horizontal_walls[new_y][new_x]:  # Up
                    # Check for diagonal jumps if direct jump is blocked
                    if dx == 0:  # Only try diagonal jumps for vertical movement
                        # Try jumping left-diagonal
                        if new_x > 0 and not state.vertical_walls[new_y][new_x]:
                            legal_moves.append((new_x - 1, new_y))
                        # Try jumping right-diagonal
                        if new_x < 8 and not state.vertical_walls[new_y][new_x + 1]:
                            legal_moves.append((new_x + 1, new_y))
                    continue
                if dy == 1 and new_y < 8 and state.horizontal_walls[new_y + 1][new_x]:  # Down
                    # Check for diagonal jumps if direct jump is blocked
                    if dx == 0:  # Only try diagonal jumps for vertical movement
                        # Try jumping left-diagonal
                        if new_x > 0 and not state.vertical_walls[new_y][new_x]:
                            legal_moves.append((new_x - 1, new_y))
                        # Try jumping right-diagonal
                        if new_x < 8 and not state.vertical_walls[new_y][new_x + 1]:
                            legal_moves.append((new_x + 1, new_y))
                    continue

                legal_moves.append((jump_x, jump_y))
            else:
                legal_moves.append((new_x, new_y))

        return legal_moves

    def _is_valid_horizontal_wall(self, state, col, row):
        """Check if placing a horizontal wall at (col, row) is valid"""
        # Check if there's already a wall here
        if state.horizontal_walls[row][col]:
            return False

        # Check for wall collisions with other horizontal walls (a horizontal wall spans 2 cells)
        if col < 7 and state.horizontal_walls[row][col + 1]:
            return False
        if col > 0 and state.horizontal_walls[row][col - 1]:
            return False

        # Check for intersections with vertical walls
        if state.vertical_walls[row - 1][col] or (col < 8 and state.vertical_walls[row - 1][col + 1]):
            return False

        # Place the wall temporarily
        state.horizontal_walls[row][col] = True

        # Check if both players still have a path to their goals
        has_path = self._check_paths_exist(state)

        # Remove the temporary wall
        state.horizontal_walls[row][col] = False

        return has_path

    def _is_valid_vertical_wall(self, state, col, row):
        """Check if placing a vertical wall at (col, row) is valid"""
        # Check if there's already a wall here
        if state.vertical_walls[row][col]:
            return False

        # Check for wall collisions with other vertical walls (a vertical wall spans 2 cells)
        if row < 7 and state.vertical_walls[row + 1][col]:
            return False
        if row > 0 and state.vertical_walls[row - 1][col]:
            return False

        # Check for intersections with horizontal walls
        if state.horizontal_walls[row + 1][col - 1] or (row < 8 and state.horizontal_walls[row + 1][col]):
            return False

        # Place the wall temporarily
        state.vertical_walls[row][col] = True

        # Check if both players still have a path to their goals
        has_path = self._check_paths_exist(state)

        # Remove the temporary wall
        state.vertical_walls[row][col] = False

        return has_path

    def _check_paths_exist(self, state):
        """Check if both players have a path to their goals"""
        # Check if White has a path to the bottom row
        if not self._has_path_to_goal(state, 0):
            return False

        # Check if Black has a path to the top row
        if not self._has_path_to_goal(state, 1):
            return False

        return True

    def _has_path_to_goal(self, state, player):
        """Check if player has a path to their goal using BFS"""
        start = state.player_positions[player]
        visited = set([start])
        queue = deque([start])

        target_row = 8 if player == 0 else 0

        while queue:
            x, y = queue.popleft()

            # Check if we've reached the goal
            if y == target_row:
                return True

            # Try all four directions
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy

                # Check if the move is within bounds
                if not (0 <= new_x < state.size and 0 <= new_y < state.size):
                    continue

                # Check if there's a wall blocking the move
                if dx == -1 and x <= 0 and state.vertical_walls[y][x]:  # Left
                    continue
                if dx == 1 and x >= 8 and state.vertical_walls[y][x + 1]:  # Right
                    continue
                if dy == -1 and y <= 0 and state.horizontal_walls[y][x]:  # Up
                    continue
                if dy == 1 and y >= 8 and state.horizontal_walls[y + 1][x]:  # Down
                    continue

                # If we haven't visited this cell, add it to the queue
                if (new_x, new_y) not in visited:
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))

        # If we've exhausted all possibilities without reaching the goal, there's no path
        return False

    def apply_action(self, state, action):
        """Apply an action to the state"""
        action_type, action_data = action

        if action_type == 'move':
            state.player_positions[state.current_player] = action_data
        elif action_type == 'wall_h':
            col, row = action_data
            state.horizontal_walls[row][col] = True
            if col < 8:
                state.horizontal_walls[row][col+1] = True
                state.remaining_walls[state.current_player] -= 1
        elif action_type == 'wall_v':
            col, row = action_data
            state.vertical_walls[row][col] = True
            if row < 7:
                state.vertical_walls[row+1][col] = True
                state.remaining_walls[state.current_player] -= 1

        # Record the action in history
        state.history.append((state.current_player, action))

        # Switch to the other player
        state.current_player = 1 - state.current_player

    def visualize_board(self, state, game_num, turn_num):
        """Generate a string representation of the board"""
        output = []
        output.append(f"Game #{game_num}")
        output.append("")
        output.append(f"Turn {turn_num} ({['W', 'B'][state.current_player]}):")

        # grid_char = '+'
        grid_char = ' '
        hw_char = '_'

        # Add column headers
        output.append("    A   B   C   D   E   F   G   H   I  ")
        output.append(f"  {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char}")

        for row in range(9):
            # Cell row
            cell_row = f"{row + 1} |"
            for col in range(9):
                if state.player_positions[0] == (col, row):
                    cell_row += " W "
                elif state.player_positions[1] == (col, row):
                    cell_row += " B "
                else:
                    cell_row += "   "

                # Vertical wall or space
                if col < 8:
                    # Fixed the index error by checking if there's a vertical wall at position col
                    if col < 8 and row < 8 and state.vertical_walls[row][col + 1]:
                        cell_row += "|"
                    else:
                        cell_row += "."
            cell_row += "|"
            output.append(cell_row)

            # Horizontal wall row
            if row < 8:
                wall_row = f"  {grid_char}"
                for col in range(9):
                    # Wall or space
                    if row + 1 < 9 and state.horizontal_walls[row][col]:
                        wall_row += f" {hw_char} "
                    else:
                        wall_row += " . "

                    # Intersection
                    wall_row += grid_char
                output.append(wall_row)

        # Bottom edge
        output.append(
            f"  {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char} - {grid_char}")

        return "\n".join(output)


def play_random_game(game_num=1, max_turns=200):
    """Play a game with random moves from both players"""
    state = QuorridorState()
    game = QuorridorGame()
    game_history = []

    turn_num = 0

    # Initial board state
    board_str = game.visualize_board(state, game_num, turn_num)
    print(board_str)
    print("\n")
    game_history.append(board_str)

    while not state.is_game_over() and turn_num < max_turns:
        turn_num += 1

        # Get legal actions
        legal_actions = game.get_legal_actions(state)

        if not legal_actions:
            print("No legal actions available! Game is a draw.")
            break

        # Choose a random action
        action = random.choice(legal_actions)

        # Apply the action
        game.apply_action(state, action)

        # Get action description
        action_desc = ""
        if action[0] == 'move':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y + 1  # Convert 0-indexed to 1-9
            action_desc = f"Move to {col}{row}"
        elif action[0] == 'wall_h':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y
            action_desc = f"Place horizontal wall at {col}{row}"
        elif action[0] == 'wall_v':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y + 1
            action_desc = f"Place vertical wall at {col}{row}"

        # Visualize the board
        player_symbol = ['W', 'B'][1 - state.current_player]  # Previous player
        board_str = game.visualize_board(state, game_num, turn_num)
        print(f"Turn {turn_num} ({player_symbol}): {action_desc}")
        print(board_str)
        print("\n")

        # Add to game history with the action description
        game_history.append(f"Turn {turn_num} ({player_symbol}): {action_desc}")
        game_history.append(board_str)

        # Add a small delay for readability
        time.sleep(0.2)

    # Game over
    winner = state.get_winner()
    if winner is not None:
        win_message = f"Game over! Player {['White', 'Black'][winner]} wins!"
        print(win_message)
        game_history.append(win_message)
    else:
        draw_message = "Game ended in a draw or reached maximum turns!"
        print(draw_message)
        game_history.append(draw_message)

    return game_history


def save_game_history(game_num, history):
    """Save the game history to a file"""
    os.makedirs("game_histories", exist_ok=True)
    filename = f"game_histories/quorridor_random_game_{game_num}.txt"
    with open(filename, 'w') as f:
        f.write("\n".join(history))
    print(f"Game history saved to {filename}")


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = None  # Lazy initialization

    def add_child(self, state, action):
        """Add a child node"""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        # Update wins if the result is for the player who just moved
        # (i.e., the opponent of the player in the current state)
        if result == 1 - self.state.current_player:
            self.wins += 1

    def get_ucb_score(self, exploration_weight):
        """Get UCB score for this node"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTSPlayer:
    def __init__(self, game, exploration_weight=1.41, simulations=100, player_id=None):
        self.game = game
        self.exploration_weight = exploration_weight
        self.simulations = simulations
        self.player_id = player_id  # For logging purposes

    def get_action(self, state):
        """Get the best action using MCTS"""
        root = MCTSNode(state.clone())

        # Perform MCTS simulations
        for _ in tqdm.tqdm(range(self.simulations)):
            # 1. Selection: Find the most promising leaf node
            leaf = self._select(root)

            # 2. Expansion: If leaf is not terminal and has untried actions, expand it
            if not leaf.state.is_game_over() and (leaf.untried_actions is None or len(leaf.untried_actions) > 0):
                expanded_node = self._expand(leaf)

                # 3. Simulation: Run a random simulation from the expanded node
                result = self._simulate(expanded_node.state)

                # 4. Backpropagation: Update statistics up the tree
                self._backpropagate(expanded_node, result)
            else:
                # If leaf is terminal or fully expanded, backpropagate from leaf
                result = leaf.state.get_winner()
                if result is None:  # If game isn't over, simulate
                    result = self._simulate(leaf.state)
                self._backpropagate(leaf, result)

        # Choose the best child based on visit count (most robust)
        if not root.children:
            return None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _select(self, node):
        """Select a leaf node using UCB1"""
        current = node

        # While not a leaf node (has children) and is fully expanded
        while current.children and (current.untried_actions is None or len(current.untried_actions) == 0):
            # Choose child with highest UCB score
            current = max(current.children, key=lambda c: c.get_ucb_score(self.exploration_weight))

        return current

    def _expand(self, node):
        """Expand a node by adding a child"""
        # If untried_actions is None, initialize it
        if node.untried_actions is None:
            node.untried_actions = self.game.get_legal_actions(node.state)

        # Choose a random untried action
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))

        # Create a new state by applying the action
        new_state = node.state.clone()
        self.game.apply_action(new_state, action)

        # Add a child node and return it
        return node.add_child(new_state, action)

    def _simulate(self, state):
        """Simulate a random game from the given state"""
        state = state.clone()

        while not state.is_game_over():
            # Get legal actions
            actions = self.game.get_legal_actions(state)

            # Choose a random action
            if not actions:
                break  # No legal moves, draw
            action = random.choice(actions)

            # Apply the action
            self.game.apply_action(state, action)

        # Return the winner (or None for draw)
        return state.get_winner()

    def _backpropagate(self, node, result):
        """Backpropagate the result up the tree"""
        while node is not None:
            node.update(result)
            node = node.parent

def play_mcts_vs_mcts_game(game_num=1, simulations1=1000, simulations2=1000, max_turns=200, track_history=False):
    """Play a game between two MCTS players"""
    state = QuorridorState()
    game = QuorridorGame()

    # Create MCTS players
    player1 = MCTSPlayer(game, simulations=simulations1, player_id="White")
    player2 = MCTSPlayer(game, simulations=simulations2, player_id="Black")

    players = [player1, player2]
    game_history = []

    turn_num = 0

    # Initial board state
    board_str = game.visualize_board(state, game_num, turn_num)
    print(board_str)
    print("\n")
    if track_history:
        game_history.append(board_str)

    start_time = time.time()

    while not state.is_game_over() and turn_num < max_turns:
        turn_num += 1
        current_player = players[state.current_player]

        print(f"Turn {turn_num}: {['White', 'Black'][state.current_player]} thinking...")
        turn_start = time.time()

        # Get action from MCTS player
        action = current_player.get_action(state)

        turn_time = time.time() - turn_start
        print(f"Move decided in {turn_time:.2f} seconds")

        if action is None:
            print("No legal actions available! Game is a draw.")
            break

        # Apply the action
        game.apply_action(state, action)

        # Get action description
        action_desc = ""
        if action[0] == 'move':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y + 1  # Convert 0-indexed to 1-9
            action_desc = f"Move to {col}{row}"
        elif action[0] == 'wall_h':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y
            action_desc = f"Place horizontal wall at {col}{row}"
        elif action[0] == 'wall_v':
            x, y = action[1]
            col = chr(97 + x).upper()  # Convert 0-indexed to A-I
            row = y + 1
            action_desc = f"Place vertical wall at {col}{row}"

        # Visualize the board
        player_symbol = ['W', 'B'][1 - state.current_player]  # Previous player
        board_str = game.visualize_board(state, game_num, turn_num)
        print(f"Turn {turn_num} ({player_symbol}): {action_desc}")
        print(board_str)
        print(f"Walls remaining - White: {state.remaining_walls[0]}, Black: {state.remaining_walls[1]}")
        print("\n")

        # Add to game history with the action description
        if track_history:
            game_history.append(f"Turn {turn_num} ({player_symbol}): {action_desc}")
            game_history.append(board_str)
            game_history.append(f"Walls remaining - White: {state.remaining_walls[0]}, Black: {state.remaining_walls[1]}")

    # Game over
    total_time = time.time() - start_time
    winner = state.get_winner()
    if winner is not None:
        win_message = f"Game over! Player {['White', 'Black'][winner]} wins!"
        print(win_message)
        if track_history:
            game_history.append(win_message)
    else:
        draw_message = "Game ended in a draw or reached maximum turns!"
        print(draw_message)
        if track_history:
            game_history.append(draw_message)

    print(f"Game completed in {total_time:.2f} seconds, {turn_num} turns")
    if track_history:
        game_history.append(f"Game completed in {total_time:.2f} seconds, {turn_num} turns")

    return game_history


class NeuralNetPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def predict(self, state):
        """Predict action probabilities for a state"""
        features = self._extract_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(features_tensor)
        return probs.squeeze(0).numpy()

    def _extract_features(self, state):
        """Extract features from a state for the neural network"""
        features = []

        # Player positions
        for player in range(2):
            pos = state.player_positions[player]
            # One-hot encode position
            for i in range(81):  # 9x9 board
                features.append(1.0 if i == pos[1] * 9 + pos[0] else 0.0)

        # Walls
        for row in range(9):
            for col in range(8):
                features.append(1.0 if state.horizontal_walls[row][col] else 0.0)

        for row in range(8):
            for col in range(9):
                features.append(1.0 if state.vertical_walls[row][col] else 0.0)

        # Remaining walls
        for player in range(2):
            features.append(state.remaining_walls[player] / 10.0)

        # Current player
        features.append(state.current_player)

        return features


def play_game(game, policy1, policy2, game_num):
    """Play a game between two policies and return the game history"""
    state = QuorridorState()
    policies = [policy1, policy2]
    mcts = MCTS(game)
    game_history = []
    examples = []

    turn_num = 0
    while not state.is_game_over() and turn_num < 200:
        current_policy = policies[state.current_player]

        # Use MCTS guided by policy
        mcts_probs = np.zeros(len(game.get_legal_actions(state)))

        print("selecting action:")
        # Use pure MCTS for now, but could integrate policy here
        action = mcts.select_action(state)
        print('action:', action)

        # Store state and action probabilities for training
        examples.append((state.clone(), mcts_probs, None))

        # Visualize the board
        board_str = game.visualize_board(state, game_num, turn_num)
        print(board_str)
        print("\n")
        game_history.append(board_str)

        # Apply the action
        game.apply_action(state, action)
        turn_num += 1

    # Set the winner for all examples
    winner = state.get_winner()
    for i in range(len(examples)):
        examples[i] = (examples[i][0], examples[i][1], winner)

    # Final board state
    board_str = game.visualize_board(state, game_num, turn_num)
    print(board_str)
    print(f"\nGame over!  {['White', 'Black'][winner] + ' wins' if winner else 'Draw'}")
    game_history.append(board_str)
    game_history.append(f"\nGame over! Player {['White', 'Black'][winner]} wins!")

    return examples, game_history


class MCTS:
    def __init__(self, game, exploration_weight=1.0):
        self.game = game
        self.exploration_weight = exploration_weight

    def select_action(self, state, num_simulations=50):
        """Select an action using MCTS"""
        root = MCTSNode(state)

        print("running simulations...")
        for _ in tqdm.tqdm(range(num_simulations)):
            node = self._tree_policy(root)
            winner = self._simulate(node.state)
            self._backpropagate(node, winner)

        # Return best child's action
        return self._best_child(root, 0).action

    def _tree_policy(self, node):
        """Select a node to run a simulation from"""
        current = node
        while not current.state.is_game_over():
            if current.untried_actions is None:
                current.untried_actions = self.game.get_legal_actions(current.state)

            if len(current.untried_actions) > 0:
                return self._expand(current)
            else:
                current = self._best_child(current, self.exploration_weight)

        return current

    def _expand(self, node):
        """Add a new child node"""
        action = node.untried_actions.pop()
        new_state = node.state.clone()
        self.game.apply_action(new_state, action)

        child = MCTSNode(new_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _best_child(self, node, exploration_weight):
        """Select the best child node"""
        if not node.children:
            return node

        # Use UCB1 formula to select the best child
        def ucb1(child):
            exploitation = child.wins / child.visits
            exploration = math.sqrt(2 * math.log(node.visits) / child.visits)
            return exploitation + exploration_weight * exploration

        return max(node.children, key=ucb1)

    def _simulate(self, state):
        """Run a random simulation from the current state"""
        state = state.clone()
        game = self.game

        while not state.is_game_over():
            actions = game.get_legal_actions(state)
            action = random.choice(actions)
            game.apply_action(state, action)

        return state.get_winner()

    def _backpropagate(self, node, winner):
        """Backpropagate the simulation result"""
        while node is not None:
            node.visits += 1
            if winner == node.state.current_player:
                node.wins += 1
            node = node.parent


def train_policy(policy, examples, optimizer):
    """Train the policy network using policy gradient"""
    policy.train()

    for state, action_probs, winner in examples:
        features = policy._extract_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        # Forward pass
        probs = policy(features_tensor)

        # Compute loss
        action_tensor = torch.FloatTensor(action_probs).unsqueeze(0)
        reward = 1.0 if winner == state.current_player else -1.0
        loss = -torch.sum(action_tensor * torch.log(probs)) * reward

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def self_play_training(game, num_iterations=5, games_per_iteration=2, simulations_per_move=100):
    """Train a policy through self-play"""
    # Neural network parameters
    input_size = 81 * 2 + 8 * 9 + 9 * 8 + 2 + 1  # Player positions, walls, remaining walls, current player
    # hidden_size = 128
    # output_size = 200  # Approximate max number of possible actions
    hidden_size = 32
    output_size = 200  # Approximate max number of possible actions

    # Initialize policy
    policy = NeuralNetPolicy(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    # Initialize frozen policy
    frozen_policy = copy.deepcopy(policy)

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        all_examples = []

        # Play games
        for game_num in range(games_per_iteration):
            print(f"\nPlaying game {game_num + 1}/{games_per_iteration}")
            examples, game_history = play_game(game, policy, frozen_policy,
                                               game_num + 1 + iteration * games_per_iteration)
            all_examples.extend(examples)
            save_game_history(game_num + 1 + iteration * games_per_iteration, game_history)

        # Train policy
        print("Training policy...")
        train_policy(policy, all_examples, optimizer)

        # Update frozen policy periodically
        if (iteration + 1) % 2 == 0:
            print("Updating frozen policy...")
            frozen_policy = copy.deepcopy(policy)
            save_policy(policy, f'quorridor_policy_it{iteration}.pt')
        save_policy(policy, f'final_quorridor_policy.pt')


def save_policy(policy, filename="quorridor_policy.pt"):
    """Save the neural network policy to a file"""
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    torch.save(policy.state_dict(), filepath)
    print(f"Policy saved to {filepath}")


def load_policy(policy, filename="quorridor_policy.pt", map_location=None):
    """Load the neural network policy from a file"""
    directory = "models"
    filepath = os.path.join(directory, filename)

    if os.path.exists(filepath):
        policy.load_state_dict(torch.load(filepath, map_location=map_location))
        print(f"Policy loaded from {filepath}")
        return True
    else:
        print(f"No saved policy found at {filepath}")
        return False


if __name__ == "__main__":
    # Play MCTS vs MCTS games
    num_games = 1

    np.random.seed(42)

    # Number of simulations for each player
    simulations_player1 = 35  # White
    simulations_player2 = 30  # Black

    for game_num in range(1, num_games + 1):
        print(f"\n===== GAME {game_num} =====\n")
        print(f"White: {simulations_player1} simulations, Black: {simulations_player2} simulations")

        # game_history = play_random_game(game_num)
        game_history = play_mcts_vs_mcts_game(
            game_num,
            simulations1=simulations_player1,
            simulations2=simulations_player2
        )
    save_game_history(game_num, game_history)
    print("\n")

    # game = QuorridorGame()
    # # Change these parameters to control the training process
    # num_iterations = 5  # Number of training iterations
    # games_per_iteration = 2  # Number of games to play per iteration
    # simulations_per_move = 50  # Number of MCTS simulations per move
    #
    # self_play_training(game, num_iterations, games_per_iteration, simulations_per_move)



# if __name__ == "__main__":
#     # Play multiple random games
#     num_games = 3
#
#     for game_num in range(1, num_games + 1):
#         print(f"\n===== GAME {game_num} =====\n")
#         game_history = play_random_game(game_num)
#         save_game_history(game_num, game_history)
#         print("\n")

# if __name__ == "__main__":
#     game = QuorridorGame()
#
#     # Change these parameters to control the training process
#     num_iterations = 5  # Number of training iterations
#     games_per_iteration = 2  # Number of games to play per iteration
#     simulations_per_move = 100  # Number of MCTS simulations per move
#
#     self_play_training(game, num_iterations, games_per_iteration, simulations_per_move)