import heapq
import numpy as np
from tqdm import tqdm

from enum import Enum

from dataclasses import dataclass


@dataclass
class TrackPiece:
    position: tuple  # (y, x)
    track_type: "MoveNeighbors"  # Type of track piece
    direction_from: tuple  # (dy, dx) direction we arrived from

    def __lt__(self, other):
        """
        Compare TrackPiece objects based on their position.
        This is required for using TrackPiece in a priority queue.
        """
        return self.position < other.position


class MoveNeighbors(Enum):
    # Track patterns: each tuple represents (center_dy, center_dx) relative positions
    NS = [(0, 0)]           # Vertical track: single cell
    EW = [(0, 0)]           # Horizontal track: single cell
    NE = [(0, 0), (-1, 1)]  # Northeast curve
    NW = [(0, 0), (-1, -1)] # Northwest curve  
    SE = [(0, 0), (1, 1)]   # Southeast curve
    SW = [(0, 0), (1, -1)]  # Southwest curve


def get_track_shape(direction):
    """
    Determine the track shape based on the incoming direction.
    
    :param direction: tuple (dy, dx) for the direction we arrived from
    :return: MoveNeighbors enum value representing the track shape
    """
    if direction is None:
        return MoveNeighbors.NS  # Default starting track
    
    dy, dx = direction
    # Determine track shape based on the movement direction
    if dy != 0 and dx == 0:  # Moving vertically
        return MoveNeighbors.NS
    elif dy == 0 and dx != 0:  # Moving horizontally
        return MoveNeighbors.EW
    elif dy < 0 and dx > 0:  # Moving northeast
        return MoveNeighbors.NE
    elif dy < 0 and dx < 0:  # Moving northwest
        return MoveNeighbors.NW
    elif dy > 0 and dx > 0:  # Moving southeast
        return MoveNeighbors.SE
    elif dy > 0 and dx < 0:  # Moving southwest
        return MoveNeighbors.SW
    else:
        return MoveNeighbors.NS  # Default


def is_diagonal_move(dy, dx):
    """Check if a move is diagonal (not pure NS or EW)."""
    return dy != 0 and dx != 0


def move_cost(cost_map, current, neighbor, previous_direction=None, turn_penalty=0.5, turn_90_multiplier=3.0):
    """
    Calculate the cost of moving from current to neighbor.

    :param cost_map: 2D numpy array with terrain costs
    :param current: tuple (y, x) for current position
    :param neighbor: tuple (y, x) for neighbor position
    :param previous_direction: tuple (dy, dx) for the direction we came from, or None for the first move
    :param turn_penalty: weight for penalty when turning (0-1 scale, where 1 is no penalty)
    :return: movement cost with momentum applied
    """
    # Base cost is altitude difference (cost_map expected as integer array)
    if cost_map[neighbor[0], neighbor[1]] == 0:
        return np.inf  # Impassable terrain
    base_cost = 1 + abs(cost_map[current[0], current[1]] - cost_map[neighbor[0], neighbor[1]])

    
    # If we have a previous direction, apply momentum/turn penalties
    if previous_direction is not None:
        current_direction = (neighbor[0] - current[0], neighbor[1] - current[1])

        # If we're turning (different direction), apply penalty
        if current_direction != previous_direction:
            # Detect a 90-degree (perpendicular) turn via dot product == 0
            dot = current_direction[0] * previous_direction[0] + current_direction[1] * previous_direction[1]
            if dot == 0:
                # Strongly discourage 90-degree turns by multiplying cost
                base_cost = base_cost * turn_90_multiplier
            else:
                # Other turns get the usual turn penalty (increase cost by dividing)
                base_cost = base_cost / turn_penalty

    return base_cost


def get_valid_neighbors(track_shape, direction):
    """
    Get valid neighbors based on track connectivity rules.
    
    :param track_shape: MoveNeighbors enum representing the current track shape
    :param direction: tuple (dy, dx) representing how we arrived at current position
    :return: List of valid neighbor directions (dy, dx)
    """
    if direction is None:
        # First move - can start with any straight or diagonal track
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Allow continuing in same direction, 90-degree turns, and diagonal moves
    dy, dx = direction
    
    # Always allow continuing in the same direction
    valid_moves = [(dy, dx)]
    
    # Allow 90-degree turns from current direction
    if dy != 0:  # Was moving vertically
        valid_moves.extend([(0, -1), (0, 1)])  # Can turn east/west
    if dx != 0:  # Was moving horizontally
        valid_moves.extend([(-1, 0), (1, 0)])  # Can turn north/south
    
    # Allow diagonal moves from any direction
    valid_moves.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    # Remove duplicates
    return list(set(valid_moves))


def find_path(start, end, cost_map, turn_penalty=0.5, turn_90_multiplier=3.0):
    """
    Finds the best path between a start and end point using the A* algorithm with momentum.

    :param start: tuple (y, x) for the start point
    :param end: tuple (y, x) for the end point
    :param cost_map: 2D numpy array where each cell has a cost to traverse
    :param turn_penalty: weight applied when turning (lower = higher penalty). Default 0.5 means turning costs 2x more
    :return: list of tuples representing the path, or None if no path is found
    """

    # cost_map = cost_map.T  # Transpose for (y, x) indexing
    # Heuristic function (Taxicab/Manhattan distance)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Convert cost_map to integer type for fast arithmetic
    cost_map = cost_map.astype(np.int32, copy=False)

    # Priority queue for open set: (f_score, counter, position, direction, track_shape)
    open_set = []
    counter = 0

    # came_from: map state (pos,dir) -> previous state (pos,dir)
    came_from = {}

    # g_score and f_score keyed by (pos,dir)
    g_score = {}
    f_score = {}

    # Precompute neighbor lookup cache for (track_shape, direction)
    neighbors_cache = {}
    directions_to_cache = [None, (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for shape in MoveNeighbors:
        for d in directions_to_cache:
            neighbors_cache[(shape, d)] = get_valid_neighbors(shape, d)

    # Initialize start states: push possible initial directions
    init_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    pushed_any = False
    for d in init_dirs:
        ny, nx = start[0] + d[0], start[1] + d[1]
        if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
            start_state = (start, d)
            g_score[start_state] = 0
            f = heuristic(start, end)
            f_score[start_state] = f
            track_shape = get_track_shape(d)
            heapq.heappush(open_set, (f, counter, start, d, track_shape))
            counter += 1
            came_from[start_state] = None
            pushed_any = True

    if not pushed_any:
        return None

    closed = set()

    iteration = 0
    with tqdm(total=cost_map.shape[0] * cost_map.shape[1], desc="Finding Path") as pbar:
        while open_set:
            if iteration % 100 == 0:
                pbar.update(1)
            f, _, current_pos, current_dir, current_track_shape = heapq.heappop(open_set)

            state = (current_pos, current_dir)

            # if we've already processed this state with a better g, skip
            if state in closed:
                iteration += 1
                continue

            closed.add(state)

            # If reached end (any direction), reconstruct path
            if current_pos == end:
                path = []
                cur = state
                while cur is not None:
                    pos, dirc = cur
                    shape = get_track_shape(dirc)
                    path.append(TrackPiece(position=pos, track_type=shape, direction_from=dirc))
                    cur = came_from.get(cur, None)
                return path[::-1]

            # Obtain valid moves from cache
            valid_moves = neighbors_cache.get((current_track_shape, current_dir))

            for dy, dx in valid_moves:
                ny, nx = current_pos[0] + dy, current_pos[1] + dx
                if not (0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]):
                    continue

                # Check altitude constraint for diagonal moves
                if is_diagonal_move(dy, dx):
                    altitude_diff = abs(cost_map[current_pos[0], current_pos[1]] - cost_map[ny, nx])
                    if altitude_diff > 0:
                        continue

                neighbor_pos = (ny, nx)
                move_direction = (dy, dx)
                neighbor_state = (neighbor_pos, move_direction)

                tentative_g = g_score.get(state, np.inf) + move_cost(
                    cost_map, current_pos, neighbor_pos, current_dir, turn_penalty, turn_90_multiplier
                )

                if tentative_g < g_score.get(neighbor_state, np.inf):
                    came_from[neighbor_state] = state
                    g_score[neighbor_state] = tentative_g
                    fval = tentative_g + heuristic(neighbor_pos, end)
                    f_score[neighbor_state] = fval
                    next_shape = get_track_shape(move_direction)
                    heapq.heappush(open_set, (fval, counter, neighbor_pos, move_direction, next_shape))
                    counter += 1

            iteration += 1

    return None  # No path found
