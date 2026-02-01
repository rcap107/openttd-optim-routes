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


def move_cost(cost_map, current, neighbor, previous_direction=None, turn_penalty=0.5):
    """
    Calculate the cost of moving from current to neighbor.

    :param cost_map: 2D numpy array with terrain costs
    :param current: tuple (y, x) for current position
    :param neighbor: tuple (y, x) for neighbor position
    :param previous_direction: tuple (dy, dx) for the direction we came from, or None for the first move
    :param turn_penalty: weight for penalty when turning (0-1 scale, where 1 is no penalty)
    :return: movement cost with momentum applied
    """
    # Base cost is altitude difference
    base_cost = 1 + abs(
        int(cost_map[current[0], current[1]]) - int(cost_map[neighbor[0], neighbor[1]])
    )

    # If we have a previous direction, apply momentum
    if previous_direction is not None:
        current_direction = (neighbor[0] - current[0], neighbor[1] - current[1])

        # If we're turning (different direction), apply penalty
        if current_direction != previous_direction:
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
        # First move - can start with any straight track
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # For now, let's allow more flexibility while still having some track logic
    # Allow continuing in same direction or making 90-degree turns
    dy, dx = direction
    
    # Always allow continuing in the same direction
    valid_moves = [(dy, dx)]
    
    # Allow 90-degree turns from current direction
    if dy != 0:  # Was moving vertically
        valid_moves.extend([(0, -1), (0, 1)])  # Can turn east/west
    if dx != 0:  # Was moving horizontally
        valid_moves.extend([(-1, 0), (1, 0)])  # Can turn north/south
    
    # Remove duplicates
    return list(set(valid_moves))


def find_path(start, end, cost_map, turn_penalty=0.5):
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

    # Priority queue for open set
    open_set = []
    # came_from dictionary to reconstruct the path
    came_from = {}

    # direction_from dictionary to track the direction we arrived from
    direction_from = {}

    # g_score: cost from start to the current node
    g_score = np.inf * np.ones(cost_map.shape, dtype=np.float32)
    g_score[start] = 0

    # f_score: g_score + heuristic
    f_score = np.inf * np.ones(cost_map.shape, dtype=np.float32)
    f_score[start] = heuristic(start, end)

    for d in [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
    ]:
        ny, nx = start[0] + d[0], start[1] + d[1]
        if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
            direction_from[start] = d
            if d[0] != 0:
                first_shape = MoveNeighbors.NS
            else:
                first_shape = MoveNeighbors.EW
            break
    first_piece = TrackPiece(position=start, track_type=first_shape, direction_from=d)
    heapq.heappush(open_set, (0, first_piece))  # (f_score, node, type)

    iteration = 0
    with tqdm(total=cost_map.shape[0] * cost_map.shape[1], desc="Finding Path") as pbar:
        while open_set:
            if iteration % 100 == 0:
                pbar.update(1)
            _, current = heapq.heappop(open_set)

            if current.position == end:
                # Reconstruct path
                path = []
                while current.position in came_from:
                    path.append(current)
                    current = came_from[current.position]
                path.append(TrackPiece(position=start, track_type=first_shape, direction_from=d))
                return path[::-1]

            # Get neighbors based on current track shape and direction
            neighbors = []
            prev_direction = current.direction_from
            current_track_shape = current.track_type

            # Get valid moves based on the current track shape and direction
            valid_moves = get_valid_neighbors(current_track_shape, prev_direction)

            for dy, dx in valid_moves:
                ny, nx = current.position[0] + dy, current.position[1] + dx

                if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
                    # Check altitude constraint: diagonal moves not allowed with altitude difference
                    if is_diagonal_move(dy, dx):
                        altitude_diff = abs(
                            int(cost_map[current.position[0], current.position[1]])
                            - int(cost_map[ny, nx])
                        )
                        if altitude_diff > 0:
                            # Skip this neighbor - can't use diagonal with altitude difference
                            continue

                    neighbors.append((ny, nx, (dy, dx)))
            for neighbor_data in neighbors:
                neighbor_pos, move_direction = neighbor_data[0:2], neighbor_data[2]
                neighbor = neighbor_pos
                
                # Get the direction we came from at the current node
                prev_direction = current.direction_from

                # tentative_g_score is the distance from start to the neighbor through current
                # The cost to move to a neighbor includes altitude difference and momentum
                tentative_g_score = g_score[current.position] + move_cost(
                    cost_map, current.position, neighbor, prev_direction, turn_penalty
                )

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    # Store the direction we're moving in
                    direction_from[neighbor] = move_direction
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    # Determine track shape for the neighbor based on the movement direction
                    next_track_shape = get_track_shape(move_direction)
                    new_piece = TrackPiece(
                        position=neighbor,
                        track_type=next_track_shape,
                        direction_from=move_direction,
                    )
                    if (f_score[neighbor], new_piece) not in open_set:
                        heapq.heappush(
                            open_set, (f_score[neighbor], new_piece)
                        )
            iteration += 1

    return None  # No path found
