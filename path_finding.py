import heapq
import numpy as np
from tqdm import tqdm

from enum import Enum


class MoveNeighbors(Enum):
    NS = ([(-1, 0), (1, 0), (0, -1), (0, 1)],)  # North-South
    EW = ([(0, -1), (0, 1), (-1, 0), (1, 0)],)  # East-West
    SW = ([(-1, 0), (-1, 1), (-1, 0), (0, 1)],)  # South-West
    SE = ([(1, 0), (1, -1), (1, 1), (0, 1)],)  # South-East
    NW = ([(-1, 0), (-1, -1), (0, -1), (0, -1)],)  # North-West
    NE = ([(1, -1), (1, 1), (1, 0), (0, -1)],)  # North-East


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
    heapq.heappush(open_set, (0, start))  # (f_score, node)

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

    with tqdm(total=cost_map.shape[0] * cost_map.shape[1], desc="Finding Path") as pbar:
        while open_set:
            pbar.update(1)
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Get neighbors
            neighbors = []
            # Four possible directions (up, down, left, right)
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = current[0] + dy, current[1] + dx

                if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
                    neighbors.append((ny, nx))

            for neighbor in neighbors:
                # Get the direction we came from at the current node
                prev_direction = direction_from.get(current, None)

                # tentative_g_score is the distance from start to the neighbor through current
                # The cost to move to a neighbor includes altitude difference and momentum
                tentative_g_score = g_score[current] + move_cost(
                    cost_map, current, neighbor, prev_direction, turn_penalty
                )

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    # Store the direction we're moving in
                    direction_from[neighbor] = (
                        neighbor[0] - current[0],
                        neighbor[1] - current[1],
                    )
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
