import heapq
import numpy as np
from tqdm import tqdm


def find_path(start, end, cost_map):
    """
    Finds the best path between a start and end point using the A* algorithm.

    :param start: tuple (y, x) for the start point
    :param end: tuple (y, x) for the end point
    :param cost_map: 2D numpy array where each cell has a cost to traverse
    :return: list of tuples representing the path, or None if no path is found
    """

    # Heuristic function (Taxicab/Manhattan distance)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority queue for open set
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f_score, node)

    # came_from dictionary to reconstruct the path
    came_from = {}

    # g_score: cost from start to the current node
    g_score = {
        (y, x): float("inf")
        for y in range(cost_map.shape[0])
        for x in range(cost_map.shape[1])
    }
    g_score[start] = 0

    # f_score: g_score + heuristic
    f_score = {
        (y, x): float("inf")
        for y in range(cost_map.shape[0])
        for x in range(cost_map.shape[1])
    }
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
                # tentative_g_score is the distance from start to the neighbor through current
                # The cost to move to a neighbor is the altitude difference
                move_cost = abs(
                    int(cost_map[current[0], current[1]])
                    - int(cost_map[neighbor[0], neighbor[1]])
                )
                tentative_g_score = g_score[current] + move_cost

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
