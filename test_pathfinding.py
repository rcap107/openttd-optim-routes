import numpy as np
from path_finding import find_path

def load_test_map(filename):
    """Load a test map from a text file where each character is an altitude."""
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Convert each line to a list of integers
    cost_map = np.array([[int(char) for char in line] for line in lines], dtype=np.float32)
    return cost_map

def main():
    # Load the test map
    cost_map = load_test_map('test.txt')
    print("Cost map (altitude):")
    print(cost_map)
    print(f"Map shape: {cost_map.shape}\n")
    
    # Define start and end points
    start = (0, 0)  # Top-left
    end = (4, 4)    # Bottom-right
    
    print(f"Finding path from {start} to {end}...")
    path = find_path(start, end, cost_map, turn_penalty=0.5)
    
    if path:
        print(f"\nPath found with {len(path)} steps:")
        for i, piece in enumerate(path):
            if hasattr(piece, 'position'):
                step = piece.position
                altitude = int(cost_map[step[0], step[1]])
                print(f"  Step {i}: {step} (altitude: {altitude}, track: {piece.track_type})")
            else:
                step = piece
                altitude = int(cost_map[step[0], step[1]])
                print(f"  Step {i}: {step} (altitude: {altitude})")
        
        # Calculate total cost
        total_cost = 0
        for i in range(len(path) - 1):
            current = path[i].position if hasattr(path[i], 'position') else path[i]
            next_pos = path[i + 1].position if hasattr(path[i + 1], 'position') else path[i + 1]
            altitude_diff = abs(int(cost_map[current[0], current[1]]) - int(cost_map[next_pos[0], next_pos[1]]))
            cost = 1 + altitude_diff
            total_cost += cost
            print(f"    Cost from {current} to {next_pos}: {cost} (altitude diff: {altitude_diff})")
        
        print(f"\nTotal cost: {total_cost}")
    else:
        print("\nNo path found!")

if __name__ == "__main__":
    main()
