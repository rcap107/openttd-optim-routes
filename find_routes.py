import cv2
import numpy as np
from path_finding import find_path

# Load the heightmap
heightmap_path = 'data/cropped/heightmap.png'
img = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise IOError(f"Error: Could not load image from {heightmap_path}")

# Define sea level (assuming it's the lowest part of the heightmap)
# This might need adjustment based on the specific map.
sea_level_threshold = 1
sea_mask = img <= sea_level_threshold

# Apply a non-linear transformation (e.g., square root) to emphasize lower altitudes
img_float = img.astype(np.float32)

# Avoid taking sqrt of zero, and handle sea level
img_transformed = np.sqrt(img_float)

# Normalize the transformed image to 0-255 range before quantization
cv2.normalize(img_transformed, img_transformed, 0, 255, cv2.NORM_MINMAX)
img_transformed = img_transformed.astype(np.uint8)

# Quantize the transformed image into a smaller number of altitude levels
num_levels = 30
quantized_img = np.floor(img_transformed / (256 / num_levels)) * (256 / num_levels)
quantized_img = quantized_img.astype(np.uint8)

# Create a color representation of the quantized image
color_quantized = cv2.cvtColor(quantized_img, cv2.COLOR_GRAY2BGR)

# Mask out the sea

color_quantized[sea_mask] = [0, 0, 0]  # Black for sea

# --- Pathfinding ---
# Define a set of start and end points to evaluate
# In a real scenario, these would be your industries
path_candidates = [
((700, 645), (274, 1443))
]

# Use the original heightmap as the cost map
# cost_map = img.transpose()

# Use the quantized image as the cost map for pathfinding
cost_map = quantized_img.transpose()

# Find and rank the paths
paths = []
for start, end in path_candidates:
    path = find_path(start, end, cost_map)
    if path:
        # Calculate path cost (sum of altitude changes)
        cost = 0
        for i in range(len(path) - 1):
            cost += abs(int(cost_map[path[i+1]]) - int(cost_map[path[i]]))
        paths.append((cost, path))

# Sort paths by cost (lower is better)
paths.sort(key=lambda x: x[0])

# Draw the top k paths
k = 2
colors = [[0, 0, 255], [0, 255, 255]] # Red, Yellow
for i in range(min(k, len(paths))):
    cost, path = paths[i]
    print(f"Path {i+1} with cost {cost}")
    for x, y in path:
        color_quantized[y, x] = colors[i]

# Save the output
output_path = 'data/cropped/top_paths.png'
cv2.imwrite(output_path, color_quantized)

print(f"Top paths map saved to {output_path}")
