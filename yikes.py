import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cv2
    from path_finding import find_path
    import matplotlib.pyplot as plt
    import numpy as np
    from wigglystuff import ChartPuck
    return ChartPuck, cv2, find_path, mo, np, plt


@app.cell
def _(cv2, np, plt):
    # Load the images
    heightmap_path = 'data/hm.png'
    quantized_map_path = 'data/cropped/quantized_map.png'

    img = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
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

    cost_map = quantized_img

    fig, ax = plt.subplots()
    ax.imshow(cost_map)
    # plt.close(fig)
    return color_quantized, cost_map, fig, quantized_img


@app.cell
def _(ChartPuck, fig, plt):
    multi_puck = ChartPuck(
        fig,
        x=[500, 500],
        y=[250, 2000],
        puck_color="#2196f3",
    )
    plt.close(fig)
    return (multi_puck,)


@app.cell
def _(mo, multi_puck):
    multi_widget = mo.ui.anywidget(multi_puck)
    return (multi_widget,)


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell
def _(mo, multi_widget):
    positions = [
        f"({x:.0f}, {y:.0f})"
        for i, (x, y) in enumerate(zip(multi_widget.x, multi_widget.y))
    ]
    mo.callout("\n".join(positions))
    return


@app.cell
def _(multi_widget):
    start, end = zip(multi_widget.x, multi_widget.y)
    return end, start


@app.cell
def _(start):
    tuple(int(_) for _ in start)
    return


@app.cell
def _(cost_map, end, find_path, mo, start):
    start_int = tuple(int(_) for _ in start)
    end_int = tuple(int(_) for _ in end)

    # Run pathfinding when both points are selected
    with mo.status.spinner("Finding path..."):
        path = find_path(start_int, end_int, cost_map)
    return (path,)


@app.cell
def _(color_quantized, mo, path, quantized_img):
    if path:
        # Draw the path on the map
        k=2
        path_map = quantized_img.copy()
        for x, y in [_.position for _ in path]:
            color_quantized[y, x] = 255
    else:
        path_display = mo.md("### No path found.")
    return (y,)


@app.cell
def _(color_quantized, plt):
    plt.imshow(color_quantized)
    return


@app.cell
def _(y):
    y
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
