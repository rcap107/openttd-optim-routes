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
    return ChartPuck, cv2, find_path, mo, plt


@app.cell
def _(cv2, plt):
    # Load the images
    heightmap_path = 'data/cropped/heightmap.png'
    quantized_map_path = 'data/cropped/quantized_map.png'

    heightmap = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
    quantized_map = cv2.imread(quantized_map_path)

    fig, ax = plt.subplots()
    ax.imshow(quantized_map)
    plt.close(fig)
    return fig, heightmap, quantized_map


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
        f"Puck {i+1}: ({x:.2f}, {y:.2f})"
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
def _(cv2, end, find_path, heightmap, mo, quantized_map, start):
    start_int = tuple(int(_) for _ in start)
    end_int = tuple(int(_) for _ in end)

    # Run pathfinding when both points are selected
    with mo.status.spinner("Finding path..."):
        path = find_path(start_int, end_int, heightmap)

    if path:
        # Draw the path on the map
        path_map = quantized_map.copy()
        for y, x in path:
            path_map[y, x] = [0, 0, 255] # Red path

        # Display the map with the path
        path_display = mo.vstack([
            mo.md("### Path found!"),
            mo.ui.image(src=cv2.imencode('.png', path_map)[1].tobytes())
        ])
    else:
        path_display = mo.md("### No path found.")
    return (path,)


@app.cell
def _(path):
    path is None
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
