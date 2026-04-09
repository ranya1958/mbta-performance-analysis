import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from acquire import get_clean_data
from model import SubwayLine


def update(frame, im, full_array, display_array):
    """Update the display array for the current frame and refresh the heatmap."""
    display_array[:, frame] = full_array[:, frame]
    im.set_data(display_array)
    return [im]


def animate_heatmap(line):
    """Create and save an animated heatmap of mean travel times by stop and day."""
    heatmap_df = line.travel_by_stop_and_day
    full_array = heatmap_df.to_numpy(dtype=float)
    display_array = np.full(full_array.shape, np.nan)

    fig, ax = plt.subplots(figsize=(14, 7))

    im = ax.imshow(
        display_array,
        aspect="auto",
        cmap="viridis",
        vmin=np.nanmin(full_array),
        vmax=np.nanmax(full_array)
    )

    ax.set_xticks(range(len(line.dates)))
    ax.set_xticklabels([date[-2:] for date in line.dates], rotation=45)

    ax.set_yticks(range(len(line.stops)))
    ax.set_yticklabels(line.stops)

    ax.set_xlabel("day in February 2026")
    ax.set_ylabel("station")
    ax.set_title(f"{line.route_name}: mean travel time by stop and day")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mean travel time (seconds)")

    anim = FuncAnimation(
        fig,
        update,
        frames=range(full_array.shape[1]),
        fargs=(im, full_array, display_array),
        interval=500,
        repeat=False
    )

    anim.save(f"mbta_{line.route_id.lower()}_animation_b.mp4", fps=2)

    plt.show()


def main():
    route_id = "Blue"
    route_name = "Blue Line"

    clean_df = get_clean_data(route_id)

    line = SubwayLine(
        route_name=route_name,
        route_id=route_id,
        raw_df=clean_df
    )

    animate_heatmap(line)


if __name__ == "__main__":
    main()
