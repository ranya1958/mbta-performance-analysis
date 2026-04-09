import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from acquire import get_clean_data
from model import SubwayLine

def update(frame, x_values, actual_values, scheduled_values, actual_line, scheduled_line):
    """update both lines by adding one more day of data"""
    actual_line.set_data(x_values[:frame + 1], actual_values[:frame + 1])
    scheduled_line.set_data(x_values[:frame + 1], scheduled_values[:frame + 1])

    return actual_line, scheduled_line


def main():
    """run animation A using computed fields from SubwayLine"""
    route_id = "Blue"
    route_name = "Blue Line"
    clean_df = get_clean_data(route_id)

    line = SubwayLine(
        route_name=route_name,
        route_id=route_id,
        raw_df=clean_df
     )
    dates = line.dates
    x_values = list(range(len(dates)))
    actual_values = [line.daily_avg_travel[date] for date in dates]
    scheduled_values = [line.daily_avg_scheduled.get(date, float("nan")) for date in dates]
    day_labels = [date[-2:] for date in dates]

    fig, ax = plt.subplots(figsize=(12, 6))

    actual_line, = ax.plot([], [], label="actual travel time")
    scheduled_line, = ax.plot([], [], label="scheduled travel time")

    y_values = []

    for value in actual_values:
        if pd.notna(value):
            y_values.append(value)

    for value in scheduled_values:
        if pd.notna(value):
            y_values.append(value)

    ax.set_xlim(0, len(x_values) - 1)
    ax.set_ylim(min(y_values) * 0.95, max(y_values) * 1.05)

    ax.set_xticks(x_values)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_xlabel("day in February 2026")
    ax.set_ylabel("mean trip time (seconds)")
    ax.set_title(f"{route_name}: actual vs scheduled travel time")
    ax.axvspan(22, 23, alpha=0.2, label="storm window")

    ax.legend()
    ax.grid(alpha=0.3)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(x_values),
        fargs=(x_values, actual_values, scheduled_values, actual_line, scheduled_line),
        interval=500,
        repeat=False
    )

    anim.save(f"mbta_{route_id.lower()}_animation_a.mp4", writer="ffmpeg", fps=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
