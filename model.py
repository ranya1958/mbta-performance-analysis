import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

BLUE_LINE_STOPS = [
    "place-wondl",
    "place-rbmnl",
    "place-bmmnl",
    "place-sdmnl",
    "place-orhte",
    "place-wimnl",
    "place-aport",
    "place-mvbcl",
    "place-aqucl",
    "place-state",
    "place-gover",
    "place-bomnl",
]


class SubwayLine(BaseModel):
    route_name: str
    route_id: str
    raw_df: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def stops(self) -> list[str]:
        """Ordered list of station names along the line."""
        if self.route_id == "Blue":
            available_stops = set(self.raw_df["parent_station"].dropna().unique())
            return [stop for stop in BLUE_LINE_STOPS if stop in available_stops]

        return (
            self.raw_df["parent_station"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    @computed_field
    @property
    def dates(self) -> list[str]:
        """Sorted list of service dates in February as strings."""
        return sorted(self.raw_df["service_date"].dropna().unique().tolist())

    @computed_field
    @property
    def daily_avg_travel(self) -> dict[str, float]:
        """Date string → mean actual travel time (seconds) across all trips."""
        trip_df = (
            self.raw_df.groupby(["service_date", "trip_id"])["travel_time_seconds"]
            .sum()
            .reset_index(name="actual_trip_time")
        )

        daily_df = (
            trip_df.groupby("service_date")["actual_trip_time"]
            .mean()
            .sort_index()
        )

        return daily_df.to_dict()

    @computed_field
    @property
    def daily_avg_scheduled(self) -> dict[str, float]:
        """Date string → mean scheduled travel time (seconds) across all trips."""
        sched_df = self.raw_df.dropna(subset=["scheduled_travel_time"]).copy()

        trip_df = (
            sched_df.groupby(["service_date", "trip_id"])["scheduled_travel_time"]
            .sum()
            .reset_index(name="scheduled_trip_time")
        )

        daily_df = (
            trip_df.groupby("service_date")["scheduled_trip_time"]
            .mean()
            .sort_index()
        )

        return daily_df.to_dict()

    @computed_field
    @property
    def travel_by_stop_and_day(self) -> pd.DataFrame:
        """
        Produce a summary-like table by stop and day where each value is
        mean travel time in seconds.
        """
        stop_day_df = (
            self.raw_df.dropna(subset=["parent_station", "service_date", "travel_time_seconds"])
            .groupby(["parent_station", "service_date"])["travel_time_seconds"]
            .mean()
            .reset_index()
        )

        pivot_df = stop_day_df.pivot(
            index="parent_station",
            columns="service_date",
            values="travel_time_seconds"
        )

        pivot_df = pivot_df.reindex(index=self.stops, columns=self.dates)

        return pivot_df