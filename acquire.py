from pathlib import Path
import pandas as pd


BASE_URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/{}-subway-on-time-performance-v1.parquet"
CACHE_FOLDER = Path("cache")
CACHE_FOLDER.mkdir(exist_ok=True)


def get_feb_dates():
    """return all dates in February 2026"""
    dates = []

    for day in range(1, 29):
        dates.append(f"2026-02-{day:02d}")

    return dates


def fetch_one_day(date_str, route_id):
    """fetch one day of data and filter to one subway line"""
    url = BASE_URL.format(date_str)
    df = pd.read_parquet(url)

    df = df[df["trunk_route_id"] == route_id].copy()

    return df


def fetch_month(route_id="Blue"):
    """fetch all of February 2026 and use a local cache if it exists"""
    cache_file = CACHE_FOLDER / f"{route_id.lower()}_feb_2026_raw.parquet"

    if cache_file.exists():
        print("loading cached file...")
        return pd.read_parquet(cache_file)

    frames = []

    for date_str in get_feb_dates():
        print(f"fetching {date_str}...")
        day_df = fetch_one_day(date_str, route_id)
        frames.append(day_df)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache_file, index=False)
    print(f"saved raw cache to {cache_file}")

    return df


def clean_data(df):
    """clean and deduplicate the MBTA data for analysis"""
    df = df.copy()

    df["service_date"] = pd.to_datetime(
        df["service_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    df["stop_timestamp"] = pd.to_datetime(
        df["stop_timestamp"],
        unit="s",
        errors="coerce"
    )

    df = df.sort_values(["service_date", "trip_id", "stop_id", "stop_timestamp"])

    df = df.drop_duplicates(
        subset=["service_date", "trip_id", "stop_id"],
        keep="first"
    )

    df["travel_time_seconds"] = pd.to_numeric(df["travel_time_seconds"], errors="coerce")
    df["scheduled_travel_time"] = pd.to_numeric(df["scheduled_travel_time"], errors="coerce")

    df = df.dropna(subset=["service_date", "trip_id", "stop_id", "travel_time_seconds"])

    df["service_date"] = df["service_date"].dt.strftime("%Y-%m-%d")

    return df


def make_actual_trip_times(df):
    """sum stop-level scheduled times into full trip durations"""
    trip_df = (
        df.groupby(["service_date", "trip_id"])["travel_time_seconds"]
        .sum()
        .reset_index()
    )

    trip_df = trip_df.rename(columns={"travel_time_seconds": "actual_trip_time"})

    return trip_df


def make_scheduled_trip_times(df):
    """make one row per day with average actual and scheduled trip time"""
    sched_df = df.dropna(subset=["scheduled_travel_time"]).copy()

    trip_df = (
        sched_df.groupby(["service_date", "trip_id"])["scheduled_travel_time"]
        .sum()
        .reset_index()
    )

    trip_df = trip_df.rename(columns={"scheduled_travel_time": "scheduled_trip_time"})

    return trip_df


def make_daily_summary(df):
    actual_trip_df = make_actual_trip_times(df)
    sched_trip_df = make_scheduled_trip_times(df)

    actual_daily = (
        actual_trip_df.groupby("service_date")["actual_trip_time"]
        .mean()
        .reset_index()
    )
    actual_daily = actual_daily.rename(columns={"actual_trip_time": "avg_actual_trip_time"})

    sched_daily = (
        sched_trip_df.groupby("service_date")["scheduled_trip_time"]
        .mean()
        .reset_index()
    )
    sched_daily = sched_daily.rename(columns={"scheduled_trip_time": "avg_scheduled_trip_time"})

    daily_df = actual_daily.merge(sched_daily, on="service_date", how="left")
    daily_df = daily_df.sort_values("service_date").reset_index(drop=True)

    return daily_df


def get_clean_data(route_id="Blue"):
    """fetch and clean one month of data for a chosen line"""
    raw_df = fetch_month(route_id)
    clean_df = clean_data(raw_df)
    return clean_df


if __name__ == "__main__":
    blue_df = get_clean_data("Blue")
    summary_df = make_daily_summary(blue_df)

    print("\ncleaned data:")
    print(blue_df[["service_date", "trip_id", "stop_id", "parent_station", "travel_time_seconds"]].head())

    print("\ndaily summary:")
    print(summary_df.head())
