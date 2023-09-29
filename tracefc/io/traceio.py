import numpy as np
import pandas as pd
from pathlib import Path

"""This class houses all code for coordinating trace conditioning events with 
imaging/ephys data.
9/4/2023 NK note: Note yet incorporated into this class, will eventually go there."""


class TraceEvents:
    def __init__(self, basedir, event_type):
        pass

    def from_csv(self):
        pass

    def to_epochs(self):
        pass


def load_events_from_csv(csvfile: str):
    """Load events into pandas format and get absolute timestamps."""
    event_header = pd.read_csv(csvfile, header=None, nrows=1)
    assert (event_header[0] == "Start time").all(), "csv file not formatted properly"
    assert (event_header[2] == "microseconds").all(), "csv file not formatted properly"

    start_time = pd.Timestamp(event_header[1][0]) + pd.Timedelta(
        event_header[3][0], unit="microseconds"
    )

    event_df = pd.read_csv(csvfile, header=1)
    event_df["Timestamp"] = start_time + pd.to_timedelta(
        event_df["Time (s)"], unit="sec"
    )

    return event_df


def load_trace_events(
    sesh_dir: str,
    session_type: str
    in [
        "tone_recall",
        "control_tone_recall",
        "ctx_recall",
        "ctx_habituation",
        "tone_habituation",
        "training",
    ],
    event_type=["CS+", "CS-", "shock", "sync_tone", "video", "baseline"],
    return_df: bool = False,
):
    """
    Loads events of a certain type from a session located in the specified session directory.

    """
    assert session_type in [
        "tone_recall",
        "control_tone_recall",
        "ctx_recall",
        "ctx_habituation",
        "tone_habituation",
        "training",
    ]
    assert event_type in ["CS+", "CS-", "shock", "sync_tone", "video", "baseline"]
    sesh_dir = Path(sesh_dir)

    session_type = "habituation" if session_type == "ctx_habituation" else session_type
    # Assemble csv file info into a dataframe.
    csv_files = sorted(sesh_dir.glob("**/" + session_type + "*.csv"))
    event_df_list = []
    for csv_file in csv_files:
        event_df_list.append(load_events_from_csv(csv_file))
    event_df = pd.concat(event_df_list, ignore_index=True)

    # Now parse events
    exclude_str = None
    event_str = "blah"
    if event_type == "CS+":
        assert (
            session_type != "tone_habituation"
        ), "No CS+ in 'tone_habituation' session_type"
        event_str = "CS_end" if session_type == "control_tone_recall" else "CS"

    elif event_type == "CS-":
        assert session_type in [
            "control_tone_recall",
            "tone_habituation",
        ], 'Can only specify "CS-" as event_type for "control_tone_recall" or "tone_habituation" session_type'
        event_str = "CS"
        exclude_str = "CS_end"

    else:
        event_str = event_type

    event_starts = event_df[
        event_df["Event"].str.contains(event_str)
        & event_df["Event"].str.contains("start")
    ]
    event_ends = event_df[
        event_df["Event"].str.contains(event_str)
        & event_df["Event"].str.contains("end")
        & ~event_df["Event"].str.contains("start")
    ]

    if exclude_str is not None:
        event_starts = event_starts[~event_starts["Event"].str.contains(exclude_str)]
        event_ends = event_ends[~event_ends["Event"].str.contains(exclude_str)]
    if not return_df:
        return event_starts, event_ends
    else:
        return event_starts, event_ends, event_df


def trace_ttl_to_openephys(
    trace_cs_df: pd.DataFrame,
    oe_ttls_df: pd.DataFrame,
    ttl_lag=pd.Timedelta(0.33, unit="seconds"),
    trace_ts_key="Timestamp",
    oe_ts_key="datetimes",
    local_time="America/Detroit",
):
    """Finds TTLs in OpenEphys that correspond to CS timestamps recorded from python in a CSV file, assuming a consistent
    time lag from CS start to delivery in OpenEphys

    :param trace_cs_df: dataframe with trace cs times from csv file
    :param oe_ttls_df: dataframe with TTL times in OE file obtained from neuropy.io.openephysio.load_all_ttl_events
    :param trace_ts_key/oe_ts_key: keys in above dataframes where datetime timestamps of each event are logged.
    :param ttl_lag: amount of time OE LAGS the csv in tracefc csv. Enter a negative number if lag is
    positive for some reason."""

    cs_bool = np.zeros(len(oe_ttls_df[oe_ts_key]), dtype=bool)
    event_ind = []
    for ide, event in enumerate(trace_cs_df[trace_ts_key]):
        cs_bool = cs_bool | (
            (oe_ttls_df[oe_ts_key] > (event - ttl_lag))
            & (oe_ttls_df[oe_ts_key] < (event + ttl_lag))
        )
        if (
            sum(
                (
                    (oe_ttls_df[oe_ts_key] > (event - ttl_lag))
                    & (oe_ttls_df[oe_ts_key] < (event + ttl_lag))
                )
            )
            == 1
        ):
            event_ind.append(ide)

    trace_cs_sync_df = oe_ttls_df[cs_bool]

    # Calculate start time difference mean and std to make sure you are getting a consistent lag
    # print(f'cs_bool sum = {cs_bool.sum()}, event_ind={event_ind}')  # For debugging
    start_diff = (
        trace_cs_sync_df[oe_ts_key] - trace_cs_df[trace_ts_key].iloc[event_ind].values
    ).dt.total_seconds()
    if np.isnan(start_diff.mean()):
        print('No matching events found. Try increasing assumed lag in "ttl_lag" param')
    else:
        print(f"start time lag: mean = {start_diff.mean()}, std = {start_diff.std()}")

    # Localize time to recording location in case recorded in different zone (e.g., UTC)
    trace_cs_sync_df["datetimes"] = trace_cs_sync_df["datetimes"].dt.tz_localize(
        local_time
    )

    return trace_cs_sync_df
