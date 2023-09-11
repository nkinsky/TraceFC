import pandas as pd
from pathlib import Path


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
