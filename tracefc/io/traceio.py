import numpy as np
import pandas as pd
from pathlib import Path

import neuropy.io.openephysio as oeio

ustart = "\033[4m"
uend = "\033[0m"

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


class TraceSync:
    """Class to synchronize trace conditioning events to other data, using OE timestamps as a common reference"""

    def __init__(self, basepath: str or Path, read_oe: bool = True):
        self.basepath = Path(basepath)
        self.sync_df = None
        if read_oe:
            print('\033[4mOpenEphys recording times\033[0m')
            self.sync_df = oeio.create_sync_df(self.basepath)
            print('')
        self.cs_oe_df = None
        self.ttl_df = None

        def load_cs():
            if "training" in str(self.basepath):
                # For tone habituation
                csn_starts, csn_stops, csn_df = load_trace_events(self.basepath, session_type="tone_habituation",
                                                                  event_type="CS-", return_df=True)
                self.csn_starts, self.csn_stops, self.csn_df = csn_starts, csn_stops, csn_df
                print(f'\033[4m{csn_starts.shape[0]} CS- events detected\033[0m')
                print(csn_starts.head(6))
                print("")

                # For CS+ during training
                cs_starts, cs_stops, cs_df = load_trace_events(self.basepath, session_type="training",
                                                               event_type="CS+", return_df=True)
                self.cs_starts, self.cs_stops, self.cs_df = cs_starts, cs_stops, cs_df

                print(f'\033[4m{cs_starts.shape[0]} CS+ events detected\033[0m')
                print(cs_starts.head(6))
            elif "recall" in str(self.basepath):
                # For tone recall CS+
                cs_starts, cs_stops, cs_df = load_trace_events(self.basepath, session_type="tone_recall",
                                                               event_type="CS+", return_df=True)
                self.cs_starts, self.cs_stops, self.cs_df = cs_starts, cs_stops, cs_df
                print(f'\033[4m{cs_starts.shape[0]} CS+ events detected\033[0m\n')
                print(cs_starts.head(6))
                print("")

                # For control tone recall CS-
                csn_starts, csn_stops, csn_df = load_trace_events(self.basepath, session_type="control_tone_recall",
                                                                  event_type="CS-", return_df=True)
                self.csn_starts, self.csn_stops, self.csn_df = csn_starts, csn_stops, csn_df
                print(f'\033[4m{csn_starts.shape[0]} CS- events detected\033[0m\n')
                print(csn_starts.head(6))
                print("")

                # For control tone recall CS+(2)
                cs2_starts, cs2_stops, cs2_df = load_trace_events(self.basepath, session_type="control_tone_recall",
                                                                  event_type="CS+", return_df=True)
                self.cs2_starts, self.cs2_stops, self.cs2_df = cs2_starts, cs2_stops, cs2_df
                print(f'\033[4m{cs2_starts.shape[0]} CS+(2) events detected\033[0m')
                print(cs2_starts.head(6))
            elif "habituation" in str(self.basepath):
                # For tone habituation
                csn_starts, csn_stops, csn_df = load_trace_events(self.basepath, session_type="tone_habituation",
                                                                  event_type="CS-", return_df=True)
                self.csn_starts, self.csn_stops, self.csn_df = csn_starts, csn_stops, csn_df
                print(f'\033[4m{csn_starts.shape[0]} CS- events detected\033[0m')
                print(csn_starts.head(6))

        load_cs()

    def load_ttls(self, sanity_check_channel: int = 1, zero_timestamps: bool = True):
        """Import TTLs for CS from OpenEphys"""

        self.ttl_df = oeio.load_all_ttl_events(self.basepath, sanity_check_channel=sanity_check_channel,
                                               zero_timestamps=zero_timestamps)
        print(self.ttl_df[self.ttl_df['channel_states'].abs() == 2].head(5))

    def cs_to_oe(self, cs_name: str in ['cs', 'csn', 'cs2'], cs_ttl_chan: int = 2,
                 ttl_lag_use: pd.Timedelta = pd.Timedelta(0.8, unit="seconds")):
        """Sync CS events in .csv file to OE timestamps"""

        df_list = []
        print(f"{ustart}{cs_name.upper()} lag times{uend}")
        for event in ["starts", "stops"]:
            # Grab CS times corresponding to OE timestamps
            cs_oe_df = trace_ttl_to_openephys(getattr(self, f"{cs_name}_{event}"),
                                                      self.ttl_df[self.ttl_df['channel_states'].abs() == cs_ttl_chan],
                                                      ttl_lag=ttl_lag_use)
            cs_times_eeg = oeio.recording_events_to_combined_time(cs_oe_df, self.sync_df)
            cs_oe_df.insert(loc=cs_oe_df.shape[1], column="eeg_time", value=cs_times_eeg)
            cs_oe_df.insert(loc=cs_oe_df.shape[1], column="label", value=[f"{cs_name}_{event[:-1]}"]*len(cs_times_eeg))
            df_list.append(cs_oe_df)
        df_list

        if isinstance(self.cs_oe_df, pd.DataFrame):
            cs_oe_df = pd.concat(df_list, axis=0).sort_values(by="eeg_time").reset_index()
            df_list = [self.cs_oe_df, cs_oe_df]
        cs_oe_df = pd.concat(df_list, axis=0).sort_values(by="eeg_time").reset_index()

        return cs_oe_df

    def all_cs_to_oe(self, **kwargs):
        """Loads in ALL CS types to oe_cs_df"""
        for cs_type in ["cs", "csn", "cs2"]:
            if hasattr(self, f"{cs_type}_starts"):
                try:
                    self.cs_oe_df = self.cs_to_oe(cs_type, **kwargs)
                except ValueError:
                    print('cs_oe_df already created for all event types - nothing ran')

        return self.cs_oe_df

    def correct_wav_drift(self):
        """Corrects wav file drift if you know the correct start time of a recording"""
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
    return_diff=False,
):
    """Finds TTLs in OpenEphys that correspond to CS timestamps recorded from python in a CSV file, assuming a consistent
    time lag from CS start to delivery in OpenEphys

    :param trace_cs_df: dataframe with trace cs times from csv file
    :param oe_ttls_df: dataframe with TTL times in OE file obtained from neuropy.io.openephysio.load_all_ttl_events
    :param trace_ts_key/oe_ts_key: keys in above dataframes where datetime timestamps of each event are logged.
    :param ttl_lag: amount of time OE LAGS the csv in tracefc csv. Enter a negative number if lag is
    positive for some reason."""

    # loop through each event in trace_cs_df and look for corresponding time in oe_ttls_df
    # way too slow for anything above tens of events.
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
        print(f"TTL to CSV lag: mean = {start_diff.mean()}, std = {start_diff.std()}")

    # Localize time to recording location in case recorded in different zone (e.g., UTC)
    trace_cs_sync_df.loc[:, ("datetimes")] = trace_cs_sync_df["datetimes"].dt.tz_localize(
        local_time
    )

    if return_diff:
        return trace_cs_sync_df, start_diff
    else:
        return trace_cs_sync_df


def grab_usv_folder(basepath: Path, cs_type: str in ['csp', 'csn', 'cs2', 'sync'], ext="WAV"):
    """Locate and return correct .wav or .WAV file for usv detection"""
    try:
        if cs_type == "csp":
            if "recall" in str(basepath):
                wav_file = sorted((basepath / "1_tone_recall").glob(f"**/*.{ext}"))[0]
            elif "training" in str(basepath):
                wav_file = sorted((basepath / "2_training").glob(f"**/*.{ext}"))[0]
        elif cs_type == "csn":
            if "recall" in str(basepath):
                wav_file = sorted((basepath / "2_control_tone_recall").glob(f"**/*.{ext}"))[0]
            elif "training" in str(basepath):
                wav_file = sorted((basepath / "1_tone_habituation").glob(f"**/*.{ext}"))[0]
        elif cs_type == "sync":
            if "recall" in str(basepath):
                wav_file = sorted((basepath / "3_ctx_recall").glob(f"**/*.{ext}"))[0]
            elif "training" in str(basepath):
                wav_file = sorted((basepath / "3_post").glob(f"**/*.{ext}"))[0]
    except IndexError:
        if ext == "wav":
            wav_file = None
        else:
            wav_file = grab_usv_folder(basepath, cs_type, ext="wav")

    return wav_file


if __name__ == "__main__":
    from session_directory import get_session_dir
    animal, sess_name = 'Rose', 'training'
    sess_dir = get_session_dir(animal, sess_name)
    tfc_sync = TraceSync(sess_dir)
    tfc_sync.load_ttls()
    tfc_sync.all_cs_to_oe()