"""
Functions to easily grab trace FC data across multiple computers
"""
import pathlib

import pandas as pd
import os
from pathlib import Path

# Use shell to figure out computer
if os.environ["SHELL"] == "/bin/zsh":  # Laptop
    base_dir = Path("/Users/nkinsky/Documents/UM/Working/Trace_FC/Recording_Rats")
elif os.environ["SHELL"] == "/bin/sh":  # Linux Desktop
    base_dir = Path("/data2/Trace_FC/Recording_Rats")


def get_session_dir(
    animal: str,
    session: str
    in ["habituation1", "habituation2", "training", "recall1", "recall2", "recall7"],
    base_dir: pathlib.Path = base_dir,
) -> pathlib.Path:
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    # Import session info
    sesh_df = pd.read_csv(base_dir / "TraceFC_SessionInfo.csv")
    sesh_use = sesh_df[
        (sesh_df["Name"] == animal) & (sesh_df["Session"] == session.capitalize())
    ]
    date = str(sesh_use["Date"].iloc[0])
    datestr = "_".join(
        [date.split("_")[id] for id in [2, 0, 1]]
    )  # reorder date to match file structure
    session_dir = Path(base_dir / animal / f"{datestr}_{session.lower()}")

    return session_dir
