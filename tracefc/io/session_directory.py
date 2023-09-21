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
    dir_key = "NatLaptop"
elif os.environ["SHELL"] in ["/bin/sh", "/bin/bash"]:  # Linux Desktop
    base_dir = Path("/data2/Trace_FC/Recording_Rats")
    dir_key = "LNX00004"


def get_session_dir(
    animal: str,
    session: str
    in [
        "habituation1",
        "habituation2",
        "training",
        "recall1",
        "recall2",
        "recall7",
        "recall8",
    ],
    base_dir: pathlib.Path = base_dir,
) -> pathlib.Path:

    assert session.lower() in [
        "habituation1",
        "habituation2",
        "training",
        "recall1",
        "recall2",
        "recall7",
        "recall8",
    ], "Incorrectly specified session"
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    # Import session info
    sesh_df = pd.read_csv(base_dir / "TraceFC_SessionInfo.csv")
    sesh_use = sesh_df[
        (sesh_df["Name"] == animal) & (sesh_df["Session"] == session.capitalize())
    ]

    # This was an attempt to use directory structure for everything, scrapped
    date = str(sesh_use["Date"].iloc[0])
    datestr = "_".join(
        [date.split("_")[id] for id in [2, 0, 1]]
    )  # reorder date to match file structure
    # session_dir = Path(base_dir / animal / f"{datestr}_{session.lower()}")

    session_dir = Path(sesh_use[dir_key].iloc[0]) / f"{datestr}_{session.lower()}"

    return session_dir


if __name__ == "__main__":
    print("test")
