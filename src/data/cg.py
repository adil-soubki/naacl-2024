# -*- coding: utf-8 -*
import itertools
import math
import operator
import os
from glob import glob
from typing import Any

import pandas as pd
from more_itertools import one

from ..core.path import dirparent


CG_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "cg")
CIDS = (4245, 4248, 4310, 4431)
ANNOTATORS = ("Erica", "Lana", "Lee", "Magda")
BELIEFS = ("CT+", "CT-", "NB", "None", "PS")
CG_UPDATES = ("JA", "IN", "RT", "NA")


def is_float(string: str) -> bool:
    try:
        float(string)
    except ValueError:
        return False
    return True


def ann_to_dict(string: str) -> dict[float, str]:
    """
        >>> ann_to_dict("PS 3.5,CT- 3.2,CT- 3.1")
        <<< {3.1: "CT-", 3.2: "CT-", 3.5: "PS"}
    """
    # There are some errors in the annotation file where an event is given with
    # no belief. They appear to be a mistake in the export and are not real
    # annotations.
    if is_float(string):
        return {}
    if string == "None":
        return {}
    ret = {}
    for s in string.split(","):
        ann, event = s.split(" ")
        assert ann in BELIEFS or ann in CG_UPDATES, ann
        ret[float(event)] = ann
    return ret


def diff_dicts(prev: dict[Any, Any], curr: dict[Any, Any]) -> dict[Any, Any]:
    """
        >>> DICT_1 = {1:'donkey', 2:'chicken', 3:'dog'}
        >>> DICT_2 = {1:'donkey', 2:'chimpansee', 4:'chicken'}
        >>> diff_dicts(DICT_1, DICT_2)
        <<< {4: 'chicken', 2: 'chimpansee'}
    """
    prev, curr = set(prev.items()), set(curr.items())
    return dict(curr - prev)


def load(cid: int, annotator: str) -> pd.DataFrame:
    assert cid in CIDS and annotator in ANNOTATORS
    path = one(glob(os.path.join(CG_DIR, f"{cid}*{annotator}*.tsv")))
    cols = ["Sentence", "Eno.", "Event", "Bel(A)", "Bel(B)", "CG(A)", "CG(B)"]
    df = pd.read_table(path, usecols=cols).assign(CID=cid, Annotator=annotator)
    df["Sno."] = df["Eno."].transform(math.floor)
    df["Sentence"] = df["Sentence"].str.strip()
    df["Speaker"] = df["Sentence"].str.split(":").str[0]
    # Normalize NaNs.
    for col in ("Event", "Bel(A)", "Bel(B)", "CG(A)", "CG(B)"):
        df[col] = df[col].str.strip()         # Strip trailing whitespace.
        df[col] = df[col].replace("", "None") # Replace empty lines with "None".
        df[col] = df[col].fillna("None")      # Replace NaN with "None".
    # Clean up annotations for Bel/CG.
    for col in ("Bel(A)", "Bel(B)", "CG(A)", "CG(B)"):
        df[col] = list(map(ann_to_dict, df[col]))
        df[col] = list(itertools.accumulate(df[col], operator.or_))
    # Ensure only the "Sentence" column has NaN values before filling foward.
    for col in set(cols) - {"Sentence"}:
        assert len(df[col]) == len(df[col].dropna(how="any"))
    rcols = ["CID", "Annotator", "Speaker"] + cols[:1] + ["Sno."] + cols[1:]
    return df.ffill()[rcols]


def _load_events(cid: int, annotator: str) -> pd.DataFrame:
    df = load(cid, annotator)
    assert len(df["CID"].unique()) == 1
    ret = []
    # Resolve beliefs based on the the final embedded proposition.
    df_max_eno = df[df.groupby(["Sno."])["Eno."].transform("max") == df["Eno."]]
    for _, row in df_max_eno.iterrows():
        for speaker in ("A", "B"):
            assert len(
                set(row[f"CG({speaker})"].keys()) -
                set(row[f"Bel({speaker})"].keys())
            ) == 0
            for event, belief in row[f"Bel({speaker})"].items():
                ret.append({
                    "cid": row["CID"],
                    "speaker": speaker,
                    "sno": row["Sno."],
                    "event": event,
                    "belief": belief,
                    "cg": row[f"CG({speaker})"].get(event, "NA"),
                })
    ret = pd.DataFrame(ret)
    # Merge speaker beliefs to be side-by-side.
    ret_a = ret[ret.speaker == "A"]
    ret_b = ret[ret.speaker == "B"]
    mcols = ["event", "sno", "belief", "cg"]
    return ret_a[mcols].merge(
        ret_b[mcols],
        how="left" if len(ret_a) > len(ret_b) else "right",
        on=["sno", "event"],
        suffixes=("_A", "_B")
    ).rename(columns={"event": "eno"}).merge(
        df.rename(
            columns={"Eno.": "eno", "Event": "event", "CID": "cid"}
        )[["eno", "event", "cid"]], how="left", on="eno"
    )


def load_events(cid: int, annotator: str) -> pd.DataFrame:
    """
    This adds rows before the event happens that have belief_A and belief_B
    as NB. It also considers cg_A and cg_B to be NA (no annotation).
    """
    ret = []
    df = _load_events(cid, annotator)
    for eno, data in df.groupby("eno"):
        missing = []
        present = df[df.eno == eno]
        event = one(present.event.unique())
        cid = one(present.cid.unique())
        missing_snos = sorted(
            set(range(1, df.sno.max() + 1)) - set(present.sno.unique())
        )
        for sno in missing_snos:
            missing.append({
                "eno": eno,
                "sno": sno,
                "belief_A": "NB",
                "belief_B": "NB",
                "cg_A": "NA",
                "cg_B": "NA",
                "event": event,
                "cid": cid,
            })
        ret.append(pd.concat([pd.DataFrame(missing), present]).sort_values("sno"))
    return pd.concat(ret).reset_index(drop=True).assign(annotator=annotator)
