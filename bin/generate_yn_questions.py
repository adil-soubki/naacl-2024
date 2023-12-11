#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates yes/no questions from CG annotations.

Usage Examples:
    $ generate_yn_questions.py # No args needed.
    $ generate_yn_questions.py --outdir path/to/outdir
"""
import itertools
import os

import pandas as pd

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import cg


def resolve_1st_order_yn_answer(qbel, sbel):
    if qbel == sbel:
        return True
    elif qbel == "PS" and sbel == "CT+":
        return True
    else:
        return False


# XXX: I feel like something is wrong here... case where question is positive
#      polarity but the CG is RT/NA don't they also know hearer belief?
# XXX: It matters if the question is about CT- or CT+/PS...
def resolve_2nd_order_yn_answer(qbel, sbel1, sbel2, cg1, cg2):
    """
    This method resolves the truthiness of a second order belief question with
    the following form based on the annotations provided.

        Is it the case that {spkr1} believes that
        {spkr2} believes that it is {qbel} true that {event}?

    Args:
        qbel (str): question belief (CT-, PS, CT+).
        sbel1 (str): speaker 1 belief (CT-, PS, CT+, NB).
        sbel2 (str): speaker 2 belief (CT-, PS, CT+, NB).
        cg1 (str): speaker 1 common ground (JA, IN, RT, NA).
        cg2 (str): speaker 2 common ground (JA, IN, RT, NA).

    Returns:
        bool: True if the question is correct given the annotations.
    """
    # Case 1: The question is positive polarity.
    if qbel in ("PS", "CT+") and cg1 in ("JA", "IN") and (
        (qbel == sbel1) or ((qbel == "PS") and (sbel1 == "CT+"))
    ):
        return True
    # Case 2: The question is negative polarity.
    elif qbel == "CT-" and cg1 == "RT" and qbel == sbel2: # XXX: sbel2 here!!!
        return True
    return False


# XXX: I feel like something is wrong here... case where question is positive
#      polarity but the CG is RT/NA don't they also know hearer belief?
# XXX: It matters if the question is about CT- or CT+/PS...
def resolve_3rd_order_yn_answer(qbel, sbel1, sbel2, cg1, cg2):
    """
    This method resolves the truthiness of a third order belief question with
    the following form based on the annotations provided.

        Is it the case that {spkr1} believes that {spkr2} believes
        that {spkr1} believes it is {qbel} true that {event}?

    Args:
        qbel (str): question belief (CT-, PS, CT+).
        sbel1 (str): speaker 1 belief (CT-, PS, CT+, NB).
        sbel2 (str): speaker 2 belief (CT-, PS, CT+, NB).
        cg1 (str): speaker 1 common ground (JA, IN, RT, NA).
        cg2 (str): speaker 2 common ground (JA, IN, RT, NA).

    Returns:
        bool: True if the question is correct given the annotations.
    """
    if qbel in ("PS", "CT+") and cg1 in ("JA", "IN") and (
        (qbel == sbel1) or ((qbel == "PS") and (sbel1 == "CT+"))
    ):
        return True
    elif qbel == "CT-" and cg1 in ("RT", "NA") and qbel == sbel1: # XXX: sbel1 here!!!
        return True
    return False


# XXX: A lot of repeated code here.
def generate_yn_questions_for_row(df: pd.Series):
    ret = []
    qmap = {
        "CT-": "certainly not",
        "CT+": "certainly",
        "PS": "possibly",
    }
    prefix = "At the time indicated by 🛑, is it the case that"
    # First order questions.
    for spkr, bel in itertools.product(("A", "B"), qmap.keys()):
        ret.append({
            "sno": df.sno,
            "eno": df.eno,
            "belief_A": df.belief_A,
            "belief_B": df.belief_B,
            "belief_Q": bel,
            "cg_A": df.cg_A,
            "cg_B": df.cg_B,
            "order": 1,
            "question": f"{prefix} {spkr} believes it is {qmap[bel]} true that {df.event}?",
            "answer": "Yes" if resolve_1st_order_yn_answer(bel, df[f"belief_{spkr}"]) else "No",
            "context_type": df.context_type,
        })
    # Second order questions.
    for spkrs, bel in itertools.product((("A", "B"), ("B", "A")), qmap.keys()):
        spkr1, spkr2 = spkrs
        ret.append({
            "sno": df.sno,
            "eno": df.eno,
            "belief_A": df.belief_A,
            "belief_B": df.belief_B,
            "belief_Q": bel,
            "cg_A": df.cg_A,
            "cg_B": df.cg_B,
            "order": 2,
            "question": (
                f"{prefix} {spkr1} believes that {spkr2} believes "
                f"it is {qmap[bel]} true that {df.event}?"
            ),
            "answer": "Yes" if resolve_2nd_order_yn_answer(
                bel,
                df[f"belief_{spkr1}"], df[f"belief_{spkr2}"],
                df[f"cg_{spkr1}"], df[f"cg_{spkr2}"],
            ) else "No",
            "context_type": df.context_type,
        })
    # Third order questions.
    for spkrs, bel in itertools.product((("A", "B", "A"), ("B", "A", "B")), qmap.keys()):
        spkr1, spkr2, spkr3 = spkrs
        ret.append({
            "sno": df.sno,
            "eno": df.eno,
            "belief_A": df.belief_A,
            "belief_B": df.belief_B,
            "belief_Q": bel,
            "cg_A": df.cg_A,
            "cg_B": df.cg_B,
            "order": 3,
            "question": (
                f"{prefix} {spkr1} believes that {spkr2} believes "
                f"that {spkr3} believes it is {qmap[bel]} true that {df.event}?"
            ),
            "answer": "Yes" if resolve_3rd_order_yn_answer(
                bel,
                df[f"belief_{spkr1}"], df[f"belief_{spkr2}"],
                df[f"cg_{spkr1}"], df[f"cg_{spkr2}"],
            ) else "No",
            "context_type": df.context_type,
        })
    return pd.DataFrame(ret)


# XXX: This is dumb.
def load_context_for(cid: int, annotator: str, sno: int) -> str:
    df = cg.load(cid, annotator)[
        ["Sno.", "Sentence"]
    ].drop_duplicates().set_index("Sno.")
    df.loc[sno, "Sentence"] = f"{df.loc[sno, 'Sentence']} 🛑"
    return "\n".join(df.Sentence)


# XXX: This is dumb.
def load_contexts(cid: int, annotator: str) -> pd.DataFrame:
    ret = []
    for sno in sorted(cg.load(cid, annotator)["Sno."].unique()):
        ret.append({
            "sno": sno,
            "context": load_context_for(cid, annotator, sno),
        })
    return pd.DataFrame(ret)


# Filters speech act events from questions (e.g. A asks ...)
def filter_out_speech_act_events(df: pd.DataFrame) -> pd.DataFrame:
    assert len(df.cid.unique()) == 1
    speech_act_enos = []
    events = df[["eno", "event"]].drop_duplicates()
    events = events.assign(sno=list(map(int, events.eno)))[["sno", "eno", "event"]]
    events = events.sort_values("eno")
    for sno, data in events.groupby("sno"):
        for left, right in itertools.pairwise(data.to_dict("records")):
            if not left["event"].endswith(right["event"]):
                continue
            pretkns = left["event"].replace(right["event"], "").split(" ")
            if pretkns[1] in ("asks", "jokes"):
                speech_act_enos.append(left["eno"])
            elif pretkns[1] == "said" and pretkns[2] == "that":
                speech_act_enos.append(left["eno"])
    return df[~df["eno"].isin(speech_act_enos)]


def filter_to_interesting_events(df: pd.DataFrame) -> pd.DataFrame:
    ret = []
    df = filter_out_speech_act_events(df)
    min_sno, max_sno = df.sno.min(), df.sno.max()
    for eno, data in df.groupby("eno"):
        updates = data.drop_duplicates(
            ["eno", "belief_A", "belief_B"],
            keep="first"
        ).sort_values("sno")
        ends = sorted({min_sno} | set(updates.sno) | {max_sno})
        mids = list(map(lambda x: (x[0] + x[1]) // 2, itertools.pairwise(ends)))
        mids = sorted(set(mids) - set(ends))
        curr = pd.concat([
            data[data.sno.isin(ends[1:-1])].assign(context_type="end"),
            data[data.sno.isin(mids)].assign(context_type="mid")
        ]).sort_values("sno")
        ret.append(curr[curr.context_type == "end"])
    return pd.concat(ret)


def filter_to_interesting_questions(df: pd.DataFrame) -> pd.DataFrame:
    ret = []
    for grp, data in df.groupby(["belief_A", "belief_B", "cg_A", "cg_B"]):
        if grp == ("CT+", "CT+", "JA", "JA") or grp == ("NB", "NB", "NA", "NA"):
            ret.append(data.sample(frac=0.1, replace=False, random_state=42))
        else:
            ret.append(data)
    return pd.concat(ret).reset_index(drop=True)


def generate_yn_questions(cid: int, annotator: str) -> pd.DataFrame:
    ret = []
    events = filter_to_interesting_events(cg.load_events(cid, annotator))
    for _, row in events.iterrows():
        ret.append(generate_yn_questions_for_row(row))
    return filter_to_interesting_questions(
        pd.concat(ret).assign(cid=cid, annotator=annotator).merge(
            load_contexts(cid, annotator), how="left", on="sno"
        )
    )


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "questions"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-a", "--annotator", default="Magda")
    args = ctx.parser.parse_args()
    # Generate questions.
    qs = []
    os.makedirs(args.outdir, exist_ok=True)
    for cid in cg.CIDS:
        (q := generate_yn_questions(cid, args.annotator)).to_csv(outpath := os.path.join(
            args.outdir, f"{cid}_{args.annotator}_yn_questions.csv.gz"
        ), index=False, compression="gzip")
        qs.append(q)
        ctx.log.info("wrote: %s", outpath)
    qs = pd.concat(qs).reset_index(drop=True)
    # Summarize questions.
    ctx.log.info("example output:\n%s", qs)
    summary = qs.value_counts(["belief_A", "belief_B", "cg_A", "cg_B"])
    summary = pd.concat([
        summary, (summary / summary.sum()).to_frame("percent")
    ], axis=1)
    summary.loc[("", "", "", "total")] = summary.sum()
    summary = summary.reset_index()
    ctx.log.info("summary:\n%s", summary.to_string(index=False))
    ans = qs.value_counts("answer")
    ans = pd.concat([ans, (ans / ans.sum()).to_frame("percent")], axis=1)
    ctx.log.info("answers:\n%s", ans.reset_index().to_string(index=False))


if __name__ == "__main__":
    harness(main)
