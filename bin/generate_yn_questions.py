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

        Does {spkr1} believe that
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

        Does {spkr1} believe that {spkr2} believes
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


def generate_yn_questions_for_row(df: pd.Series):
    ret = []
    qmap = {
        "CT-": "certainly not",
        "CT+": "certainly",
        "PS": "possibly",
    }
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
            "question": f"Does {spkr} believe it is {qmap[bel]} true that {df.event}?",
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
                f"Does {spkr1} believe that {spkr2} believes "
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
                f"Does {spkr1} believe that {spkr2} believes "
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


def filter_to_interesting_events(df: pd.DataFrame):
    ret = []
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
            data[data.sno.isin(ends)].assign(context_type="end"),
            data[data.sno.isin(mids)].assign(context_type="mid")
        ]).sort_values("sno")
        ret.append(curr)
    return pd.concat(ret)


def generate_yn_questions(cid: int, annotator: str):
    ret = []
    events = filter_to_interesting_events(cg.load_events(cid, annotator))
    for _, row in events.iterrows():
        ret.append(generate_yn_questions_for_row(row))
    return pd.concat(ret).assign(cid=cid, annotator=annotator)


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "questions"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-a", "--annotator", default="Magda")
    args = ctx.parser.parse_args()
    # Generate questions.
    os.makedirs(args.outdir, exist_ok=True)
    for cid in cg.CIDS:
        generate_yn_questions(cid, args.annotator).to_csv(outpath := os.path.join(
            args.outdir, f"{cid}_{args.annotator}_yn_questions.csv"
        ), index=False)
        ctx.log.info("wrote: %s", outpath)
    ctx.log.info("example output:\n%s", pd.read_csv(outpath))


if __name__ == "__main__":
    harness(main)
