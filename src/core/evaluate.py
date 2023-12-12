# -*- coding: utf-8 -*
import collections
from typing import Any

import evaluate
import pandas as pd


def compute_one(
    preds: list[str],
    refs: list[str],
    metric: str,
    labels: list[str]
) -> dict[str, Any]:
    mtr = evaluate.load(metric)
    rslts = mtr.compute(predictions=preds, references=refs, average=None)[metric]
    macro = mtr.compute(predictions=preds, references=refs, average="macro")[metric]
    micro = mtr.compute(predictions=preds, references=refs, average="micro")[metric]
    return collections.OrderedDict(
        **{"macro": macro, "micro": micro},
        **{labels[idx]: rslt for idx, rslt in enumerate(rslts)},
    )


def compute(
    preds: list[str],
    refs: list[str],
    metrics: list[str],
    labels: list[str]
) -> dict[str, Any]:
    assert len(preds) == len(refs)
    return {mtr: compute_one(preds, refs, mtr, labels) for mtr in metrics}


#  def from_csv(path: str, ref_column: str, pred_column: str) -> dict[str, Any]:
#      df = pd.read_csv(path, keep_default_na=False)
#      df[pred_column] = df[pred_column].str.replace(".", "")
#      metrics = ["f1", "precision", "recall"]
#      labels = sorted(list(set(df[pred_column]) | set(df[ref_column])))
#      preds = list(map(lambda lbl: labels.index(lbl), df[pred_column]))
#      refs = list(map(lambda lbl: labels.index(lbl), df[ref_column]))
#      return compute(preds, refs, metrics, labels)
