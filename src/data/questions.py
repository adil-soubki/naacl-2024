# -*- coding: utf-8 -*
import os
from glob import glob

import pandas as pd

from ..core.path import dirparent


QS_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "questions")


def load() -> pd.DataFrame:
    ret = []
    for path in glob(os.path.join(QS_DIR, "*")):
        ret.append(pd.read_csv(path))
    return pd.concat(ret)
