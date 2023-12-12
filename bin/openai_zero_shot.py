#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run OpenAI models zero-shot on our generated questions.

Usage Examples:
    $ openai_zero_shot.py                   # No args needed.
    $ openai_zero_shot.py -o path/to/outdir # Custom outdir.
    $ openai_zero_shot.py --model gpt-4 --temperature 0.75

TODO: Make calls to the API asynchronous.
"""
import datetime
import hashlib
import os
import time
from typing import Any

import backoff
import openai
import pandas as pd
from tqdm import tqdm

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.core import keychain
from src.data import prompts, questions


TEMPLATE = prompts.load("gpt-zero-shot")


def load_questions(window_size: int = 5) -> pd.DataFrame:
    def fn(row: pd.Series) -> pd.Series:
        ctx = row.context.split("\n")
        start, end = max(0, row.sno - 1 - window_size), row.sno + window_size
        assert len(ctx[start:end]) <= (window_size * 2) + 1
        assert ctx[row.sno - 1].endswith("🛑")
        row.context = "\n".join(ctx[start:end])
        return row
    return questions.load().apply(fn, axis=1)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_completion(
    question: pd.Series, model_name: str, temperature: float = 1.0
) -> dict[str, Any]:
    client = openai.OpenAI()
    prompt = TEMPLATE.format(context=question.context, question=question.question)
    result = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=temperature,
    )
    assert len(result.choices) == 1
    return question.to_dict() | {
        "prompt_template": TEMPLATE,
        "prompt": prompt,
        "temperature": temperature,
        "model_name": result.model,
        "timestamp": datetime.datetime.now(),
        "generation": result.choices[0].message.content,
    }


def get_completions(
    model_name: str, temperature: float = 1.0
) -> list[dict[str, Any]]:
    ret = []
    qs = load_questions()
    for _, q in tqdm(list(qs.iterrows())):
        ret.append(get_completion(q, model_name, temperature))
    return ret 


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "zero-shot"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-m", "--model", default="gpt-3.5-turbo")
    ctx.parser.add_argument("-t", "--temperature", type=float, default=1.0)
    args = ctx.parser.parse_args()
    # Generate dialogues asynchronously.
    os.environ["OPENAI_API_KEY"] = keychain.get("IACS")
    completions = pd.DataFrame(get_completions(args.model, args.temperature))
    # Write generations to file.
    model_name = completions[0]["model_name"]
    phash = hashlib.shake_256(TEMPLATE.encode("utf-8")).hexdigest(8)
    outname = datetime.datetime.now().strftime(
        f"{model_name}_{phash}_%Y%m%d.%H%M%S.csv.gz"
    )
    outpath = os.path.join(args.outdir, outname)
    os.makedirs(args.outdir, exist_ok=True)
    completions.to_csv(outpath, index=False, compression="gzip")
    ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
