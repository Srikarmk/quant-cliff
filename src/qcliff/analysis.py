"""Every number in the report, from the stored rows. Intervals are item bootstraps (stats.py).

The reference for each family is its fp16 build. A level's "drop" is its accuracy minus fp16's on the same
items; its "agreement" is the share of items where it gave the same final answer as fp16 (for extraction,
the same normalised record), which moves before accuracy does: a changed answer can be wrong in a new way
or right by luck. Only the base variant counts toward accuracy; the A/A repeat is the noise floor.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as C
from .stats import mean_ci

TASK_LABEL = {"gsm8k": "GSM8K", "mmlu": "MMLU", "drift": "extraction"}


def load(split: str = "main") -> pd.DataFrame:
    rows = []
    d = os.path.join(C.RESULTS, "answers")
    suffix = "" if split == "main" else f"_{split}"
    for fam, lv in C.all_runs():
        path = os.path.join(d, f"{fam}_{lv}{suffix}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if len(df):
        df["correct"] = df["correct"].astype(bool)
        df["score"] = np.where(df.task == "drift", df.get("fields_correct", 0).fillna(0) / 8, df.correct)
        df["bits"] = df.level.map(C.BITS)
    return df


def levels_of(df: pd.DataFrame, family: str) -> list[str]:
    have = set(df[df.family == family].level)
    return [lv for lv in C.LEVELS if lv in have]


def _ci(values, groups) -> dict:
    return mean_ci(np.asarray(values, dtype=float), np.asarray(groups))


def wide(df: pd.DataFrame, family: str, task: str, variant: str, col: str) -> pd.DataFrame:
    """One row per item, one column per level."""
    s = df[(df.family == family) & (df.task == task) & (df.variant == variant)]
    return s.pivot(index="id", columns="level", values=col)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per family, level, task (and extraction variant): score, its interval, and the paired drop from fp16.
    Score is accuracy for GSM8K and MMLU, and the share of the 8 fields right for extraction."""
    out = []
    for fam in C.FAMILIES:
        for task in C.TASKS:
            for variant in (["base", "reformatted"] if task == "drift" else ["base"]):
                w = wide(df, fam, task, variant, "score")
                if not len(w) or C.REFERENCE not in w:
                    continue
                for lv in levels_of(df, fam):
                    if lv not in w:
                        continue
                    a = _ci(w[lv], np.zeros(len(w)))
                    d = _ci(w[lv] - w[C.REFERENCE], np.zeros(len(w)))
                    out.append({"family": fam, "task": task, "variant": variant, "level": lv, "bits": C.BITS[lv],
                                "score": a["mean"], "lo": a["lo"], "hi": a["hi"], "diff": d["mean"],
                                "diff_lo": d["lo"], "diff_hi": d["hi"], "n": a["n"]})
    return pd.DataFrame(out)


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of items where a level's final answer equals fp16's (unparsed counts as different)."""
    out = []
    for fam in C.FAMILIES:
        for task in C.TASKS:
            w = wide(df, fam, task, "base", "answer")
            if not len(w) or C.REFERENCE not in w:
                continue
            ref = w[C.REFERENCE]
            for lv in levels_of(df, fam):
                if lv not in w:
                    continue
                same = [(a is not None and not pd.isna(a)) and a == r for a, r in zip(w[lv], ref)]
                c = _ci(same, np.zeros(len(same)))
                out.append({"family": fam, "task": task, "level": lv, "agree": c["mean"], "lo": c["lo"],
                            "hi": c["hi"], "n": c["n"]})
    return pd.DataFrame(out)


def aa_table(df: pd.DataFrame) -> pd.DataFrame:
    """The A/A repeat: how many answers changed when the same prompt was sent again (q4_K_M; fp16 in the pilot)."""
    out = []
    for fam in C.FAMILIES:
        for lv in [C.REFERENCE] + C.AA_LEVELS:
            for task in ["gsm8k", "mmlu"]:
                s = df[(df.family == fam) & (df.level == lv) & (df.task == task)]
                if not len(s):
                    continue
                p = s.pivot(index="id", columns="variant", values="answer")
                if "aa" not in p:
                    continue
                changed = int((p["base"].astype(str) != p["aa"].astype(str)).sum())
                out.append({"family": fam, "level": lv, "task": task, "n": len(p), "changed": changed})
    return pd.DataFrame(out)


def cliff_table(acc: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """For each family and task, the highest-precision level whose drop from fp16 is at least `threshold`
    with an interval that excludes zero, i.e. the first level, walking down the ladder, that clearly broke.
    QAT is left out of the walk; it is compared with q4_K_M separately."""
    out = []
    for (fam, task, variant), g in acc.groupby(["family", "task", "variant"]):
        g = g[g.level != "qat"].sort_values("bits", ascending=False)
        broke = g[(g["diff"] <= -threshold) & (g.diff_hi < 0)]
        first = broke.iloc[0] if len(broke) else None
        out.append({"family": fam, "task": task, "variant": variant,
                    "first clear drop": first.level if first is not None else "none",
                    "drop there": first["diff"] if first is not None else np.nan,
                    "lowest level tested": g.iloc[-1].level, "drop at lowest": g.iloc[-1]["diff"]})
    return pd.DataFrame(out)


def validity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per family and level: unparsed and truncated replies, JSON validity on extraction, and output length."""
    out = []
    for (fam, lv), s in df[df.variant != "aa"].groupby(["family", "level"]):
        gm = s[s.task != "drift"]
        dr = s[s.task == "drift"]
        out.append({"family": fam, "level": lv, "bits": C.BITS[lv],
                    "unparsed (maths+MC)": int(gm.answer.isna().sum()),
                    "truncated": int(s.truncated.sum()),
                    "extraction JSON parsed": f"{int(dr.parsed.sum())}/{len(dr)}" if len(dr) else "–",
                    "extraction strict JSON": f"{int(dr.json_strict.sum())}/{len(dr)}" if len(dr) else "–",
                    "mean output tokens (GSM8K)": round(s[s.task == "gsm8k"].output_tokens.mean(), 0)})
    out = pd.DataFrame(out)
    out["order"] = out.level.map({lv: i for i, lv in enumerate(C.LEVELS)})
    return out.sort_values(["family", "order"]).drop(columns="order")


def speed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generation speed (output tokens per second of generation time, median over GSM8K replies) and file
    size. Wall-clock on a shared laptop, so a rough guide, not a benchmark."""
    path = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(path)) if os.path.exists(path) else {}
    out = []
    s = df[(df.task == "gsm8k") & (df.variant == "base") & (df.eval_seconds > 0)]
    for (fam, lv), g in s.groupby(["family", "level"]):
        tps = (g.output_tokens / g.eval_seconds).median()
        size = meta.get(f"{fam}/{lv}", {}).get("size_gb")
        out.append({"family": fam, "level": lv, "tokens/s": round(tps, 1), "size_gb": size})
    return pd.DataFrame(out)
