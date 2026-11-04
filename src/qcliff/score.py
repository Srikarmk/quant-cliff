"""Score the 13 predictions in HYPOTHESES.md mechanically, from the same tables as the report.

Written against the pilot split before any main-split result existed, so each pass rule is fixed before
the numbers it judges. Readings HYPOTHESES.md leaves open:

- "Within ±x pp" is |drop| <= x on the point estimate; "no worse than −x" is drop >= −x.
- "Every task" means GSM8K, MMLU and extraction from the base source; the reformatted source is scored
  only where a prediction names it (6).
- Prediction 7's "lower at q2_K than q4_K_M than q8_0" is strict on the point estimates.
- A prediction with several parts holds only if every part holds; "partly" if some do; a part with no
  data is "untestable".
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import analysis as A
from . import config as C
from .report import LABEL, md_table

LADDER = ["llama3.1-8b", "qwen2.5-7b"]
TASKS3 = ["gsm8k", "mmlu", "drift"]


def _pp(x) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:+.1f} pp"


def _pc(x) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:.1f}%"


def _part(desc, value, ok) -> dict:
    return {"desc": desc, "value": value, "ok": None if ok is None else bool(ok)}


class T:
    def __init__(self, df):
        self.df = df
        self.acc = A.accuracy_table(df)
        self.ag = A.agreement_table(df)

    def d(self, fam, task, lv, variant="base"):
        r = self.acc[(self.acc.family == fam) & (self.acc.task == task) & (self.acc.level == lv) &
                     (self.acc.variant == variant)]
        return r.iloc[0] if len(r) else None

    def agree(self, fam, task, lv):
        r = self.ag[(self.ag.family == fam) & (self.ag.task == task) & (self.ag.level == lv)]
        return r.iloc[0].agree if len(r) else np.nan


def _drop_part(t, fam, task, lv, rule, desc, variant="base", need_ci=False):
    r = t.d(fam, task, lv, variant)
    if r is None:
        return _part(desc, "–", None)
    ok = rule(r["diff"]) and (not need_ci or r.diff_lo <= 0 <= r.diff_hi)
    return _part(desc, f"{_pp(r['diff'])} [{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}]", ok)


def predictions(t: T) -> list[tuple[int, str, list[dict]]]:
    P = []
    fams = [f for f in C.FAMILIES if f in set(t.df.family)]

    P.append((1, "q8_0 is free", [
        _drop_part(t, f, task, "q8_0", lambda x: abs(x) <= 0.02, f"{LABEL[f]} {A.TASK_LABEL[task]}: |drop| ≤ 2 pp, CI includes 0",
                   need_ci=True) for f in fams for task in TASKS3]))

    P.append((2, "q6_K and q5_K_M are free", [
        _drop_part(t, f, task, lv, lambda x: abs(x) <= 0.03, f"{LABEL[f]} {A.TASK_LABEL[task]} {lv}: |drop| ≤ 3 pp")
        for f in LADDER for lv in ["q6_K", "q5_K_M"] for task in TASKS3]))

    P.append((3, "Ollama's default costs little", [
        _drop_part(t, f, task, "q4_K_M", lambda x: x >= -0.04, f"{LABEL[f]} {A.TASK_LABEL[task]}: drop ≥ −4 pp")
        for f in fams for task in TASKS3]))

    parts = []
    for f in LADDER:
        clear = []
        for lv in ["q8_0", "q6_K", "q5_K_M", "q4_K_M"]:
            r = t.d(f, "gsm8k", lv)
            if r is not None and r["diff"] <= -0.05 and r.diff_hi < 0:
                clear.append(lv)
        parts.append(_part(f"{LABEL[f]} GSM8K: no clear drop above q3_K_M", ", ".join(clear) or "none", not clear))
    ll, qw = t.d("llama3.1-8b", "gsm8k", "q2_K"), t.d("qwen2.5-7b", "gsm8k", "q2_K")
    if ll is not None and qw is not None:
        parts.append(_part("Llama GSM8K q2_K drop ≥ 5 pp", _pp(ll["diff"]), ll["diff"] <= -0.05))
        parts.append(_part("Llama's q2_K GSM8K drop larger than Qwen2.5's",
                           f"Llama {_pp(ll['diff'])}, Qwen2.5 {_pp(qw['diff'])}", ll["diff"] < qw["diff"]))
    P.append((4, "The cliff is at 3 bits or below", parts))

    parts = []
    for f in LADDER:
        g, m = t.d(f, "gsm8k", "q2_K"), t.d(f, "mmlu", "q2_K")
        if g is not None and m is not None:
            parts.append(_part(f"{LABEL[f]} q2_K: GSM8K drop larger than MMLU drop",
                               f"GSM8K {_pp(g['diff'])}, MMLU {_pp(m['diff'])}", g["diff"] < m["diff"]))
    P.append((5, "Maths breaks before multiple choice", parts))

    parts = []
    for f in LADDER:
        b, r = t.d(f, "drift", "q2_K", "base"), t.d(f, "drift", "q2_K", "reformatted")
        if b is not None and r is not None:
            parts.append(_part(f"{LABEL[f]} q2_K base extraction: |drop| ≤ 2 pp", _pp(b["diff"]), abs(b["diff"]) <= 0.02))
            parts.append(_part(f"{LABEL[f]} q2_K: reformatted loses more than base",
                               f"reformatted {_pp(r['diff'])}, base {_pp(b['diff'])}", r["diff"] < b["diff"]))
    P.append((6, "Extraction holds longest", parts))

    below = [f for f in fams if any(t.agree(f, task, "q8_0") < 1 for task in TASKS3)]
    parts = [_part("q8_0 agreement below 100% on some task for ≥ 3 of 4 models",
                   f"{len(below)} of {len(fams)}: {', '.join(LABEL[f] for f in below)}", len(below) >= 3)]
    for f in LADDER:
        for task in ["gsm8k", "mmlu"]:
            a8, a4, a2 = (t.agree(f, task, lv) for lv in ["q8_0", "q4_K_M", "q2_K"])
            parts.append(_part(f"{LABEL[f]} {A.TASK_LABEL[task]}: agreement q2_K < q4_K_M < q8_0",
                               f"{_pc(a8)} → {_pc(a4)} → {_pc(a2)}", a2 < a4 < a8))
    P.append((7, "Answers move before accuracy does", parts))

    s = t.df[(t.df.family == "llama3.1-8b") & (t.df.task == "gsm8k") & (t.df.variant == "base")]
    lf, l2 = s[s.level == "fp16"].output_tokens.mean(), s[s.level == "q2_K"].output_tokens.mean()
    P.append((8, "Low-bit Llama talks longer", [
        _part("Llama GSM8K output at q2_K ≥ 1.15 × fp16", f"{lf:.0f} → {l2:.0f} tokens ({l2 / lf:.2f}×)",
              l2 >= 1.15 * lf)]))

    s = t.df[(t.df.family == "llama3.1-8b") & (t.df.task == "drift")]
    sf, s2 = s[s.level == "fp16"].json_strict.mean(), s[s.level == "q2_K"].json_strict.mean()
    P.append((9, "Low-bit Llama changes its JSON habit", [
        _part("Llama strict-JSON share, q2_K vs fp16, differs by ≥ 50 pp", f"{_pc(sf)} → {_pc(s2)}",
              abs(s2 - sf) >= 0.5)]))

    g = "gemma3-12b"
    w = {lv: np.mean([t.agree(g, task, lv) for task in ["gsm8k", "mmlu"]]) for lv in ["qat", "q4_K_M"]}
    parts = [_part("Gemma: QAT agreement with fp16 > q4_K_M's, pooled", f"QAT {_pc(w['qat'])}, q4_K_M {_pc(w['q4_K_M'])}",
                   w["qat"] > w["q4_K_M"])]
    for task in ["gsm8k", "mmlu"]:
        a, b = t.d(g, task, "qat"), t.d(g, task, "q4_K_M")
        if a is not None and b is not None:
            parts.append(_part(f"Gemma {A.TASK_LABEL[task]}: QAT drop no worse than q4_K_M's",
                               f"QAT {_pp(a['diff'])}, q4_K_M {_pp(b['diff'])}", a["diff"] >= b["diff"]))
    P.append((10, "QAT beats the ordinary 4-bit build", parts))

    P.append((11, "Bigger and newer models hold 4-bit better", [
        _drop_part(t, f, task, "q4_K_M", lambda x: abs(x) <= 0.03, f"{LABEL[f]} {A.TASK_LABEL[task]}: |drop| ≤ 3 pp")
        for f in ["qwen3-8b", "gemma3-12b"] for task in TASKS3]))

    aa = A.aa_table(t.df)
    aa = aa[aa.level == "q4_K_M"]
    P.append((12, "The A/A repeat is quiet", [
        _part(f"{LABEL[r.family]} {A.TASK_LABEL[r.task]}: ≤ 3% changed", f"{r.changed}/{r.n}", r.changed / r.n <= 0.03)
        for r in aa.itertuples()]))

    sp = A.speed_table(t.df).set_index(["family", "level"])["tokens/s"]
    parts = []
    for f in fams:
        if (f, "fp16") in sp and (f, "q4_K_M") in sp:
            ratio = sp[(f, "q4_K_M")] / sp[(f, "fp16")]
            parts.append(_part(f"{LABEL[f]}: q4_K_M ≥ 2.5 × fp16 tokens/s",
                               f"{sp[(f, 'fp16')]:.1f} → {sp[(f, 'q4_K_M')]:.1f} ({ratio:.1f}×)", ratio >= 2.5))
    P.append((13, "4-bit is much faster", parts))
    return P


def write_scorecard(split: str = "main") -> str:
    t = T(A.load(split))
    rows, detail = [], []
    for k, name, parts in predictions(t):
        scored = [p for p in parts if p["ok"] is not None]
        n_ok = sum(p["ok"] for p in scored)
        verdict = ("untestable" if not scored else "held" if n_ok == len(scored)
                   else "failed" if n_ok == 0 else f"partly ({n_ok}/{len(scored)})")
        rows.append([k, name, verdict])
        detail += [f"**{k}. {name}: {verdict}**", ""]
        detail += [f"- {'–' if p['ok'] is None else '✓' if p['ok'] else '✗'} {p['desc']}: {p['value']}" for p in parts]
        detail.append("")
    out = [f"# Prediction scorecard ({split} items)", "",
           "Generated by `qcliff score`; the pass rules are in `src/qcliff/score.py`.", "",
           md_table(pd.DataFrame(rows, columns=["#", "prediction", "verdict"])), ""] + detail
    name = "scorecard.md" if split == "main" else f"scorecard_{split}.md"
    os.makedirs(C.REPORTS, exist_ok=True)
    with open(os.path.join(C.REPORTS, name), "w") as f:
        f.write("\n".join(out))
    print(f"wrote reports/{name}")
    return name
