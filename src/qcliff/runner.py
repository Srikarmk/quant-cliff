"""Send every item to one model at one quantization level, parse the answer, store one row per call."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .llm import CacheMiss, Client
from .parse import check_a, check_drift
from .prompts import render, variants


def answers_path(family: str, level: str, split: str = "main") -> str:
    name = f"{family}_{level}" + ("" if split == "main" else f"_{split}")
    return os.path.join(C.RESULTS, "answers", f"{name}.jsonl")


class Progress:
    def __init__(self, total: int, label: str, every: int = 100):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0, self.lock = 0, time.time(), threading.Lock()

    def tick(self, client: Client):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                el = time.time() - self.t0
                print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                      f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def jobs_for(items: list[dict], level: str) -> list[tuple[dict, str]]:
    """Item by item, variants shuffled within an item so the A/A repeat is not sent back to back."""
    rng = random.Random(C.SEED)
    jobs = []
    for it in items:
        js = [(it, v) for v in variants(it["task"], level)]
        rng.shuffle(js)
        jobs += js
    return jobs


def run_one(client: Client, family: str, level: str, it: dict, variant: str) -> dict:
    model = C.tag(family, level)
    r = render(it, variant)
    out = client.chat(model, r.messages, C.options(it["task"]), tag=r.tag)
    row = {
        "family": family, "level": level, "model": model, "id": it["id"], "task": it["task"],
        "split": it["split"], "variant": variant, "output": out["text"],
        "truncated": out.get("done_reason") == "length", "prompt_tokens": out["prompt_tokens"],
        "output_tokens": out["output_tokens"], "seconds": out["seconds"],
        "eval_seconds": out.get("eval_seconds"), "cached": out["cached"],
    }
    if it["task"] == "drift":
        chk = check_drift(out["text"], r.gold)
        row.update(parsed=chk["parsed"], json_strict=chk["json_strict"], keys_exact=chk["keys_exact"],
                   answer=chk["answer"], fields=chk["fields"], filled=chk["filled"],
                   fields_correct=sum(chk["fields"].values()), correct=all(chk["fields"].values()))
    else:
        row.update(check_a(it["task"], "free_cot", out["text"], r.gold, it.get("options")))
    return row


def run(items: list[dict], client: Client, family: str, level: str, workers: int = 1, split: str = "main",
        skip_missing: bool = False) -> list[dict]:
    jobs = jobs_for(items, level)
    prog = Progress(len(jobs), f"{family} {level}")

    def work(job):
        try:
            row = run_one(client, family, level, *job)
        except CacheMiss:
            if not skip_missing:
                raise
            row = None
        prog.tick(client)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(work, jobs) if r is not None]
    path = answers_path(family, level, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["task"], r["id"], r["variant"]))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
