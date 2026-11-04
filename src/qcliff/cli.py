"""qcliff command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C


def _client(args):
    from .llm import Cache, Client
    os.makedirs(os.path.dirname(C.CACHE_DB), exist_ok=True)
    return Client(Cache(C.CACHE_DB), cache_only=getattr(args, "cache_only", False))


def _items(args):
    from .data import load_items
    tasks = args.tasks.split(",") if args.tasks else None
    items = load_items(args.split, tasks)
    if args.per_task:
        keep = []
        for t in C.TASKS:
            keep += [it for it in items if it["task"] == t][:args.per_task]
        items = keep
    return items


def _runs(args) -> list[tuple[str, str]]:
    fams = args.families.split(",") if args.families else list(C.FAMILIES)
    levels = args.levels.split(",") if args.levels else C.LEVELS
    return [(f, lv) for f, lv in C.all_runs() if f in fams and lv in levels]


def cmd_fetch(args):
    from .data import fetch
    for name, meta in fetch().items():
        print(f"{name}: {meta['repo']}@{meta['revision'][:10]} {meta['path']} {meta['bytes']} bytes")


def cmd_items(args):
    from collections import Counter

    from .data import write_items
    items = write_items()
    print(f"wrote {len(items)} items to data/items.jsonl:",
          dict(Counter(f"{i['task']}/{i['split']}" for i in items)))


def cmd_run(args):
    from .runner import run
    items = _items(args)
    client = _client(args)
    for family, level in _runs(args):
        if not args.cache_only:
            _record_model(family, level)
        rows = run(items, client, family, level, workers=args.workers, split=args.split,
                   skip_missing=args.skip_missing)
        for task in C.TASKS:
            rs = [r for r in rows if r["task"] == task and r["variant"] == "base"]
            if rs:
                print(f"{family} {level} {task}: {len(rs)} replies, accuracy "
                      f"{sum(r['correct'] for r in rs) / len(rs):.3f}", flush=True)
        _unload(C.tag(family, level))
    print("cache", client.stats)


def _unload(model: str):
    """Free the model's memory before the next level loads (fp16 builds are up to 24 GB)."""
    import urllib.request
    try:
        req = urllib.request.Request(C.OLLAMA_URL + "/api/generate", json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def _record_model(family: str, level: str):
    from .llm import model_info, think_flag
    path = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(path)) if os.path.exists(path) else {}
    m = C.tag(family, level)
    meta[f"{family}/{level}"] = {**model_info(m), "think": think_flag(m), "bits": C.BITS[level]}
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def cmd_show(args):
    from .data import load_items
    from .runner import answers_path
    it = next(i for i in load_items(None) if i["id"] == args.id)
    print(f"{it['id']}  gold={it['gold']}\n")
    for family, level in _runs(args):
        path = answers_path(family, level, it["split"])
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in (r for r in map(json.loads, f) if r["id"] == it["id"]):
                print(f"===== {family} {level} {r['variant']}: answer={r['answer']} correct={r['correct']} "
                      f"tokens={r['output_tokens']}")
                if args.full:
                    print(r["output"] + "\n")


def cmd_report(args):
    from .report import write_report
    write_report(args.split)


def cmd_score(args):
    from .score import write_scorecard
    write_scorecard(args.split)


def main(argv=None):
    p = argparse.ArgumentParser(prog="qcliff")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned benchmark files")
    sub.add_parser("items", help="build data/items.jsonl from the downloaded files")
    for name in ("run", "show"):
        q = sub.add_parser(name)
        q.add_argument("--families", default="")
        q.add_argument("--levels", default="")
        if name == "run":
            q.add_argument("--split", default="main", choices=["main", "pilot"])
            q.add_argument("--tasks", default="")
            q.add_argument("--per-task", type=int, default=0, help="only the first N items of each task")
            q.add_argument("--workers", type=int, default=1)
            q.add_argument("--cache-only", action="store_true")
            q.add_argument("--skip-missing", action="store_true")
        else:
            q.add_argument("id")
            q.add_argument("--full", action="store_true")
    for name in ("report", "score"):
        q = sub.add_parser(name)
        q.add_argument("--split", default="main", choices=["main", "pilot"])
    args = p.parse_args(argv)
    fn = {"fetch": cmd_fetch, "items": cmd_items, "run": cmd_run, "show": cmd_show, "report": cmd_report,
          "score": cmd_score}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
