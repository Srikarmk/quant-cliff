"""Prompt templates: format-tax's free-text reasoning-first prompts for GSM8K and MMLU, and its drift
prompt for extraction, unchanged, so the q4_K_M rows can be checked against format-tax's.

Extraction is shown in two source variants: the base JSON export and the same record with values
reformatted ("£20-25" as "20-25 GBP"), the one change that cost a model accuracy in format-tax. Replies
are prompt-only JSON (no schema), so JSON validity is measured as well. Every template is written out
in full so it can be read and checked by eye.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import config as C

G_T = "Solve the following math problem."
GSM8K = f'{G_T} Reason step by step, then write the final answer on the last line as "Answer: <number>".'
M_T = "Answer the following multiple-choice question about {subject}."
MMLU = (f'{M_T} Reason step by step, then write the letter of the correct option on the last line as '
        '"Answer: <letter>".')

DRIFT_PROMPT = (
    "Convert the restaurant record below into a JSON object with exactly these keys:\n"
    "- name: the venue's name\n"
    '- eat_type: one of "restaurant", "pub", "coffee shop"\n'
    "- food: the type of food served\n"
    '- price_range: one of "cheap", "moderate", "high", "less than £20", "£20-25", "more than £30"\n'
    '- customer_rating: one of "low", "average", "high", "1 out of 5", "3 out of 5", "5 out of 5"\n'
    '- area: one of "riverside", "city centre"\n'
    "- family_friendly: true or false\n"
    "- near: a nearby landmark\n"
    "Use null for any field the record does not give. Reply with only the JSON object.\n\n"
    "Record:\n{record}"
)
REFORMAT = {
    "priceRange": {"less than £20": "under 20 GBP", "£20-25": "20-25 GBP", "more than £30": "over 30 GBP"},
    "customer rating": {"1 out of 5": "1/5", "3 out of 5": "3/5", "5 out of 5": "5/5"},
    "familyFriendly": {"yes": "Y", "no": "N"},
    "eatType": {"coffee shop": "coffee_shop"},
}

# per task: the variants each item is sent in. "aa" is the base prompt sent again (only at C.AA_LEVELS).
VARIANTS = {"gsm8k": ["base", "aa"], "mmlu": ["base", "aa"], "drift": ["base", "reformatted"]}


@dataclass
class Rendered:
    messages: list[dict]
    gold: object
    tag: str = ""


def mmlu_body(it: dict) -> str:
    opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", it["options"]))
    return f"Question: {it['question']}\n\n{opts}"


def render(it: dict, variant: str) -> Rendered:
    tag = "aa" if variant == "aa" else ""
    if it["task"] == "drift":
        rec = it["record"]
        if variant == "reformatted":
            rec = {k: REFORMAT.get(k, {}).get(x, x) for k, x in rec.items()}
        text = json.dumps(rec, indent=2, ensure_ascii=False)
        return Rendered([{"role": "user", "content": DRIFT_PROMPT.format(record=text)}], it["gold"], tag)
    if it["task"] == "gsm8k":
        user = f"{GSM8K}\n\nProblem: {it['question']}"
    else:
        subject = it["subject"].replace("high_school_", "high school ").replace("_", " ")
        user = MMLU.format(subject=subject) + "\n\n" + mmlu_body(it)
    return Rendered([{"role": "user", "content": user}], it["gold"], tag)


def variants(task: str, level: str) -> list[str]:
    return [v for v in VARIANTS[task] if v != "aa" or level in C.AA_LEVELS]
