import json

from qcliff import config as C
from qcliff.data import load_items
from qcliff.prompts import render, variants

G = {"id": "gsm8k-0001", "task": "gsm8k", "split": "main", "question": "What is 2 + 2?", "gold": "4"}


def test_gsm8k_prompt_is_format_taxs_free_cot():
    m = render(G, "base").messages
    assert m[0]["content"].startswith("Solve the following math problem. Reason step by step")
    assert m[0]["content"].endswith("Problem: What is 2 + 2?")
    assert render(G, "aa").tag == "aa" and render(G, "base").tag == ""


def test_aa_only_at_q4():
    assert variants("gsm8k", "fp16") == ["base"]
    assert variants("gsm8k", "q4_K_M") == ["base", "aa"]
    assert variants("gsm8k", "q2_K") == ["base"]
    assert variants("drift", "fp16") == ["base", "reformatted"]


def test_reformatted_changes_values_not_gold():
    it = next(i for i in load_items("main", ["drift"]) if i["record"].get("priceRange") == "£20-25")
    base, ref = render(it, "base"), render(it, "reformatted")
    assert "£20-25" in base.messages[0]["content"] and "20-25 GBP" in ref.messages[0]["content"]
    assert base.gold == ref.gold == it["gold"]


def test_every_level_names_a_tag_and_bits():
    for fam, lv in C.all_runs():
        assert C.tag(fam, lv) and lv in C.BITS
    assert len(C.all_runs()) == 21
    assert all(C.REFERENCE in C.FAMILIES[f] for f in C.FAMILIES)


def test_items_match_earlier_studies():
    items = load_items("main")
    assert sum(i["task"] == "gsm8k" for i in items) == 160 and sum(i["task"] == "drift" for i in items) == 150
    json.dumps(items)
