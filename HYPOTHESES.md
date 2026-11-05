# Hypotheses and analysis plan

Written 2026-11-03 at 19:40 EST, before the full run. Nothing below is edited after the full run starts;
corrections go in a dated addendum at the end.

What had been seen when this was written: a pilot on the 10 pilot items per task (10 GSM8K, 10 MMLU, 10
extraction records in two variants) at every level of every model, 1,000 calls, used to check the prompts,
the parser, the analysis and the timing. The pilot items are excluded from every reported number. Pilot
observations that informed a prediction are marked (P). On far too few items to estimate anything, the
pilot showed: almost no accuracy change at any level, the largest being Llama 3.1 8B at q2_K on maths
(7 of 10 right against 9 of 10 at fp16); even q8_0 changing an occasional answer relative to fp16; Llama's GSM8K replies
lengthening at q4_K_M and below (about 200 tokens at fp16, about 290 at q4_K_M to q2_K); and Llama at
q2_K dropping the markdown code fence it puts around JSON at every other level. Earlier studies on the
same items give the q4_K_M baselines: GSM8K 93% / 88% / 95% and MMLU 91% / 78% / 90% for qwen3 8B,
Llama 3.1 8B and Gemma 3 12B (format-tax, free text, reasoning first; 100 of these items).

## Question

At what bit-width does a quantized LLM stop working, and does it break the same way on maths, knowledge
and extraction? Is the 4-bit build most people run (Ollama's default) free, and does a quantization-aware
4-bit build do better than the ordinary one?

## Design

- Models and levels (Ollama library builds, llama.cpp GGUF): Llama 3.1 8B Instruct and Qwen2.5 7B
  Instruct at fp16, q8_0, q6_K, q5_K_M, q4_K_M, q3_K_M and q2_K; qwen3 8B (thinking off) at fp16, q8_0 and
  q4_K_M; Gemma 3 12B at fp16, q8_0, q4_K_M and its quantization-aware-trained 4-bit build (QAT). 21
  builds. The untagged Ollama names are the q4_K_M builds (same digests). Digests in `results/models.json`.
- Items: 160 GSM8K and 160 MMLU (four high-school subjects; the 320 items of groupthink), answered with
  format-tax's free-text reasoning-first prompt; and format-tax's 150 E2E restaurant records converted to
  a fixed 8-field JSON schema (prompt-only, no constrained decoding), from the base export and from a
  version with values reformatted ("£20-25" as "20-25 GBP").
- Temperature 0, seed 7, `num_ctx` 4096, output caps 1,024 (GSM8K), 768 (MMLU) and 300 (extraction).
- Reference: each model's own fp16 build. Noise floor: GSM8K and MMLU sent a second time at q4_K_M (A/A).
- Scoring: GSM8K and MMLU right or wrong, unparsed counts as wrong; extraction as the share of the 8 fields
  right per record (format-tax's field checker), plus JSON validity.

## Predictions

pp = percentage points. A "drop" is a level's score minus fp16's, paired by item; negative is worse.
"Agreement" is the share of items where a level gives the same final answer as fp16. "Ladder models" are
Llama 3.1 8B and Qwen2.5 7B.

High precision is free

1. q8_0 is free: for every model and task, the q8_0 drop is within ±2 pp, with an interval that includes
   zero.
2. q6_K and q5_K_M are free: for both ladder models, every drop at q6_K and q5_K_M is within ±3 pp.
3. Ollama's default costs little: the q4_K_M drop is no worse than −4 pp for every model and task.

Where it breaks

4. The cliff is at 3 bits or below: for both ladder models on GSM8K, no level above q3_K_M has a clear
   drop (at least 5 pp with an interval excluding zero). (P) Llama's GSM8K drop at q2_K is at least 5 pp,
   and larger than Qwen2.5's.
5. Maths breaks before multiple choice: at q2_K, the GSM8K drop is larger than the MMLU drop for both
   ladder models.
6. Extraction holds longest: at q2_K, the base extraction drop is within ±2 pp for both ladder models,
   and the reformatted source loses more than the base source.

Changes before failures

7. Answers move before accuracy does: at q8_0, agreement with fp16 is below 100% on at least one task for
   at least three of the four models, even where the drop is within ±2 pp; and for both ladder models,
   agreement on GSM8K and on MMLU is lower at q2_K than at q4_K_M, and lower at q4_K_M than at q8_0.
8. (P) Low-bit Llama talks longer: Llama's mean GSM8K output at q2_K is at least 15% longer than at fp16.
9. (P) Low-bit Llama changes its JSON habit: Llama's strict-JSON share on extraction (the reply is
   exactly one JSON object, no code fence) differs between q2_K and fp16 by at least 50 pp.

Which 4-bit

10. QAT beats the ordinary 4-bit build: Gemma's QAT build agrees with fp16 more often than its q4_K_M
    build, pooled over GSM8K and MMLU, and its drop is no worse on either task.
11. Bigger and newer models hold 4-bit better: at q4_K_M, the qwen3 8B and Gemma 3 12B drops are within
    ±3 pp on every task.

Noise and cost

12. The A/A repeat at q4_K_M changes at most 3% of answers for each model and task.
13. 4-bit is much faster: q4_K_M generates tokens at least 2.5× as fast as fp16 for every model (median
    tokens per second on GSM8K replies).

What would support "quantization degrades gracefully" over "it falls off a cliff": drops that grow
steadily level by level from q6_K down, with no single step accounting for most of the loss at q2_K.
What would contradict prediction 4: a clear GSM8K drop at q4_K_M or above.

## Analysis plan (fixed in advance)

- Unit of resampling: the item. Every level of a model is paired with the same item at fp16. 95%
  percentile bootstrap, 2,000 draws.
- Primary: score by level per model and task, and the paired drop from fp16. "First clear drop": walking
  down from fp16 (QAT left out), the first level whose drop is at least 5 pp with an interval excluding
  zero. "Within x pp" is judged on the point estimate.
- Secondary: agreement with fp16; unparsed and truncated replies; JSON validity and the code-fence rate on
  extraction; output length; the A/A changes; tokens per second and file size (wall-clock on a laptop, a
  rough guide).
- Not done: no prompt was tuned on the main items, and no item is dropped after the run.

## Addendum, 2026-11-04 21:30 EST: scoring after the full run

The full run ended 2026-11-04 about 21:00 EST, about 25 hours after it started: 14,300 calls on the main
items (21 builds × 620, plus the A/A repeat at q4_K_M), every build complete. Nothing above was edited.
The pass rules were fixed in `src/qcliff/score.py` (commit 182b222) against the pilot before any main
result existed; the only change since is the wording of one check's label in prediction 7, which printed
the levels in the opposite order to its values. The scorecard is `reports/scorecard.md`.

Checks before scoring: no call came near the context window (longest prompt plus reply 1,201 of 4,096
tokens). Every unparsed GSM8K/MMLU reply (17), every wrong reply cut off at the cap (38), and every wrong
reply whose last lines mention the right answer (76) was read. All are the model's own failures:
intermediate values that equal the gold, loops to the cap, "none of the above", refusing to choose.
None is a parsing artifact. Re-scoring all 9,870 main replies with an independent copy of the checkers
(evalpair) gave the same score on every one.

Scored: 8 held, 5 partly, 0 failed.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | q8_0 drop within ±2 pp, interval includes 0, every model and task | largest Llama MMLU −1.9 [−5.0, +0.6] | held |
| 2 | q6_K and q5_K_M drops within ±3 pp, ladder models | 11 of 12; Llama MMLU q6_K −6.2 [−11.9, −0.6] | partly |
| 3 | q4_K_M drop no worse than −4 pp, every model and task | 11 of 12; Llama GSM8K −4.4 [−9.4, +0.6] | partly |
| 4 | No clear GSM8K drop above q3_K_M; Llama q2_K ≥ 5 pp and worse than Qwen2.5 | none above q3_K_M; Llama −27.5 [−35.6, −20.0], Qwen2.5 −6.9 | held |
| 5 | At q2_K, GSM8K drop larger than MMLU drop, both ladder models | Llama −27.5 vs −12.5; Qwen2.5 −6.9 vs −9.4 | partly |
| 6 | q2_K base extraction within ±2 pp; reformatted loses more | Llama +0.3 / −0.5; Qwen2.5 +0.1 / −0.6 | held |
| 7 | q8_0 agreement < 100% for ≥ 3 models; agreement falls q8_0 → q4_K_M → q2_K | 4 of 4; e.g. Llama GSM8K 93.8% → 85.6% → 60.0% | held |
| 8 | Llama GSM8K output at q2_K ≥ 1.15 × fp16 | 223 → 306 tokens (1.37×) | held |
| 9 | Llama strict-JSON share, q2_K vs fp16, differs by ≥ 50 pp | 0% → 100% | held |
| 10 | Gemma QAT agrees with fp16 more than q4_K_M, pooled; drop no worse on either task | 94.1% vs 93.1%; MMLU −0.6 vs −1.9; GSM8K +0.6 vs +1.9 (missed) | partly |
| 11 | qwen3 and Gemma q4_K_M drops within ±3 pp, every task | largest Gemma MMLU −1.9 | held |
| 12 | A/A at q4_K_M changes ≤ 3% per model and task | 2 of 1,280 answers (qwen3 MMLU) | held |
| 13 | q4_K_M ≥ 2.5 × fp16 tokens per second, every model | 2.53×, 2.59×, 2.54×, 2.61× | held |

Three of the misses are on point estimates whose intervals include the threshold: prediction 3 by
0.4 pp, and prediction 10's GSM8K check, where QAT gains less than q4_K_M (+0.6 against +1.9 pp, both
intervals spanning zero). They are scored as missed because the rule says so. Prediction 13 passes for
two models by 0.03× and 0.04×, on wall-clock timings from one laptop.

Agreement counts an unparsed reply as a changed answer, as fixed in the plan. That is why fp16 agrees
with itself on 98.8% of Llama's and 99.4% of Qwen2.5's MMLU items (2 and 1 unparsed replies).

Not pre-registered, from reading the replies:

- **Llama's 2-bit maths loss is wrong reasoning, not running out of room.** At q2_K, 49 GSM8K items go
  right → wrong and 5 wrong → right. Only 10 of the 49 are replies cut off at the cap. Among replies that
  finished, accuracy is 66.9% (145 items), against 88.1% at fp16.
- **Llama's MMLU answers move well before its accuracy settles.** The q6_K dip (−6.2 pp: 16 items right
  → wrong, 6 wrong → right) is followed by −1.2 at q5_K_M and −3.8 at q4_K_M, so it is not a trend. From
  q6_K to q4_K_M, Llama gives a different MMLU answer than fp16 on 15–17% of items; Qwen2.5 on 8%.
- **Extraction barely notices quantization.** Every base extraction drop is within ±0.3 pp down to
  q2_K. The largest extraction drop anywhere is Llama's reformatted source at q3_K_M, −2.3 [−3.2, −1.4].
- **What 2 bits looks like.** At q2_K, the cut-off replies loop ("The Equator is not used to determine
  absolute location." repeated to the cap; "Step 29: Based on the above steps..."), Qwen2.5 answers one
  MMLU item in Chinese and on another repeats "hinterland" until the cap.
- **q2_K buys little speed.** At q2_K, Llama generates 62.5 tokens per second against 48.1 at q4_K_M
  (1.3×) and Qwen2.5 62.2 against 52.6 (1.2×), for about 1.7 GB less memory, while scoring 23 and 9 pp
  lower on GSM8K than at q4_K_M.
