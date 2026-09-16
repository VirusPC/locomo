# Edges evaluation smoke (VirusPC fork)

Thin delta on top of [snap-research/locomo](https://github.com/snap-research/locomo) so [VirusPC/edges](https://github.com/VirusPC/edges) can run **official** truncated-context QA (`task_eval/evaluate_qa.py` → `get_gpt_answers` → `eval_question_answering` F1) with OpenAI-compatible model ids such as `kimi-for-coding`.

This is an **evaluation smoke**, not a claim that a model has been proven on the LoCoMo benchmark.

## What changed vs upstream

Upstream `evaluate_qa.py` only sent names containing `gpt` through the Chat Completions path. Other ids (`kimi-for-coding`, `k3`, …) raised `NotImplementedError`. `run_chatgpt` likewise accepted only `chatgpt` / `gpt-4*`.

This fork:

- Routes non-Claude / non-Gemini / non-HF model names through the official GPT QA helpers (`task_eval/gpt_utils.py` prompts + truncated context).
- Accepts `KIMI_API_KEY` or `OPENAI_API_KEY`. For non-gpt OpenAI-compatible models, `OPENAI_BASE_URL` defaults to `https://api.kimi.com/coding/v1`.
- Leaves `task_eval/evaluation.py` (`eval_question_answering` / F1) unchanged.
- Adds optional `--sample-id` and `--qa-per-category` so a full `data/locomo10.json` run is not required for smoke. Full-file runs are unchanged when those flags are omitted.
- Does **not** add Project Memory / RAG arms. Observation and `session_summary` fields stay on the data samples for later RAG work.

## Environment

```bash
export KIMI_API_KEY=...          # or OPENAI_API_KEY
# optional; this is the default for non-gpt OpenAI-compatible models
export OPENAI_BASE_URL=https://api.kimi.com/coding/v1
```

Do not commit keys. `scripts/env.sh` no longer overwrites keys that are already set.

Minimal Python packages for this path (the repo `requirements.txt` is still the upstream conda export):

```bash
pip install -r requirements-openai-compat.txt
```

`evaluate_qa.py` still imports `task_eval/evaluation.py`, which depends on `bert-score` / `nltk` even though QA scoring itself is F1.

## Tiny smoke

Official entry point, one conversation, one QA item per category:

```bash
export KIMI_API_KEY=...
python3 task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/locomo10_kimi_smoke.json \
    --model kimi-for-coding \
    --sample-id conv-26 \
    --qa-per-category 1 \
    --batch-size 1
```

Or:

```bash
export KIMI_API_KEY=...
bash scripts/evaluate_kimi_smoke.sh
```

Scoring is the official F1 path. Predictions land on each QA as `kimi-for-coding_prediction`; F1 as `kimi-for-coding_f1`. Aggregate stats are written next to `--out-file` as `*_stats.json`.

Other OpenAI-compatible names work the same way if you set `OPENAI_BASE_URL` / `OPENAI_API_KEY` for that provider:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-compatible-host/v1
python3 task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/locomo10_compat_smoke.json \
    --model your-model-id \
    --sample-id conv-26 \
    --qa-per-category 1 \
    --batch-size 1
```

## Optional cropped data-file

In-memory flags do not strip `observation` / `session_summary`. If you want a reusable file instead:

```bash
python3 scripts/crop_locomo_data.py \
    --data-file data/locomo10.json \
    --out-file data/locomo10_smoke.json \
    --sample-id conv-26 \
    --qa-per-category 1
```

Then pass `--data-file data/locomo10_smoke.json` to `evaluate_qa.py`. Full-file runs should keep using `data/locomo10.json`.

## kimi-for-coding notes

Verified against Kimi Code docs and the official gpt helpers:

- Model id: `kimi-for-coding` (also `kimi-for-coding-highspeed`, `k3`, `k3-256k`).
- Official gpt QA uses `temperature=0` and `max_tokens=32` (single-item). Those defaults are **preserved for gpt-\*** names.
- For non-gpt OpenAI-compatible names this fork defaults to `temperature=1` and `max_tokens=1024`. Several Kimi models fix temperature at `1.0` and return HTTP 400 for other values; thinking/reasoning can consume a tiny completion budget so `content` is empty. Override with `--temperature` / `--max-tokens` if needed.
- If the API rejects `temperature`, `run_chatgpt` retries once without that field.
- Empty `content` is not replaced with `reasoning_content` (that would change the scored string). Raise `--max-tokens` instead.
- Truncated-context token counting uses tiktoken `cl100k_base` when the model name is unknown. Context window defaults to 128000 (same as official `gpt-4-turbo`) unless the model is listed or `OPENAI_MAX_CONTEXT` is set.

## What this is not

- Not a LoCoMo leaderboard run.
- Not a RAG / Project Memory implementation.
- Not a rewrite of official prompts or F1 scoring.
