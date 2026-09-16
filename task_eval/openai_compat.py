"""Thin helpers so official LoCoMo QA can call OpenAI-compatible chat models.

Official ``evaluate_qa.py`` only routed names containing ``gpt`` through
``get_gpt_answers`` / ``run_chatgpt``. Model ids such as ``kimi-for-coding``
hit ``NotImplementedError``. This module keeps that GPT path and scoring
untouched while recognizing other Chat Completions model names.
"""

import os


HF_NAME_MARKERS = ("gemma", "llama", "mistral")
DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_COMPAT_CONTEXT = 128000
DEFAULT_COMPAT_MAX_TOKENS = 1024
GPT_SINGLE_QA_MAX_TOKENS = 32
FALLBACK_ENCODING = "cl100k_base"


def uses_hf_model(model):
    return any(name in model for name in HF_NAME_MARKERS)


def uses_claude_model(model):
    return "claude" in model


def uses_gemini_model(model):
    return "gemini" in model


def uses_gpt_model(model):
    return "gpt" in model


def uses_openai_compat_model(model):
    """True for gpt-* and other Chat Completions ids (e.g. kimi-for-coding)."""
    if uses_hf_model(model) or uses_claude_model(model) or uses_gemini_model(model):
        return False
    return True


def parse_sample_ids(values):
    """Flatten repeatable / comma-separated ``--sample-id`` values."""
    if not values:
        return None
    sample_ids = []
    for value in values:
        sample_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return sample_ids or None


def filter_locomo_samples(samples, sample_ids=None, qa_per_category=None):
    """In-memory crop. Keeps observation / session_summary / conversation."""
    if sample_ids:
        wanted = set(sample_ids)
        samples = [sample for sample in samples if sample.get("sample_id") in wanted]
        found = {sample.get("sample_id") for sample in samples}
        missing = wanted - found
        if missing:
            raise ValueError("Unknown sample_id(s): %s" % ", ".join(sorted(missing)))

    if qa_per_category is None:
        return samples

    cropped = []
    for sample in samples:
        sample = dict(sample)
        counts = {}
        qa_out = []
        for qa in sample.get("qa", []):
            category = qa.get("category")
            counts[category] = counts.get(category, 0) + 1
            if counts[category] <= qa_per_category:
                qa_out.append(qa)
        sample["qa"] = qa_out
        cropped.append(sample)
    return cropped


def encoding_for_eval_model(model):
    """Match official gpt tiktoken selection; fall back for non-gpt names."""
    import tiktoken

    name = "gpt-3.5-turbo-16k" if any(k in model for k in ["16k", "12k", "8k", "4k"]) else model
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def context_length_for_model(model, max_length_table):
    if model in max_length_table:
        return max_length_table[model]
    env_value = os.environ.get("OPENAI_MAX_CONTEXT")
    if env_value:
        return int(env_value)
    return DEFAULT_COMPAT_CONTEXT


def qa_temperature(model, override=None):
    """Official gpt QA uses temperature=0. Kimi-style models often require 1."""
    if override is not None:
        return override
    if uses_gpt_model(model):
        return 0
    return 1


def qa_max_tokens(model, batch_size, batched, override=None, per_qa_token_budget=50):
    """Official single-QA gpt budget is 32. Compat models need more headroom."""
    if override is not None:
        return override
    if uses_gpt_model(model):
        if batched:
            return batch_size * per_qa_token_budget
        return GPT_SINGLE_QA_MAX_TOKENS
    return max(DEFAULT_COMPAT_MAX_TOKENS, batch_size * per_qa_token_budget)


def message_text(message):
    """Return assistant ``content`` only. Do not score ``reasoning_content``."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                text = getattr(part, "text", None)
                parts.append(text if text is not None else str(part))
        return "".join(parts)
    return content


def resolve_openai_credentials(model=None):
    """Return (api_key, api_base) for the official openai==0.28 client."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY or KIMI_API_KEY")

    api_base = os.environ.get("OPENAI_BASE_URL")
    if (
        not api_base
        and model
        and uses_openai_compat_model(model)
        and not uses_gpt_model(model)
    ):
        api_base = DEFAULT_KIMI_BASE_URL
    return api_key, api_base
