# save generated outputs to this location
OUT_DIR=./outputs

# save embeddings to this location
EMB_DIR=./outputs

# path to LoCoMo data file
DATA_FILE_PATH=./data/locomo10.json

# filenames for different outputs
QA_OUTPUT_FILE=locomo10_qa.json
OBS_OUTPUT_FILE=locomo10_observation.json
SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples

# OpenAI API Key (do not clobber a key already in the environment)
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# Optional alias for OpenAI-compatible models such as kimi-for-coding
export KIMI_API_KEY="${KIMI_API_KEY:-}"

# Optional Chat Completions base URL. Non-gpt OpenAI-compatible models
# default to https://api.kimi.com/coding/v1 when this is unset.
# export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

# Google API Key
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# Anthropic API Key
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# HuggingFace Token
export HF_TOKEN="${HF_TOKEN:-}"
