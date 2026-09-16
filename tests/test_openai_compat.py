#!/usr/bin/env python3
"""Unit tests for the thin OpenAI-compatible LoCoMo QA patch."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from task_eval.openai_compat import (
    DEFAULT_KIMI_BASE_URL,
    context_length_for_model,
    filter_locomo_samples,
    message_text,
    parse_sample_ids,
    qa_max_tokens,
    qa_temperature,
    resolve_openai_credentials,
    uses_openai_compat_model,
)


class RoutingTests(unittest.TestCase):
    def test_gpt_and_kimi_use_compat_path(self):
        self.assertTrue(uses_openai_compat_model("gpt-4-turbo"))
        self.assertTrue(uses_openai_compat_model("gpt-3.5-turbo-16k"))
        self.assertTrue(uses_openai_compat_model("kimi-for-coding"))
        self.assertTrue(uses_openai_compat_model("k3"))

    def test_other_providers_stay_on_their_paths(self):
        self.assertFalse(uses_openai_compat_model("claude-sonnet"))
        self.assertFalse(uses_openai_compat_model("gemini-pro-1.0"))
        self.assertFalse(uses_openai_compat_model("llama2"))
        self.assertFalse(uses_openai_compat_model("mistral-7b"))
        self.assertFalse(uses_openai_compat_model("gemma-7b"))


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {
                "sample_id": "conv-26",
                "observation": {"keep": True},
                "session_summary": {"keep": True},
                "conversation": {"session_1": []},
                "qa": [
                    {"question": "q1", "category": 1},
                    {"question": "q2", "category": 1},
                    {"question": "q3", "category": 2},
                ],
            },
            {
                "sample_id": "conv-30",
                "observation": {"other": True},
                "qa": [{"question": "q4", "category": 1}],
            },
        ]

    def test_parse_sample_ids(self):
        self.assertEqual(parse_sample_ids(["conv-26, conv-30", "conv-41"]),
                         ["conv-26", "conv-30", "conv-41"])
        self.assertIsNone(parse_sample_ids(None))

    def test_filter_keeps_rag_fields_and_limits_qa(self):
        cropped = filter_locomo_samples(
            self.samples, sample_ids=["conv-26"], qa_per_category=1
        )
        self.assertEqual(len(cropped), 1)
        self.assertEqual(cropped[0]["sample_id"], "conv-26")
        self.assertEqual(cropped[0]["observation"], {"keep": True})
        self.assertEqual(cropped[0]["session_summary"], {"keep": True})
        self.assertEqual(cropped[0]["conversation"], {"session_1": []})
        self.assertEqual([qa["category"] for qa in cropped[0]["qa"]], [1, 2])

    def test_unknown_sample_id_errors(self):
        with self.assertRaises(ValueError):
            filter_locomo_samples(self.samples, sample_ids=["missing"])


class DefaultTests(unittest.TestCase):
    def test_official_gpt_defaults_preserved(self):
        self.assertEqual(qa_temperature("gpt-4-turbo"), 0)
        self.assertEqual(qa_max_tokens("gpt-4-turbo", 1, batched=False), 32)
        self.assertEqual(qa_max_tokens("gpt-4-turbo", 10, batched=True), 500)

    def test_kimi_defaults(self):
        self.assertEqual(qa_temperature("kimi-for-coding"), 1)
        self.assertEqual(qa_max_tokens("kimi-for-coding", 1, batched=False), 1024)

    def test_overrides(self):
        self.assertEqual(qa_temperature("kimi-for-coding", override=0.7), 0.7)
        self.assertEqual(qa_max_tokens("gpt-4-turbo", 1, batched=False, override=64), 64)

    def test_context_length_fallback(self):
        table = {"gpt-4-turbo": 128000}
        self.assertEqual(context_length_for_model("gpt-4-turbo", table), 128000)
        self.assertEqual(context_length_for_model("kimi-for-coding", table), 128000)
        with mock.patch.dict(os.environ, {"OPENAI_MAX_CONTEXT": "64000"}):
            self.assertEqual(context_length_for_model("unknown-model", table), 64000)


class CredentialTests(unittest.TestCase):
    def test_kimi_key_and_default_base(self):
        env = {"KIMI_API_KEY": "kimi-secret"}
        with mock.patch.dict(os.environ, env, clear=True):
            key, base = resolve_openai_credentials("kimi-for-coding")
        self.assertEqual(key, "kimi-secret")
        self.assertEqual(base, DEFAULT_KIMI_BASE_URL)

    def test_gpt_does_not_default_to_kimi_base(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            key, base = resolve_openai_credentials("gpt-4-turbo")
        self.assertEqual(key, "sk-test")
        self.assertIsNone(base)

    def test_explicit_base_url_wins(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://example.test/v1"},
            clear=True,
        ):
            _, base = resolve_openai_credentials("kimi-for-coding")
        self.assertEqual(base, "https://example.test/v1")


class MessageTextTests(unittest.TestCase):
    def test_empty_content_is_not_reasoning(self):
        message = SimpleNamespace(content="", reasoning_content="hidden")
        self.assertEqual(message_text(message), "")

    def test_none_content(self):
        self.assertEqual(message_text(SimpleNamespace(content=None)), "")

    def test_list_content(self):
        self.assertEqual(message_text({"content": [{"text": "hello"}]}), "hello")


class CropScriptTests(unittest.TestCase):
    def test_crop_script_preserves_fields(self):
        import json
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            dst = Path(tmp) / "out.json"
            src.write_text(json.dumps([
                {
                    "sample_id": "conv-26",
                    "observation": {"session_1_observation": ["x"]},
                    "session_summary": {"session_1_summary": "s"},
                    "event_summary": {},
                    "conversation": {"speaker_a": "A"},
                    "qa": [
                        {"question": "a", "category": 1},
                        {"question": "b", "category": 1},
                    ],
                }
            ]))
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "crop_locomo_data.py"),
                    "--data-file", str(src),
                    "--out-file", str(dst),
                    "--sample-id", "conv-26",
                    "--qa-per-category", "1",
                ]
            )
            out = json.loads(dst.read_text())
            self.assertEqual(len(out), 1)
            self.assertIn("observation", out[0])
            self.assertIn("session_summary", out[0])
            self.assertEqual(len(out[0]["qa"]), 1)


class EncodingTests(unittest.TestCase):
    def test_unknown_model_falls_back_to_cl100k(self):
        try:
            from task_eval.openai_compat import encoding_for_eval_model
            encoding = encoding_for_eval_model("kimi-for-coding")
        except ImportError:
            self.skipTest("tiktoken is not installed")
        self.assertGreater(len(encoding.encode("hello world")), 0)


class RunChatgptCompatTests(unittest.TestCase):
    def test_kimi_model_uses_chat_completions(self):
        try:
            import openai  # noqa: F401
            from global_methods import run_chatgpt
        except ImportError:
            self.skipTest("openai is not installed")

        fake_choice = SimpleNamespace(message=SimpleNamespace(content="short answer"))
        fake_completion = SimpleNamespace(choices=[fake_choice])
        with mock.patch("global_methods.openai.ChatCompletion.create", return_value=fake_completion) as create:
            text = run_chatgpt("hello", model="kimi-for-coding", num_tokens_request=1024, temperature=1)
        self.assertEqual(text, "short answer")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "kimi-for-coding")
        self.assertEqual(kwargs["max_tokens"], 1024)
        self.assertEqual(kwargs["temperature"], 1)
        self.assertEqual(kwargs["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
