#!/usr/bin/env python3
"""Write a cropped LoCoMo JSON while keeping RAG fields.

Use this when you want a reusable data-file for smoke runs. Official
``evaluate_qa.py`` also accepts ``--sample-id`` / ``--qa-per-category``
in-memory; this script is for writing a file that can be committed or
passed as ``--data-file``.

Preserved on each sample: conversation, observation, session_summary,
event_summary, and any other non-qa fields.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_eval.openai_compat import filter_locomo_samples, parse_sample_ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, help="Source LoCoMo JSON")
    parser.add_argument("--out-file", required=True, help="Cropped JSON to write")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Keep these sample_id values (repeat or comma-separate).",
    )
    parser.add_argument(
        "--qa-per-category",
        type=int,
        default=None,
        help="Keep at most N QA items per category in each sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = json.load(open(args.data_file))
    cropped = filter_locomo_samples(
        samples,
        sample_ids=parse_sample_ids(args.sample_id),
        qa_per_category=args.qa_per_category,
    )
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(cropped, handle, indent=2)
    n_qa = sum(len(sample.get("qa", [])) for sample in cropped)
    print(
        "Wrote %s sample(s), %s QA item(s) to %s"
        % (len(cropped), n_qa, out_path)
    )


if __name__ == "__main__":
    main()
