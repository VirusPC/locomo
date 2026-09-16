import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
import argparse
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from task_eval.openai_compat import (
    filter_locomo_samples,
    parse_sample_ids,
    uses_claude_model,
    uses_gemini_model,
    uses_hf_model,
    uses_openai_compat_model,
)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--sample-id', action='append', default=None,
                        help='Restrict to these sample_id values (repeat or comma-separate).')
    parser.add_argument('--qa-per-category', type=int, default=None,
                        help='Keep at most N QA items per category in each sample.')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Chat temperature. Official gpt QA defaults to 0; '
                             'OpenAI-compatible models default to 1.')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Completion token budget. Official gpt single-QA is 32; '
                             'OpenAI-compatible models default to 1024.')
    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if uses_openai_compat_model(args.model):
        # gpt-* and OpenAI-compatible Chat Completions (e.g. kimi-for-coding)
        set_openai_key(args.model)

    elif uses_claude_model(args.model):
        from task_eval.claude_utils import get_claude_answers
        set_anthropic_key()

    elif uses_gemini_model(args.model):
        import google.generativeai as genai
        from task_eval.gemini_utils import get_gemini_answers
        set_gemini_key()
        if args.model == "gemini-pro-1.0":
            model_name = "models/gemini-1.0-pro-latest"

        gemini_model = genai.GenerativeModel(model_name)
    
    elif uses_hf_model(args.model):
        from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
        hf_pipeline, hf_model_name = init_hf_model(args)

    else:
        raise NotImplementedError


    # load conversations
    samples = json.load(open(args.data_file))
    samples = filter_locomo_samples(
        samples,
        sample_ids=parse_sample_ids(args.sample_id),
        qa_per_category=args.qa_per_category,
    )
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}
    if args.sample_id or args.qa_per_category:
        keep = {sample['sample_id'] for sample in samples}
        out_samples = {k: v for k, v in out_samples.items() if k in keep}


    for data in samples:

        out_data = {'sample_id': data['sample_id']}
        prev_qa = out_samples.get(data['sample_id'], {}).get('qa')
        if prev_qa is not None and len(prev_qa) == len(data['qa']):
            out_data['qa'] = prev_qa.copy()
        else:
            out_data['qa'] = data['qa'].copy()

        if uses_openai_compat_model(args.model):
            # official truncated-context prompts + scoring via get_gpt_answers
            from task_eval.gpt_utils import get_gpt_answers
            answers = get_gpt_answers(data, out_data, prediction_key, args)
        elif uses_claude_model(args.model):
            from task_eval.claude_utils import get_claude_answers
            answers = get_claude_answers(data, out_data, prediction_key, args)
        elif uses_gemini_model(args.model):
            from task_eval.gemini_utils import get_gemini_answers
            answers = get_gemini_answers(gemini_model, data, out_data, prediction_key, args)
        elif uses_hf_model(args.model):
            from task_eval.hf_llm_utils import get_hf_answers
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score
        from task_eval.evaluation import eval_question_answering
        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)

        out_samples[data['sample_id']] = answers


    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

    
    from task_eval.evaluation_stats import analyze_aggr_acc
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)
    # encoder=tiktoken.encoding_for_model(args.model))


if __name__ == "__main__":
    main()

