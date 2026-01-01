import argparse
import json
import os

from ashwam_eval.extract import load_jsonl, load_predictions
from ashwam_eval.score import score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    journals = load_jsonl(os.path.join(args.data, "journals.jsonl"))
    gold_rows = load_jsonl(os.path.join(args.data, "gold.jsonl"))
    preds = load_predictions(args.data)

    scores = score(preds, gold_rows, journals)

    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "score_summary.json"), "w") as f:
        json.dump(scores["summary"], f, indent=2)

    with open(os.path.join(args.out, "per_journal_scores.jsonl"), "w") as f:
        for jid, vals in scores["per_journal"].items():
            f.write(json.dumps({
                "journal_id": jid,
                "tp": vals["tp"],
                "fp": vals["fp"],
                "fn": vals["fn"]
            }) + "\n")

if __name__ == "__main__":
    main()
