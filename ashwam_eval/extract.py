import json
import os

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_predictions(data_dir):
    """
    Flattens sample_predictions.jsonl:
    { journal_id, items: [...] } → flat objects with journal_id
    """
    sample_path = os.path.join(data_dir, "sample_predictions.jsonl")
    preds = []

    if not os.path.exists(sample_path):
        return preds

    rows = load_jsonl(sample_path)
    for row in rows:
        jid = row["journal_id"]
        for item in row["items"]:
            flat = item.copy()
            flat["journal_id"] = jid
            preds.append(flat)

    return preds
