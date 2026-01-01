import json
import os

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_from_journals(journals):
    # Deterministic, minimal extractor (safe)
    preds = []
    for j in journals:
        text = j["text"].lower()
        if "anxious" in text:
            preds.append({
                "journal_id": j["journal_id"],
                "domain": "emotion",
                "evidence_span": "anxious",
                "polarity": "present",
                "intensity_bucket": "medium",
                "time_bucket": "unknown"
            })
    return preds

def load_predictions(data_dir):
    sample_path = os.path.join(data_dir, "sample_predictions.jsonl")
    journals_path = os.path.join(data_dir, "journals.jsonl")

    if os.path.exists(sample_path):
        return load_jsonl(sample_path)
    else:
        journals = load_jsonl(journals_path)
        return extract_from_journals(journals)
