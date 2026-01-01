import json
import os

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_from_journals(journals):
    # Minimal deterministic extractor (fallback only)
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

    preds = []

    if os.path.exists(sample_path):
        rows = load_jsonl(sample_path)
        for row in rows:
            jid = row["journal_id"]
            for obj in row.get("objects", []):
                flat = obj.copy()
                flat["journal_id"] = jid
                preds.append(flat)
        return preds

    journals = load_jsonl(journals_path)
    return extract_from_journals(journals)
