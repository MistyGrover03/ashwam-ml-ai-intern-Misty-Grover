from collections import defaultdict

def evidence_overlap(a, b):
    if not a or not b:
        return False
    return a in b or b in a

def match(pred, gold):
    return (
        pred.get("journal_id") == gold.get("journal_id") and
        pred.get("domain") == gold.get("domain") and
        evidence_overlap(pred.get("evidence_span"), gold.get("evidence_span"))
    )

def score(preds, golds, journals):
    preds = [p for p in preds if p.get("evidence_span")]

    per_journal = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    matched_gold = set()
    TP = FP = FN = 0
    polarity_correct = 0
    bucket_correct = 0

    # Match predictions
    for p in preds:
        found = False
        for i, g in enumerate(golds):
            if i in matched_gold:
                continue
            if match(p, g):
                found = True
                matched_gold.add(i)

                TP += 1
                per_journal[p["journal_id"]]["tp"] += 1

                if p["polarity"] == g["polarity"]:
                    polarity_correct += 1
                if (
                    p["intensity_bucket"] == g["intensity_bucket"] and
                    p["time_bucket"] == g["time_bucket"]
                ):
                    bucket_correct += 1
                break

        if not found:
            FP += 1
            per_journal[p["journal_id"]]["fp"] += 1

    # Count FN per journal
    for i, g in enumerate(golds):
        if i not in matched_gold:
            FN += 1
            per_journal[g["journal_id"]]["fn"] += 1

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0

    journal_map = {j["journal_id"]: j["text"] for j in journals}
    valid_evidence = sum(
        1 for p in preds
        if p["evidence_span"] in journal_map.get(p["journal_id"], "")
    )

    evidence_coverage = valid_evidence / len(preds) if preds else 0

    return {
        "summary": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "polarity_accuracy": polarity_correct / TP if TP else 0,
            "bucket_accuracy": bucket_correct / TP if TP else 0,
            "evidence_coverage": evidence_coverage,
        },
        "per_journal": per_journal
    }
