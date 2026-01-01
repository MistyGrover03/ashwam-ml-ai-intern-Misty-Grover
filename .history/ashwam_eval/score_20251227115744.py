def evidence_overlap(a, b):
    return a in b or b in a

def match(pred, gold):
    return (
        pred["journal_id"] == gold["journal_id"] and
        pred["domain"] == gold["domain"] and
        evidence_overlap(pred["evidence_span"], gold["evidence_span"])
    )

def score(preds, golds, journals):
    matched_gold = set()
    TP = FP = FN = 0

    polarity_correct = 0
    bucket_correct = 0

    for p in preds:
        found = False
        for i, g in enumerate(golds):
            if i in matched_gold:
                continue
            if match(p, g):
                TP += 1
                matched_gold.add(i)
                found = True

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

    FN = len(golds) - len(matched_gold)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0

    # Evidence coverage
    journal_map = {j["journal_id"]: j["text"] for j in journals}
    valid_evidence = sum(
        1 for p in preds
        if p["evidence_span"] in journal_map[p["journal_id"]]
    )
    evidence_coverage = valid_evidence / len(preds) if preds else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "polarity_accuracy": polarity_correct / TP if TP else 0,
        "bucket_accuracy": bucket_correct / TP if TP else 0,
        "evidence_coverage": evidence_coverage,
        "tp": TP,
        "fp": FP,
        "fn": FN
    }
