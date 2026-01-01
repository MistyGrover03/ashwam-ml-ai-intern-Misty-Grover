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
    # Restraint: skip predictions without evidence
    preds = [p for p in preds if p.get("evidence_span")]

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

                if p.get("polarity") == g.get("polarity"):
                    polarity_correct += 1

                if (
                    p.get("intensity_bucket") == g.get("intensity_bucket") and
                    p.get("time_bucket") == g.get("time_bucket")
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
    valid_evidence = 0
    for p in preds:
        jid = p.get("journal_id")
        ev = p.get("evidence_span")
        if jid in journal_map and ev in journal_map[jid]:
            valid_evidence += 1

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
