from collections import defaultdict


def evaluate(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["condition"]].append(record)
    report = {}
    for condition, items in groups.items():
        matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "invalid": 0}
        total = correct = count_exact = count_error = 0
        labeled = 0
        for record in items:
            truth = record.get("truth")
            if truth is None:
                continue
            labeled += 1
            prediction = record.get("prediction")
            for index, label in enumerate(truth):
                total += 1
                if prediction is None or index >= len(prediction["images"]):
                    matrix["invalid"] += 1
                    continue
                actual_count = label["count"]
                predicted_count = prediction["images"][index]["count"]
                key = ("tp" if predicted_count > 0 else "fn") if actual_count > 0 else ("fp" if predicted_count > 0 else "tn")
                matrix[key] += 1
                correct += (actual_count > 0) == (predicted_count > 0)
                count_exact += actual_count == predicted_count
                count_error += abs(actual_count - predicted_count)
        valid = total - matrix["invalid"]
        report[condition] = {
            "samples": len(items), "labeled_samples": labeled, "labeled_images": total,
            "confusion_matrix": matrix,
            "accuracy": correct / total if total else None,
            "error_rate": (total - correct) / total if total else None,
            "precision": matrix["tp"] / (matrix["tp"] + matrix["fp"]) if matrix["tp"] + matrix["fp"] else None,
            "recall": matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else None,
            "f1": 2 * matrix["tp"] / (2 * matrix["tp"] + matrix["fp"] + matrix["fn"]) if 2 * matrix["tp"] + matrix["fp"] + matrix["fn"] else None,
            "valid_response_rate": valid / total if total else None,
            "count_exact_rate": count_exact / total if total else None,
            "count_mae": count_error / valid if valid else None,
        }
    return report
