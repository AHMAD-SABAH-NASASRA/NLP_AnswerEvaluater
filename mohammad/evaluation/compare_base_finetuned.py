import json
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paths import (
    BASE_QWEN_OUTPUT,
    FINETUNED_OUTPUT,
)

BASE_PATH = str(BASE_QWEN_OUTPUT)
FT_PATH = str(FINETUNED_OUTPUT)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(data):
    total = len(data)

    parsed = [
        x for x in data
        if x["true_score"] is not None
        and x["pred_score"] is not None
    ]

    failed = total - len(parsed)

    correct_all = sum(
        1
        for x in data
        if x["pred_score"] is not None
        and x["pred_score"] == x["true_score"]
    )

    accuracy_all = correct_all / total

    y_true = [x["true_score"] for x in parsed]
    y_pred = [x["pred_score"] for x in parsed]

    result = {
        "total": total,
        "parsed": len(parsed),
        "failed": failed,
        "accuracy_all": accuracy_all,
        "accuracy_parsed": None,
        "mae_parsed": None,
        "qwk_parsed": None,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    if len(parsed) > 0:
        result["accuracy_parsed"] = accuracy_score(
            y_true,
            y_pred,
        )

        result["mae_parsed"] = mean_absolute_error(
            y_true,
            y_pred,
        )

        result["qwk_parsed"] = cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic",
        )

    return result


base_data = load_data(BASE_PATH)
ft_data = load_data(FT_PATH)

if len(base_data) != len(ft_data):
    raise ValueError(
        f"Different number of samples: "
        f"base={len(base_data)}, "
        f"finetuned={len(ft_data)}"
    )

base = compute_metrics(base_data)
ft = compute_metrics(ft_data)

print("\n================ BASE QWEN vs FINETUNED QWEN ================")

print("\nMetric                     Base Qwen        Fine-tuned Qwen")
print("-------------------------------------------------------------")

print(
    f"Total samples              "
    f"{base['total']:<16} "
    f"{ft['total']}"
)

print(
    f"Parsed samples             "
    f"{base['parsed']:<16} "
    f"{ft['parsed']}"
)

print(
    f"Failed samples             "
    f"{base['failed']:<16} "
    f"{ft['failed']}"
)

print(
    f"Accuracy all samples       "
    f"{base['accuracy_all']:<16.4f} "
    f"{ft['accuracy_all']:.4f}"
)

if (
    base["accuracy_parsed"] is not None
    and ft["accuracy_parsed"] is not None
):
    print(
        f"Accuracy parsed only       "
        f"{base['accuracy_parsed']:<16.4f} "
        f"{ft['accuracy_parsed']:.4f}"
    )

    print(
        f"MAE parsed only            "
        f"{base['mae_parsed']:<16.4f} "
        f"{ft['mae_parsed']:.4f}"
    )

    print(
        f"QWK parsed only            "
        f"{base['qwk_parsed']:<16.4f} "
        f"{ft['qwk_parsed']:.4f}"
    )

print("\n================ IMPROVEMENT ================")

print(
    "Accuracy all improvement:",
    round(
        ft["accuracy_all"] - base["accuracy_all"],
        4,
    ),
)

if (
    base["mae_parsed"] is not None
    and ft["mae_parsed"] is not None
):
    print(
        "MAE reduction:",
        round(
            base["mae_parsed"] - ft["mae_parsed"],
            4,
        ),
    )

if (
    base["qwk_parsed"] is not None
    and ft["qwk_parsed"] is not None
):
    print(
        "QWK improvement:",
        round(
            ft["qwk_parsed"] - base["qwk_parsed"],
            4,
        ),
    )

print("\n================ FINETUNED CONFUSION MATRIX ================")

print(
    confusion_matrix(
        ft["y_true"],
        ft["y_pred"],
        labels=[0, 1, 2, 3, 4],
    )
)

print("\n================ FINETUNED CLASSIFICATION REPORT ================")

print(
    classification_report(
        ft["y_true"],
        ft["y_pred"],
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
)