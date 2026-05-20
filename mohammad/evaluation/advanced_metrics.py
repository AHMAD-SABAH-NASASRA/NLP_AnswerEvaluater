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

from paths import FINETUNED_OUTPUT

INPUT_PATH = str(FINETUNED_OUTPUT)


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


y_true = [x["true_score"] for x in data]
y_pred = [x["pred_score"] for x in data]


acc = accuracy_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
qwk = cohen_kappa_score(
    y_true,
    y_pred,
    weights="quadratic",
)


print("\n===== METRICS =====")
print("Accuracy:", round(acc, 4))
print("MAE:", round(mae, 4))
print("QWK:", round(qwk, 4))

print("\n===== CONFUSION MATRIX =====")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
    )
)

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
)