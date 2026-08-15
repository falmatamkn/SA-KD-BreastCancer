"""
Final evaluation utilities for the SA-KD framework.

The held-out test set is used only after hyperparameter selection.
"""

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from configs.config import CLASSIFICATION_THRESHOLD


def collect_predictions(model, test_ds):
    """
    Collect true labels and positive-class probabilities
    from the held-out test set.
    """

    y_true = []
    y_prob = []

    for images, labels in test_ds:

        probabilities = model.predict(
            images,
            verbose=0,
        )

        # One-hot labels -> binary labels
        true_labels = np.argmax(
            labels.numpy(),
            axis=1,
        )

        # Probability of malignant / positive class
        positive_probabilities = probabilities[:, 1]

        y_true.extend(true_labels)
        y_prob.extend(positive_probabilities)

    return (
        np.asarray(y_true),
        np.asarray(y_prob),
    )


def evaluate_model(
    model,
    test_ds,
    threshold=CLASSIFICATION_THRESHOLD,
):
    """
    Evaluate a trained model on the held-out test set.

    ROC-AUC is calculated from continuous positive-class
    probabilities.

    Accuracy, F1, precision, sensitivity/recall and specificity
    are calculated after applying the fixed decision threshold.
    """

    y_true, y_prob = collect_predictions(
        model,
        test_ds,
    )

    # --------------------------------------------------------
    # Threshold-independent metric
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    # --------------------------------------------------------
    # Fixed-threshold predictions
    # --------------------------------------------------------

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    results = {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return results


def print_results(results):
    """
    Print final test metrics in a readable format.
    """

    print("\nFinal Test Results")
    print("------------------")

    print(
        f"AUC:         "
        f"{results['auc']:.4f}"
    )

    print(
        f"Accuracy:    "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print(
        f"F1-score:    "
        f"{results['f1'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{results['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:      "
        f"{results['recall'] * 100:.2f}%"
    )

    print(
        f"Sensitivity: "
        f"{results['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{results['specificity'] * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN={results['tn']}  "
        f"FP={results['fp']}"
    )

    print(
        f"FN={results['fn']}  "
        f"TP={results['tp']}"
    )
