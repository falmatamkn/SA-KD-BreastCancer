"""
Alternative KD hyperparameter search baselines.

Implements the budget-matched random search and grid search
used for comparison with simulated annealing.
"""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from configs.config import (
    ALPHA_MIN,
    ALPHA_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    SA_MAX_EVALUATIONS,
    SA_CANDIDATE_EPOCHS,
)

from src.models import build_student
from src.distillation import build_distiller


# ============================================================
# Candidate Evaluation
# ============================================================

def evaluate_candidate(
    teacher,
    baseline_weights,
    train_ds,
    val_ds,
    alpha,
    temperature,
):
    """
    Train one KD candidate from the fixed supervised baseline
    weights and return its final validation AUC.
    """

    student = build_student()

    student.set_weights(baseline_weights)

    distiller = build_distiller(
        student=student,
        teacher=teacher,
        alpha=alpha,
        temperature=temperature,
        learning_rate=1e-4,
    )

    history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=SA_CANDIDATE_EPOCHS,
        verbose=0,
    )

    val_auc = float(
        history.history["val_auc"][-1]
    )

    return val_auc, student


# ============================================================
# Random Search
# ============================================================

def random_search(
    teacher,
    baseline_student,
    train_ds,
    val_ds,
    seed=42,
):
    """
    Perform budget-matched random search over alpha and T.

    alpha ~ Uniform(0, 1)
    T     ~ Uniform(1, 10)

    A total of 30 candidate configurations are evaluated.
    """

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    baseline_weights = [
        w.copy()
        for w in baseline_student.get_weights()
    ]

    best_val_auc = -np.inf
    best_alpha = None
    best_temperature = None
    best_weights = None

    results = []

    for iteration in range(
        1,
        SA_MAX_EVALUATIONS + 1,
    ):

        alpha = random.uniform(
            ALPHA_MIN,
            ALPHA_MAX,
        )

        temperature = random.uniform(
            TEMPERATURE_MIN,
            TEMPERATURE_MAX,
        )

        val_auc, student = evaluate_candidate(
            teacher=teacher,
            baseline_weights=baseline_weights,
            train_ds=train_ds,
            val_ds=val_ds,
            alpha=alpha,
            temperature=temperature,
        )

        results.append(
            {
                "iteration": iteration,
                "alpha": float(alpha),
                "temperature": float(temperature),
                "val_auc": float(val_auc),
            }
        )

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_alpha = alpha
            best_temperature = temperature

            best_weights = [
                w.copy()
                for w in student.get_weights()
            ]

            tag = "NEW BEST"

        else:
            tag = ""

        print(
            f"Random {iteration:02d}/"
            f"{SA_MAX_EVALUATIONS} | "
            f"val_auc={val_auc:.4f} | "
            f"alpha={alpha:.4f} | "
            f"T={temperature:.4f} | "
            f"{tag}"
        )

    return {
        "method": "random_search",
        "seed": seed,
        "best_alpha": float(best_alpha),
        "best_temperature": float(best_temperature),
        "best_val_auc": float(best_val_auc),
        "best_weights": best_weights,
        "results": results,
    }


# ============================================================
# Grid Search
# ============================================================

def grid_search(
    teacher,
    baseline_student,
    train_ds,
    val_ds,
    seed=42,
):
    """
    Perform the 30-point coarse grid search used in the
    experiments.

    Grid:
        6 alpha values between 0.1 and 0.9
        5 temperature values between 1 and 10

    Total:
        6 x 5 = 30 evaluations.
    """

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --------------------------------------------------------
    # Exact grid used in the experiment
    # --------------------------------------------------------

    alpha_grid = np.linspace(
        0.1,
        0.9,
        6,
    )

    temperature_grid = np.linspace(
        1.0,
        10.0,
        5,
    )

    grid_points = [
        (float(alpha), float(temperature))
        for alpha in alpha_grid
        for temperature in temperature_grid
    ]

    assert len(grid_points) == SA_MAX_EVALUATIONS

    baseline_weights = [
        w.copy()
        for w in baseline_student.get_weights()
    ]

    best_val_auc = -np.inf
    best_alpha = None
    best_temperature = None
    best_weights = None

    results = []

    # --------------------------------------------------------
    # Evaluate grid
    # --------------------------------------------------------

    for iteration, (
        alpha,
        temperature,
    ) in enumerate(
        grid_points,
        start=1,
    ):

        # Deterministic seed used for each grid candidate.
        trial_seed = seed * 100_000 + iteration

        keras.utils.set_random_seed(
            trial_seed
        )

        val_auc, student = evaluate_candidate(
            teacher=teacher,
            baseline_weights=baseline_weights,
            train_ds=train_ds,
            val_ds=val_ds,
            alpha=alpha,
            temperature=temperature,
        )

        results.append(
            {
                "iteration": iteration,
                "alpha": float(alpha),
                "temperature": float(temperature),
                "val_auc": float(val_auc),
                "seed": int(trial_seed),
            }
        )

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_alpha = alpha
            best_temperature = temperature

            best_weights = [
                w.copy()
                for w in student.get_weights()
            ]

            tag = "NEW BEST"

        else:
            tag = ""

        print(
            f"Grid {iteration:02d}/"
            f"{SA_MAX_EVALUATIONS} | "
            f"val_auc={val_auc:.4f} | "
            f"alpha={alpha:.4f} | "
            f"T={temperature:.4f} | "
            f"{tag}"
        )

    return {
        "method": "grid_search",
        "seed": seed,
        "best_alpha": float(best_alpha),
        "best_temperature": float(best_temperature),
        "best_val_auc": float(best_val_auc),
        "best_weights": best_weights,
        "alpha_grid": [
            float(x)
            for x in alpha_grid
        ],
        "temperature_grid": [
            float(x)
            for x in temperature_grid
        ],
        "results": results,
    }
