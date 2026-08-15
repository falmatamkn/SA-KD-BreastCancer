"""
Simulated annealing for KD hyperparameter optimization.

Supports the three SA configurations evaluated in the paper:

    FS - Full Search
    SM - Small Move
    BM - Big Move
"""

import math
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from configs.config import (
    ALPHA_MIN,
    ALPHA_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    SA_INITIAL_TEMPERATURE,
    SA_COOLING_RATE,
    SA_MIN_TEMPERATURE,
    SA_MAX_EVALUATIONS,
    SA_PATIENCE,
    SA_CANDIDATE_EPOCHS,
    SM_ALPHA_STEP,
    SM_TEMPERATURE_STEP,
    BM_ALPHA_STEP,
    BM_TEMPERATURE_STEP,
)

from src.models import build_student
from src.distillation import build_distiller


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducible SA runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Candidate Proposal
# ============================================================

def propose_candidate(
    current_alpha,
    current_temperature,
    configuration,
):
    """
    Generate a new (alpha, T) candidate.

    FS:
        Sample independently from the complete search space.

    SM:
        Local move with delta_alpha=0.01 and delta_T=0.10.

    BM:
        Local move with delta_alpha=0.05 and delta_T=0.50.
    """

    configuration = configuration.upper()

    if configuration == "FS":

        new_alpha = random.uniform(
            ALPHA_MIN,
            ALPHA_MAX,
        )

        new_temperature = random.uniform(
            TEMPERATURE_MIN,
            TEMPERATURE_MAX,
        )

    elif configuration == "SM":

        new_alpha = random.uniform(
            current_alpha - SM_ALPHA_STEP,
            current_alpha + SM_ALPHA_STEP,
        )

        new_temperature = random.uniform(
            current_temperature - SM_TEMPERATURE_STEP,
            current_temperature + SM_TEMPERATURE_STEP,
        )

    elif configuration == "BM":

        new_alpha = random.uniform(
            current_alpha - BM_ALPHA_STEP,
            current_alpha + BM_ALPHA_STEP,
        )

        new_temperature = random.uniform(
            current_temperature - BM_TEMPERATURE_STEP,
            current_temperature + BM_TEMPERATURE_STEP,
        )

    else:
        raise ValueError(
            "configuration must be 'FS', 'SM', or 'BM'."
        )

    # Keep candidate inside the search space.
    new_alpha = np.clip(
        new_alpha,
        ALPHA_MIN,
        ALPHA_MAX,
    )

    new_temperature = np.clip(
        new_temperature,
        TEMPERATURE_MIN,
        TEMPERATURE_MAX,
    )

    return float(new_alpha), float(new_temperature)


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
    Evaluate one KD hyperparameter configuration.

    Every candidate starts from the same baseline student weights
    and is trained for five epochs.

    Validation AUC is the SA objective.
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

    val_auc = history.history["val_auc"][-1]

    return float(val_auc)


# ============================================================
# Simulated Annealing
# ============================================================

def simulated_annealing(
    teacher,
    baseline_student,
    train_ds,
    val_ds,
    seed=42,
    configuration="SM",
):
    """
    Optimize KD alpha and temperature using simulated annealing.

    Parameters
    ----------
    teacher :
        Trained EfficientNetB0 teacher.

    baseline_student :
        Supervised lightweight student. Every candidate starts
        from these same baseline weights.

    train_ds :
        Training dataset.

    val_ds :
        Validation dataset used to calculate the SA objective.

    seed : int
        Random seed.

    configuration : str
        "FS", "SM", or "BM".

    Returns
    -------
    dict
        Best alpha, temperature, validation AUC and optimization
        history.
    """

    set_seed(seed)

    configuration = configuration.upper()

    baseline_weights = baseline_student.get_weights()

    # --------------------------------------------------------
    # Initial SA state
    # --------------------------------------------------------

    sa_temperature = SA_INITIAL_TEMPERATURE

    current_temperature = random.uniform(
        TEMPERATURE_MIN,
        TEMPERATURE_MAX,
    )

    current_alpha = random.uniform(
        ALPHA_MIN,
        ALPHA_MAX,
    )

    current_val_auc = evaluate_candidate(
        teacher=teacher,
        baseline_weights=baseline_weights,
        train_ds=train_ds,
        val_ds=val_ds,
        alpha=current_alpha,
        temperature=current_temperature,
    )

    best_alpha = current_alpha
    best_temperature = current_temperature
    best_val_auc = current_val_auc

    no_improve_count = 0
    iteration = 0

    history = []

    print(
        f"\nInitial state | "
        f"T={current_temperature:.4f} | "
        f"alpha={current_alpha:.4f} | "
        f"ValAUC={current_val_auc:.4f}"
    )

    # --------------------------------------------------------
    # SA optimization
    # --------------------------------------------------------

    while (
        sa_temperature > SA_MIN_TEMPERATURE
        and no_improve_count < SA_PATIENCE
        and iteration < SA_MAX_EVALUATIONS
    ):

        iteration += 1

        new_alpha, new_temperature = propose_candidate(
            current_alpha=current_alpha,
            current_temperature=current_temperature,
            configuration=configuration,
        )

        new_val_auc = evaluate_candidate(
            teacher=teacher,
            baseline_weights=baseline_weights,
            train_ds=train_ds,
            val_ds=val_ds,
            alpha=new_alpha,
            temperature=new_temperature,
        )

        # ----------------------------------------------------
        # SA acceptance rule
        # ----------------------------------------------------

        delta = new_val_auc - current_val_auc

        accepted = False

        if (
            delta > 0
            or random.random()
            < math.exp(delta / sa_temperature)
        ):
            current_alpha = new_alpha
            current_temperature = new_temperature
            current_val_auc = new_val_auc

            accepted = True

        # ----------------------------------------------------
        # Update best configuration
        # ----------------------------------------------------

        if current_val_auc > best_val_auc:

            best_val_auc = current_val_auc
            best_alpha = current_alpha
            best_temperature = current_temperature

            no_improve_count = 0

        else:
            no_improve_count += 1

        # ----------------------------------------------------
        # Store optimization history
        # ----------------------------------------------------

        history.append(
            {
                "iteration": iteration,
                "sa_temperature": float(sa_temperature),
                "alpha": float(current_alpha),
                "temperature": float(current_temperature),
                "val_auc": float(current_val_auc),
                "delta": float(delta),
                "accepted": accepted,
            }
        )

        print(
            f"[{configuration} | Seed {seed}] "
            f"Iter {iteration:02d} | "
            f"T={current_temperature:.4f} | "
            f"alpha={current_alpha:.4f} | "
            f"ValAUC={current_val_auc:.4f}"
        )

        # Geometric cooling
        sa_temperature *= SA_COOLING_RATE

    return {
        "configuration": configuration,
        "seed": seed,
        "best_alpha": float(best_alpha),
        "best_temperature": float(best_temperature),
        "best_val_auc": float(best_val_auc),
        "history": history,
    }
