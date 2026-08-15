"""
Supervised training procedure for the lightweight CNN student.

This model provides the non-distilled student baseline.
"""

from tensorflow import keras

from src.models import build_student


def train_baseline_student(
    train_ds,
    val_ds,
    epochs=20,
):
    """
    Train the lightweight student directly from ground-truth
    labels without knowledge distillation.
    """

    student = build_student()

    optimizer = keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-4,
    )

    student.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(
                name="acc"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
        run_eagerly=True,
    )

    history = student.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
    )

    return student, history
