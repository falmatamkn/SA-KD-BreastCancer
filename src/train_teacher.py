"""
Training procedure for the EfficientNetB0 teacher.
"""

import tensorflow as tf
from tensorflow import keras

from src.models import build_teacher


def train_teacher(
    train_ds,
    val_ds,
    stage_a_epochs=10,
    stage_b_epochs=40,
):
    """
    Train the EfficientNetB0 teacher in two stages.

    Stage A:
        Train the classification head while the EfficientNetB0
        backbone remains frozen.

    Stage B:
        Unfreeze the top approximately 30% of the EfficientNetB0
        backbone and fine-tune using a newly created optimizer.

    Early stopping during fine-tuning monitors validation AUC.
    """

    # ========================================================
    # Build Teacher
    # ========================================================

    teacher = build_teacher()

    # ========================================================
    # Stage A
    # ========================================================

    steps_per_epoch = tf.data.experimental.cardinality(
        train_ds
    ).numpy()

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=steps_per_epoch * stage_b_epochs,
    )

    optimizer_a = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
    )

    teacher.compile(
        optimizer=optimizer_a,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    print("\nStage A: Training teacher classification head")

    teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage_a_epochs,
        verbose=1,
    )

    # ========================================================
    # Stage B
    # ========================================================

    print("\nStage B: Fine-tuning top 30% of EfficientNetB0")

    base_model = teacher.get_layer("efficientnetb0")

    base_model.trainable = True

    # Keep the bottom 70% frozen.
    cut = int(len(base_model.layers) * 0.7)

    for layer in base_model.layers[:cut]:
        layer.trainable = False

    # IMPORTANT:
    # Create a new optimizer after changing trainable layers.
    optimizer_b = keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-4,
    )

    teacher.compile(
        optimizer=optimizer_b,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
        run_eagerly=True,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage_b_epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    return teacher, history
