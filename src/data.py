"""
Dataset loading and preprocessing utilities for the BreakHis 400x dataset.

The experimental setup follows the AFRICAI @ MICCAI 2026 paper:
"Resource-Efficient Knowledge Distillation via Simulated Annealing
for Lightweight Breast Cancer Detection."
"""

import tensorflow as tf

from configs.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
)


AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
    ],
    name="data_augmentation",
)


# ============================================================
# Dataset Loading
# ============================================================

def load_datasets(train_dir, test_dir, seed=42):
    """
    Load the BreakHis 400x training, validation, and test datasets.

    The training directory is split into:
        90% training
        10% validation

    The test directory remains completely separate and is used only
    for final model evaluation.

    Expected directory structure:

        train_dir/
            benign/
            malignant/

        test_dir/
            benign/
            malignant/

    Parameters
    ----------
    train_dir : str
        Path to the pre-split BreakHis training directory.

    test_dir : str
        Path to the pre-split BreakHis test directory.

    seed : int
        Random seed used to reproduce the training/validation split.

    Returns
    -------
    train_ds, val_ds, test_ds
        TensorFlow datasets with integer class labels.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


# ============================================================
# Student Preprocessing
# ============================================================

def preprocess_student(images, labels, training=False):
    """
    Preprocess images for the lightweight student.

    Student inputs are normalized to [0, 1] using 1/255 rescaling.
    Data augmentation is applied only during training.
    """

    images = tf.cast(images, tf.float32)

    if training:
        images = data_augmentation(images, training=True)

    images = images / 255.0

    return images, labels


# ============================================================
# Teacher Preprocessing
# ============================================================

def preprocess_teacher(images, labels, training=False):
    """
    Preprocess images for EfficientNetB0.

    Data augmentation is applied only during training.

    TensorFlow/Keras EfficientNetB0 contains its input rescaling
    preprocessing as part of the model implementation, so images
    remain in their standard [0, 255] representation here.
    """

    images = tf.cast(images, tf.float32)

    if training:
        images = data_augmentation(images, training=True)

    return images, labels


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_student_datasets(train_ds, val_ds, test_ds):
    """
    Prepare datasets for training/evaluating the lightweight student.
    """

    train_ds = train_ds.map(
        lambda x, y: preprocess_student(x, y, training=True),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x, y: preprocess_student(x, y, training=False),
        num_parallel_calls=AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda x, y: preprocess_student(x, y, training=False),
        num_parallel_calls=AUTOTUNE,
    )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )


def prepare_teacher_datasets(train_ds, val_ds, test_ds):
    """
    Prepare datasets for training/evaluating EfficientNetB0.
    """

    train_ds = train_ds.map(
        lambda x, y: preprocess_teacher(x, y, training=True),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x, y: preprocess_teacher(x, y, training=False),
        num_parallel_calls=AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda x, y: preprocess_teacher(x, y, training=False),
        num_parallel_calls=AUTOTUNE,
    )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )
