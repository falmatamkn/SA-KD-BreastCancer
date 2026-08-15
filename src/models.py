"""
Teacher and student model architectures for SA-KD.

The architectures correspond to the models used in:
"Resource-Efficient Knowledge Distillation via Simulated Annealing
for Lightweight Breast Cancer Detection."
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from configs.config import INPUT_SHAPE


# ============================================================
# EfficientNetB0 Teacher
# ============================================================

def build_teacher(input_shape=INPUT_SHAPE, num_classes=2):
    """
    Build the ImageNet-pretrained EfficientNetB0 teacher.

    The EfficientNetB0 backbone is initially frozen.
    """

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="teacher_input")

    x = preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="teacher_classifier"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="teacher_efficientnetb0"
    )

    return model


# ============================================================
# Lightweight CNN Student
# ============================================================

def build_student(input_shape=INPUT_SHAPE, num_classes=2):
    """
    Build the lightweight CNN student used for knowledge distillation.
    """

    inputs = keras.Input(
        shape=input_shape,
        name="student_input"
    )

    # Block 1
    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="student_classifier"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="student_cnn"
    )

    return model


# ============================================================
# Logit Extraction
# ============================================================

def get_classifier_and_features(model):
    """
    Return the final two-class Dense classifier and a feature model
    exposing the input to that classifier.

    This allows the KD implementation to reconstruct true pre-softmax
    logits from:

        logits = features @ W + b
    """

    classifier = None

    for layer in reversed(model.layers):
        if isinstance(layer, layers.Dense) and layer.units == 2:
            classifier = layer
            break

    if classifier is None:
        raise ValueError(
            "Could not find a Dense classifier with 2 output units."
        )

    feature_model = keras.Model(
        inputs=model.input,
        outputs=classifier.input,
        name=f"{model.name}_features"
    )

    return classifier, feature_model
