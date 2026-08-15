"""
Logits-based knowledge distillation for the SA-KD framework.

The implementation follows the distillation procedure used in:
"Resource-Efficient Knowledge Distillation via Simulated Annealing
for Lightweight Breast Cancer Detection."
"""

import tensorflow as tf
from tensorflow import keras

from src.models import get_classifier_and_features


class TrueLogitsDistiller(keras.Model):
    """
    Knowledge-distillation model using reconstructed pre-softmax logits.

    Total loss:

        L_total = alpha * L_CE
                + (1 - alpha) * T^2 * L_KL

    where:
        alpha = hard/soft loss balancing coefficient
        T     = distillation temperature
    """

    def __init__(
        self,
        student,
        teacher,
        temperature=4.0,
        alpha=0.7,
    ):
        super().__init__()

        self.student = student
        self.teacher = teacher

        self.temperature = temperature
        self.alpha = alpha

        # ----------------------------------------------------
        # Extract classifier layers and feature models
        # ----------------------------------------------------

        self.s_cls, self.s_feat_model = (
            get_classifier_and_features(student)
        )

        self.t_cls, self.t_feat_model = (
            get_classifier_and_features(teacher)
        )

        # ----------------------------------------------------
        # Loss functions
        # ----------------------------------------------------

        self.ce = keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        )

        self.kld = keras.losses.KLDivergence()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        self.acc = keras.metrics.CategoricalAccuracy(
            name="accuracy"
        )

        self.auc = keras.metrics.AUC(
            name="auc"
        )

    @property
    def metrics(self):
        return [
            self.acc,
            self.auc,
        ]

    # ========================================================
    # Training Step
    # ========================================================

    def train_step(self, data):

        x, y = data

        # ----------------------------------------------------
        # Teacher logits
        # ----------------------------------------------------

        teacher_features = self.t_feat_model(
            x,
            training=False,
        )

        teacher_logits = (
            tf.matmul(
                teacher_features,
                self.t_cls.kernel,
            )
            + self.t_cls.bias
        )

        # ----------------------------------------------------
        # Student forward pass
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            student_probs = self.student(
                x,
                training=True,
            )

            student_features = self.s_feat_model(
                x,
                training=True,
            )

            student_logits = (
                tf.matmul(
                    student_features,
                    self.s_cls.kernel,
                )
                + self.s_cls.bias
            )

            # -----------------------------------------------
            # Hard-label loss
            # -----------------------------------------------

            loss_ce = self.ce(
                y,
                student_probs,
            )

            # -----------------------------------------------
            # Soft-label KD loss
            # -----------------------------------------------

            T = self.temperature

            teacher_soft = tf.nn.softmax(
                teacher_logits / T
            )

            student_soft = tf.nn.softmax(
                student_logits / T
            )

            loss_kd = (
                self.kld(
                    teacher_soft,
                    student_soft,
                )
                * (T ** 2)
            )

            # -----------------------------------------------
            # Combined KD objective
            # -----------------------------------------------

            total_loss = (
                self.alpha * loss_ce
                + (1.0 - self.alpha) * loss_kd
            )

        # ----------------------------------------------------
        # Update student only
        # ----------------------------------------------------

        gradients = tape.gradient(
            total_loss,
            self.student.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.student.trainable_variables,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        self.acc.update_state(
            y,
            student_probs,
        )

        self.auc.update_state(
            y[:, 1],
            student_probs[:, 1],
        )

        return {
            "loss": total_loss,
            "ce": loss_ce,
            "kd": loss_kd,
            "accuracy": self.acc.result(),
            "auc": self.auc.result(),
        }

    # ========================================================
    # Validation / Test Step
    # ========================================================

    def test_step(self, data):

        x, y = data

        student_probs = self.student(
            x,
            training=False,
        )

        self.acc.update_state(
            y,
            student_probs,
        )

        self.auc.update_state(
            y[:, 1],
            student_probs[:, 1],
        )

        return {
            "accuracy": self.acc.result(),
            "auc": self.auc.result(),
        }


# ============================================================
# Helper
# ============================================================

def build_distiller(
    student,
    teacher,
    alpha,
    temperature,
    learning_rate=1e-4,
):
    """
    Construct and compile a KD model for a given (alpha, T).

    The SA, random-search, and grid-search experiments use
    Adam with learning rate 1e-4 for candidate evaluation.
    """

    distiller = TrueLogitsDistiller(
        student=student,
        teacher=teacher,
        temperature=temperature,
        alpha=alpha,
    )

    distiller.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate
        )
    )

    return distiller
