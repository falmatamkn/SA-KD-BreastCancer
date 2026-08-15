"""
Configuration for SA-KD Breast Cancer Classification.

Experimental settings follow the AFRICAI @ MICCAI 2026 paper:
"Resource-Efficient Knowledge Distillation via Simulated Annealing
for Lightweight Breast Cancer Detection."
"""

# ============================================================
# Dataset
# ============================================================

IMAGE_SIZE = (320, 320)
INPUT_SHAPE = (320, 320, 3)

BATCH_SIZE = 16

# 10% of the training set is held out for validation
VALIDATION_SPLIT = 0.10


# ============================================================
# Reproducibility
# ============================================================

SEEDS = [42, 50, 100]


# ============================================================
# Knowledge Distillation Search Space
# ============================================================

ALPHA_MIN = 0.0
ALPHA_MAX = 1.0

TEMPERATURE_MIN = 1.0
TEMPERATURE_MAX = 10.0


# ============================================================
# Simulated Annealing
# ============================================================

SA_INITIAL_TEMPERATURE = 1000.0
SA_COOLING_RATE = 0.95
SA_MIN_TEMPERATURE = 1e-8

# Maximum number of candidate evaluations
SA_MAX_EVALUATIONS = 30

# Stop after this many evaluations without improvement
SA_PATIENCE = 10

# Each candidate KD configuration is evaluated for 5 epochs
SA_CANDIDATE_EPOCHS = 5


# ============================================================
# SA Proposal Configurations
# ============================================================

# Small Move (SM)
SM_ALPHA_STEP = 0.01
SM_TEMPERATURE_STEP = 0.10

# Big Move (BM)
BM_ALPHA_STEP = 0.05
BM_TEMPERATURE_STEP = 0.50


# ============================================================
# Evaluation
# ============================================================

# Fixed decision threshold used for final test-set metrics
CLASSIFICATION_THRESHOLD = 0.5
