
import numpy as np
#import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("--> Loading parameters...")

"""
Independent parameters
"""

TASKS = [
    "task_go",
    "task_go_rt",
    "task_go_delay",
    "decision_making_rf0",
    "decision_making_rf1",
    "ctx_decision_making_rf0",
    "ctx_decision_making_rf1",
    "decision_making_multisensory",
    "delayed_decision_making_rf0",
    "delayed_decision_making_rf1",
    "delayed_ctx_decision_making_rf0",
    "delayed_ctx_decision_making_rf1",
    "delayed_decision_making_multisensory",
    "delayed_match_sample",
    "delayed_match_category",
    "non_delayed_match_sample",
    "non_delayed_match_category",
]

train_params = {
    "learning_rate": 0.002,
    "num_train_epochs": 8,
    "num_test_epochs": 4,
    "batch_size": 512,
    "num_workers": 16,
    "RL": False,
}

model_params = {
    "hidden_dim": 512,
}
