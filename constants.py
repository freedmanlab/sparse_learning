
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
    "decision_making",
    "ctx_decision_making",
    "decision_making_multisensory",
    "delayed_decision_making",
    "delayed_ctx_decision_making",
    "delayed_decision_making_multisensory",
    "delayed_match_sample",
    "delayed_match_category",
    "delayed_match_sample_delayed_response",
    "delayed_match_category_delayed_response",
]

train_params = {
    "learning_rate": 0.002,
    "weight_decay": 1e-5,
    "n_full_plasticity_epochs": 2,
    "n_sparse_plasticity_epochs": 5,
    "batch_size": 512,
    "num_workers": 16,
    "RL": False,
    "seed": 42,
}

model_params = {
    "n_hidden": 500,
    "tau_neuron": 100.0,
    "tau_slow": 1000.0,
    "tau_fast": 200.0,
    "dt": 20.0,
    "exc_fraction": 0.8,
    "noise_std": 0.01,
    "batch_size": train_params["batch_size"]
}

