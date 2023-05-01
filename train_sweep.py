from constants import TASKS
from train import main

"""
tasks0: list of tasks from TASKS used to train the network using full plasticity
tasks1: list of tasks from TASKS used to train the network using sparse plasticity after tasks0 is trained
"""

tasks0 = [
    "task_go",
    "task_go_rt",
    "task_go_delay",
]
tasks1 = [t for t in TASKS if t not in tasks0]

tasks1 = ["decision_making_rf0"]

main(tasks0, tasks1)