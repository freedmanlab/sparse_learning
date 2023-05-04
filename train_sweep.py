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
    "decision_making_rf0",
    "decision_making_rf1",
]

other_tasks = TASKS[16:]

for t in other_tasks:
    print()
    print("*******************************************************")
    print(t)
    main(tasks0, [t])