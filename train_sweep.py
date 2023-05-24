from typing import Any, Literal, List, Optional, Sequence, Tuple
import os
import pickle
import torch
import pandas as pd
import numpy as np
from constants import TASKS, model_params, train_params
from train import Train

class Sweep:

    def __init__(
        self,
        network: Literal["RNN, LSTM"],
        full_plasticity_tasks: List[List[Any]],
        sparse_plasticity_tasks: List[Any],
        logs_fn: Sequence[str] = "./logs/train_sweep/version_0/metrics.csv",
        save_fn: Sequence[str] = "./logs/results.pkl",
        dir_offset: Tuple[int] = [0],
        target_offset: Tuple[int] = [0],
    ):

        self.logs_fn = logs_fn
        self.save_fn = save_fn
        self.results = {
            "config": {
                "train_params": train_params,
                "model_params": model_params,
                "network": network,
            }
        }  # results will be saved here

        for full_tasks in full_plasticity_tasks:
            trainer = Train(network=network)
            trainer.define_optimizer(sparse_plasticity=False)
            print()
            print(f"*********FULL PLASTICITY: {full_tasks}")
            task_codes = create_task_set(full_tasks, dir_offset=dir_offset, target_offset=target_offset)
            trainer.create_task_loaders(task_codes, n_batches_per_epoch=50)
            trainer.train_model(train_params["n_full_plasticity_epochs"], patience=4)
            # save results
            self.add_results(full_tasks, None, "full")

            for sparse_tasks in sparse_plasticity_tasks:
                print(f"*********SPARSE PLASTICITY: {sparse_tasks}")
                trainer.network.reset_context_weights()
                task_codes = create_task_set([sparse_tasks], dir_offset=[0], target_offset=[0])
                trainer.define_optimizer(sparse_plasticity=True)
                trainer.create_task_loaders(task_codes, n_batches_per_epoch=25)
                trainer.train_model(train_params["n_sparse_plasticity_epochs"], patience=3)
                # save results
                self.add_results(full_tasks, [sparse_tasks], "sparse")

    def add_results(
        self,
        full_tasks: List[Any],
        sparse_tasks: Optional[List[Any]] = None,
        plasticity: Literal["full", "sparse"] = "full",
    ):
        assert plasticity == "full" or (
            plasticity == "sparse" and sparse_tasks is not None, "Must specify sparse tasks"
        )
        dec_acc, task_acc, loss = self.extract_results()
        full_key = "FULL:" + ",".join(full_tasks)
        result = {
            "dec_accuracy": dec_acc,
            "task_accuracy": task_acc,
            "loss": loss,
            "final_dec_accuracy": dec_acc[-1],
            "final_task_accuracy": task_acc[-1],
            "final_loss": loss[-1],
        }
        if plasticity == "full":
            self.results[full_key] = result
        else:
            sparse_key = "SPARSE:" + ",".join(sparse_tasks)
            self.results[full_key][sparse_key] = result

        pickle.dump(self.results, open(self.save_fn, "wb"))

    def extract_results(self):
        df = pd.read_csv(self.logs_fn)
        dec_acc = df.dec_acc.values
        dec_acc = dec_acc[~np.isnan(dec_acc)]
        task_acc = df.task_acc.values
        task_acc = task_acc[~np.isnan(task_acc)]
        train_loss = df.train_loss_epoch.values
        train_loss = train_loss[~np.isnan(train_loss)]

        return dec_acc, task_acc, train_loss

def generate_task_variants(task: str, target_offset: Tuple[int], dir_offset: Tuple[int]):
    task_variants = []
    for to in target_offset:
        if "delayed_match" in task:
            for do in dir_offset:
                task_variants.append([task, do, to])
        else:
            task_variants.append([task, to])
    return task_variants

def create_task_set(
    task_family: List,
    target_offset: Tuple[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    dir_offset: Tuple[int] = [0, 1, 2],
):
    task_set = []
    for task in task_family:
        task_set += generate_task_variants(task, target_offset, dir_offset)

    return task_set

def split_train_set(tasks0: List, tasks1: List, train_pct: float = 0.9):
    """NOT IN USE"""

    available_tasks = get_available_tasks()
    train_set_0 = []

    if train_pct is not None:
        n = int(train_pct * len(available_tasks))
        np.random.shuffle(available_tasks)
        idx = np.arange(len(available_tasks))

        idx0, idx1 = idx[:n], idx[n:]
        train_set_0 = [available_tasks[i] for i in idx0]
        train_set_1 = [available_tasks[i] for i in idx1]
    else:
        train_set_0 = [t for t in available_tasks if holdout_task not in t[0]]
        train_set_1 = [t for t in available_tasks if holdout_task in t[0]]

    return train_set_0, train_set_1

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")




"""
Train with full plasticity with a single task family
Then, train with sparse plasticity on another task family
Perform this for all pairs of task families:
1) 3 seeds, using a single target offset
2) 3 seeds, using all eight target offset
"""
full_plasticity_tasks = [[t] for t in TASKS]
sparse_plasticity_tasks = TASKS

for n in range(0, 2):

    set_seed(42+n)
    sweep = Sweep(
        network="RNN",
        full_plasticity_tasks=full_plasticity_tasks,
        sparse_plasticity_tasks=sparse_plasticity_tasks,
        save_fn=f"./logs/RNN_targets1_v{n}.pkl",
        target_offset=[0],
    )

for n in range(0, 2):

    set_seed(42+n)
    sweep = Sweep(
        network="RNN",
        full_plasticity_tasks=full_plasticity_tasks,
        sparse_plasticity_tasks=sparse_plasticity_tasks,
        save_fn=f"./logs/RNN_targets8_v{n}.pkl",
        target_offset=[0,1,2,3,4,5,6,7],
    )