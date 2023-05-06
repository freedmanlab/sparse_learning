import os
import torch
import pickle
import numpy as np
from typing import List, Optional
from stimulus import TaskDataset
from networks import RNN, RNN_stdp, LSTM, LSTM_ctx_bottleneck, Classifers
from tasks import SoftmaxCrossEntropy, ActorCritic
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import TASKS, train_params, model_params


def main(tasks0: List, tasks1: List):

    # initialize random seed
    np.random.seed(train_params["seed"])

    val_task = ActorCritic if train_params["RL"] else SoftmaxCrossEntropy

    # We will train on a percentage (train_pct) of the tasks in the first round fo training,
    # and the remainder during the second.
    # Alternatively, we can specify a holdout_task task for testing
    task_set = create_task_set(tasks0=tasks0, tasks1=tasks1)

    # add model params from stimulus class
    add_model_params(task_set)

    # define network
    network = RNN(**model_params)

    """First round of training"""

    for n, p in network.named_parameters():
        print('name', n)

    context_params = [p for n, p in network.named_parameters() if "context" in n or "classifier" in n]
    non_context_params = [p for n, p in network.named_parameters() if "context" not in n and "classifier" not in n]

    optim = torch.optim.AdamW(
        [
            {"params": context_params, "lr": train_params["learning_rate"], "weight_decay": 0.0},
            {"params": non_context_params, "lr": train_params["learning_rate"], "weight_decay": 0.0},
        ],
        lr=train_params["learning_rate"], weight_decay=0.00,
    )
    task = SoftmaxCrossEntropy(
        network=network,
        #clssifier_network=Classifers,
        optim_config=optim,
        n_logits=model_params["n_output"],
    )

    train_loader = DataLoader(task_set["train0"], batch_size=train_params["batch_size"], num_workers=train_params["num_workers"])
    val_loader = DataLoader(task_set["val0"], batch_size=train_params["batch_size"], num_workers=train_params["num_workers"])
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=train_params["num_train_epochs"],
        gradient_clip_val=1.0,
        precision=32,
        devices=1,
    )
    save_task_info(trainer, task_set["tasks0"] + task_set["tasks1"])

    trainer.fit(task, train_loader, val_loader)

    """Second round of training"""

    trainable_params = []
    for name, par in network.named_parameters():
        if ("context" not in name) and ("value" not in name):
            par.requires_grad = False
        else:
            trainable_params.append(par)

    optim = torch.optim.Adam(trainable_params, train_params["learning_rate"])
    task = val_task(
        network=network,
        optim_config=optim,
        n_logits=stim_prop["n_motion_dirs"] + 1,
    )

    train_loader = DataLoader(task_set["train1"], batch_size=train_params["batch_size"], num_workers=train_params["num_workers"])
    val_loader = DataLoader(task_set["val1"], batch_size=train_params["batch_size"], num_workers=train_params["num_workers"])
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=train_params["num_test_epochs"],
        gradient_clip_val=1.0,
        precision=32,
        devices=1,
    )
    trainer.fit(task, train_loader, val_loader)


def add_model_params(task_set):
    """Add model params needed from stimulus class"""
    stim_prop = task_set["train0"].get_stim_properties()
    print("Stimulus and network properties")
    for k, v in stim_prop.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v}")

    # Add network params
    for k in ["n_stimulus", "n_context", "n_output"]:
        model_params[k] = stim_prop[k]

def save_task_info(trainer, tasks):

    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    tasks_fn = trainer.logger.log_dir + "/tasks.pkl"
    print(f"Saving task info to {tasks_fn}")
    tasks_info = {"index": [], "name": [], "target_offset": [], "stim_offset": []}

    for n, t in enumerate(tasks):
        tasks_info["index"].append(n)
        tasks_info["name"].append(t[0])
        tasks_info["target_offset"].append(t[1])
        if len(t) == 3:
            tasks_info["stim_offset"].append(t[2])
        else:
            tasks_info["stim_offset"].append(-1)
    pickle.dump(tasks_info, open(tasks_fn, "wb"))



def generate_task_variants(task):
    task_variants = []
    for target_offset in [0, 1, 2, 3, 4, 5, 6, 7]:
        if "delayed_match" in task:
            for dir_offset in [0, 1, 2]:
                task_variants.append([task, dir_offset, target_offset])
        else:
            task_variants.append([task, target_offset])
    return task_variants


def create_task_set(tasks0: List, tasks1: List, train_pct: float = 0.8):

    task_set = {"tasks0": [], "tasks1": []}
    print("Task set 0")
    for t in tasks0:
        print(t)
    print("Task set 1")
    for t in tasks1:
        print(t)

    for task in tasks0 + tasks1:
        if task in tasks0 and task not in tasks1:
            task_set["tasks0"] += generate_task_variants(task)
        elif task not in tasks0 and task in tasks1:
            task_set["tasks1"] += generate_task_variants(task)
        else:
            t = generate_task_variants(task)
            n = int(train_pct * len(t))
            np.random.shuffle(t)
            task_set["tasks0"] += t[:n]
            task_set["tasks1"] += t[n:]

    n = len(task_set["tasks0"]) + len(task_set["tasks1"])
    print(f"Number of tasks in set 1: {len(task_set['tasks0'])}")
    print(f"Number of tasks in set 2: {len(task_set['tasks1'])}")

    task_set["train0"] = TaskDataset(tasks=task_set['tasks0'], n_total_tasks=n, n_batches=200, RL=train_params["RL"])
    task_set["val0"] = TaskDataset(tasks=task_set['tasks0'], n_total_tasks=n, n_batches=10, RL=train_params["RL"])
    task_set["train1"] = TaskDataset(tasks=task_set['tasks1'], n_total_tasks=n, n_batches=200, RL=train_params["RL"])
    task_set["val1"] = TaskDataset(tasks=task_set['tasks1'], n_total_tasks=n, n_batches=10, RL=train_params["RL"])

    return task_set

def split_train_set(tasks0: List, tasks1: List, train_pct: float=0.9):

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


if __name__ == "__main__":
    tasks=[2]#,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,165]
    #tasks=[3,4,6,7,8,9,11,12,13,14,15,25,35,45,55,65,75,85,95]
    seeds = 2
    #tasks=[2,3]
    #seeds = 2
    for j in range(seeds):
        print('SEED',j)
        for i in range(len(tasks)):
            print('TASKS',tasks[i])
            task=tasks[i]
            main(tasks=task,seed=j,tasks0=TASKS, tasks1=[])
