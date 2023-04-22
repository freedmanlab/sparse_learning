import os
import torch
import pickle
import numpy as np
from typing import Optional
from stimulus import TaskDataset
from networks import RNN, LSTM, LSTM_ctx_bottleneck, Classifers
from tasks import SoftmaxCrossEntropy, ActorCritic
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import TASKS


def main():

    learning_rate: float = 0.002
    max_train_epochs: int = 12
    max_test_epochs: int = 20
    network_hidden_dim: int = 512
    batch_size: int = 512
    num_workers: int = 16
    RL: bool = True

    # We will train on a percentage (train_pct) of the tasks in the first round fo training,
    # and the remainder during the second.
    # Alternatively, we can specify a holdout_task task for testing
    train_set_0, train_set_1 = split_train_set(train_pct=0.9, holdout_task=None)
    n = len(train_set_0) + len(train_set_1)
    print(f"Number of tasks in set 1: {len(train_set_0)}")
    print(f"Number of tasks in set 2: {len(train_set_1)}")

    train_data_v0 = TaskDataset(tasks=train_set_0, n_total_tasks=n, n_batches=200, RL=RL)
    val_data_v0 = TaskDataset(tasks=train_set_0, n_total_tasks=n, n_batches=10, RL=RL)
    train_data_v1 = TaskDataset(tasks=train_set_1, n_total_tasks=n, n_batches=200, RL=RL)
    val_data_v1 = TaskDataset(tasks=train_set_1, n_total_tasks=n, n_batches=10, RL=RL)


    stim_prop = train_data_v0.get_stim_properties()
    print("Stimulus and network properties")
    for k, v in stim_prop.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v}")

    # Define network
    network = LSTM_ctx_bottleneck(
        n_input=stim_prop["n_motion_tuned"] + stim_prop["n_fix_tuned"],
        n_context=stim_prop["n_rule_tuned"],
        n_output=stim_prop["n_motion_dirs"] + 1,
        hidden_dim=network_hidden_dim,
        # alpha=0.9,
    )

    for name, par in network.named_parameters():
        print(name, par.size())

    """First round of training"""

    context_params = [
        p for n, p in network.named_parameters() if "context" in n or "classifier" in n
    ]
    non_context_params = [
        p for n, p in network.named_parameters() if "context" not in n and "classifier" not in n
    ]


    optim = torch.optim.AdamW(
        [
            {"params": context_params, "lr": learning_rate, "weight_decay": 0.0},
            {"params": non_context_params, "lr": learning_rate, "weight_decay": 0.0},
        ],
        lr=learning_rate, weight_decay=0.00,
    )
    task = SoftmaxCrossEntropy(
        network=network,
        #clssifier_network=Classifers,
        optim_config=optim,
        n_logits=stim_prop["n_motion_dirs"] + 1,
    )

    train_loader = DataLoader(train_data_v0, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_data_v0, batch_size=batch_size, num_workers=num_workers)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_train_epochs,
        gradient_clip_val=1.0,
    )
    save_task_info(trainer, train_set_0 + train_set_1)

    trainer.fit(task, train_loader, val_loader)
    """
    PATH = "/home/masse/work/sparse_learning/lightning_logs/version_52/checkpoints/epoch=11-step=4800.ckpt"
    checkpoint = torch.load(PATH)
    print(checkpoint.keys())
    model_dict = {}
    for k, v in checkpoint['state_dict'].items():
        n = k.split(".")
        n = ".".join(n[1:])
        model_dict[n] = v
    network.load_state_dict(model_dict)
    """


    """Second round of training"""

    trainable_params = []
    for name, par in network.named_parameters():
        if ("context" not in name) and ("value" not in name):
            par.requires_grad = False
        else:
            trainable_params.append(par)

    optim = torch.optim.Adam(trainable_params, learning_rate)
    task = ActorCritic(
        network=network,
        optim_config=optim,
        n_logits=stim_prop["n_motion_dirs"] + 1,
    )

    train_loader = DataLoader(train_data_v1, batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(val_data_v1, batch_size=batch_size, num_workers=16)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_test_epochs,
        gradient_clip_val=1.0,
    )
    trainer.fit(task, train_loader, val_loader)

def save_task_info(trainer, tasks):

    if not os.path.exists(trainer.logger.log_dir):
        os.mkdir(trainer.logger.log_dir)
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


def get_available_tasks(n: Optional[int] = None):
    available_tasks = []
    n = n if n is not None else len(TASKS)
    tasks = np.random.choice(TASKS, size=n, replace=False)
    for task in tasks:
        for target_offset in [0, 1, 2, 3, 4, 5, 6, 7]:
            if "delayed_match" in task:
                for dir_offset in [0, 1, 2]:
                    available_tasks.append([task, dir_offset, target_offset])
            else:
                available_tasks.append([task, target_offset])
    return available_tasks

def split_train_set(train_pct: Optional[float], holdout_task: Optional[str] = None):

    # holdout_task can either be a task name or a key work (e.g. "category")
    assert np.logical_xor(train_pct is None, holdout_task is None), (
        "Either train_pct must be specified and holdout_task set to None, or vice-versa"
    )
    available_tasks = get_available_tasks()
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
    main()