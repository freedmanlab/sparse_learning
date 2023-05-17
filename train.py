import os
import torch
import pickle
import numpy as np
from typing import List, Literal, Union
from stimulus import TaskDataset
from networks import RNN, LSTM
from tasks import SoftmaxCrossEntropy, ActorCritic
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import TASKS, train_params, model_params
from pytorch_lightning.loggers import CSVLogger


class Train:

    def __init__(self, network: Union[Literal["LSTM", "RNN", torch.nn.Module]] = "LSTM"):

        # create a dummy loader in order to extract the stimulus properties
        dummy_loader = TaskDataset(tasks=None, n_batches=0, RL=False)
        self.logger = CSVLogger("logs", name="train_sweep", version=0)

        # if network is "LSTM" or "RNN", then initialize the network
        # else, we'll use the network (nn.Module) provided
        if network in ["LSTM", "RNN"]:
            add_model_params(dummy_loader)
            self.network = LSTM(**model_params) if network == "LSTM" else RNN(**model_params)

        self.define_optimizer()

    def define_optimizer(self, sparse_plasticity: bool = False):
        """Create the network optimizer. The learning rate of network parameters not involved in top-down
        context will be set to 0 if sparse_plasticity = True"""

        non_top_down_lr = 0.0 if sparse_plasticity else train_params["learning_rate"]
        context_params = [p for n, p in self.network.named_parameters() if "context" in n or "classifier" in n]
        non_context_params = [p for n, p in self.network.named_parameters() if "context" not in n and "classifier" not in n]
        optim = torch.optim.AdamW(
            [
                {"params": context_params, "lr": train_params["learning_rate"], "weight_decay": 0.0},
                {
                    "params": non_context_params,
                    "lr": non_top_down_lr,
                    "weight_decay": train_params["weight_decay"],
                },
            ],
            lr=train_params["learning_rate"], weight_decay=0.0,
        )

        # ensure non-top down parameters are not trainable if sparse is True
        if sparse_plasticity:
            for n, p in self.network.named_parameters():
                if "context" not in n and "classifier" not in n:
                    p.requires_grad = False


        self.task = SoftmaxCrossEntropy(
            network=self.network,
            optim_config=optim,
            n_logits=model_params["n_output"],
        )

    def create_task_loaders(self, task_set: List, n_batches_per_epoch: int = 50):
        """Create the trianing and validation stimulus data loader"""

        self.task_set = task_set
        train = TaskDataset(tasks=task_set, n_batches=n_batches_per_epoch, RL=train_params["RL"])
        val = TaskDataset(tasks=task_set, n_batches=10, RL=train_params["RL"])
        self.train_loader = DataLoader(
            train,
            batch_size=train_params["batch_size"],
            num_workers=train_params["num_workers"],
        )
        self.val_loader = DataLoader(
            val,
            batch_size=train_params["batch_size"],
            num_workers=train_params["num_workers"],
        )

    def train_model(self, n_epochs):
        """Train the model and return the saved model and results"""""
        # create PyTorch Lightning Trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=n_epochs,
            gradient_clip_val=1.0,
            precision=32,
            devices=1,
            logger=self.logger,
        )
        trainer.fit(self.task, self.train_loader, self.val_loader)
        save_task_info(trainer, self.task_set)

        return self.task.network


def add_model_params(task_loader, verbose=False):
    """Add model params needed from stimulus class"""
    stim_prop = task_loader.get_stim_properties()
    if verbose:
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

