import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from typing import Any, List, Mapping
from torch.distributions.categorical import Categorical

class SoftmaxCrossEntropy(pl.LightningModule):

    def __init__(
            self,
            network: pl.LightningModule,
            #clssifier_network: pl.LightningModule,
            optim_config: Any,
            n_logits: int,
    ):

        pl.LightningModule.__init__(self)
        self.network = network
        #self.clssifier_network = clssifier_network
        self.optim_config = optim_config
        self.n_logits = n_logits
        self.classifiter_time = [30, 60]


    def configure_optimizers(self):
        return self.optim_config

    """
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"tasks": self.tasks})
    """

    def _train_network(self, logits: torch.Tensor, batch: Mapping[str, torch.Tensor]):

        loss = torch.nn.functional.cross_entropy(
            torch.permute(logits, (0, 2, 1)),
            torch.permute(batch["target"], (0, 2, 1)),
            reduction="none"
        )
        loss *= batch["mask"][..., 0]
        loss = loss.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _train_classifiers(self, logits: torch.Tensor, batch: Mapping[str, torch.Tensor]):

        T = self.classifiter_time[1] - self.classifiter_time[0]
        loss = 0.0
        logits = logits[:, self.classifiter_time[0]:self.classifiter_time[1]]
        logits = torch.reshape(logits, (-1, *logits.size()[2:]))
        train_list = ["target_offset", "stim_dir0", "stim_dir1"]
        train_list = ["target_offset"]
        for i, k in enumerate(train_list):

            target = torch.reshape(torch.tile(batch[k][:, None], (1, T)), (-1,)).to(torch.int64)
            idx = torch.where(target >= 0)[0]
            loss += nn.functional.cross_entropy(
                logits[idx, i, :],
                target,
                reduction="mean"
            )
        self.log("class_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):

        logits, _, class_logits = self.network(batch["bottom_up"], batch["context_input"])
        loss = self._train_network(logits, batch)
        classifier_loss = self._train_classifiers(class_logits, batch)

        return loss + classifier_loss

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        """ Both have shape (T, B, n_pol) """

        logits, _, class_logits = self.network(batch["bottom_up"], batch["context_input"])
        logits_idx = torch.argmax(logits, dim=-1, keepdim=False)
        target_idx = torch.argmax(batch["target"], dim=-1, keepdim=False)

        decision_mask = (target_idx < self.n_logits - 1).to(torch.float32)
        decision_mask *= batch["mask"][..., 0]
        bools = (logits_idx == target_idx).to(torch.float32)  # (T, B)
        bools_decision = bools * decision_mask  # (T,B)
        decision_acc = bools_decision.sum() / decision_mask.sum()

        fix_mask = (target_idx == self.n_logits - 1).to(torch.float32)
        fix_mask *= batch["mask"][..., 0]
        fix_decision = bools * fix_mask  # (T,B)
        fix_acc = fix_decision.sum() / fix_mask.sum()
        self.log("dec_acc", decision_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("fix_acc", fix_acc, on_step=False, on_epoch=True, prog_bar=True)

        # classifier accuracy
        class_list = ["target_offset", "stim_dir0", "stim_dir1"]
        class_list = ["target_offset"]
        logits_idx = torch.argmax(
            class_logits[:, self.classifiter_time[0]:self.classifiter_time[1], 0], dim=-1, keepdim=False
        )
        T = self.classifiter_time[1] - self.classifiter_time[0]
        target_idx = torch.tile(batch["target_offset"][:, None], (1, T))
        # print("XXX", logits_idx.size(), target_idx.size())
        bools = (logits_idx == target_idx).to(torch.float32)  # (T, B)
        class_acc = bools.mean()
        self.log("class_acc", class_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"dec_accuracy": decision_acc, "fixation_accuracy": fix_acc, "class_acc": class_acc}


class ActorCritic(SoftmaxCrossEntropy):

    def training_step_other(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        gamma: float = 0.9,
        val_loss_coeff: float = 0.001,
        entropy_coeff: float = 0.0001,
        n_mask_steps: int = 5,

    ):

        logits, value, _ = self.network(batch["bottom_up"], batch["context_input"])
        device = logits.device
        B, T, N = logits.size()

        probs = nn.functional.softmax(logits, dim=-1)
        m = Categorical(logits=logits)
        action_idx = m.sample().detach()
        action_one_hot = nn.functional.one_hot(action_idx, num_classes=N)
        log_probs = - m.log_prob(action_idx)
        rewards = (batch["reward"] * action_one_hot).sum(dim=-1).detach()
        continue_trial = torch.ones(B).to(device=device).detach()
        mask = []

        for t in range(T):
            mask.append(continue_trial.clone())
            if t >= n_mask_steps:
                continue_trial = continue_trial * (rewards[:, t] == 0).to(torch.float32)

        val_pred = torch.cat((value, torch.zeros((B, 1, 1)).to(device=device)), dim=1)
        mask = torch.stack(mask, dim=1)

        # terminal = (rewards != 0).to(torch.float32)
        val_future = (rewards + gamma * val_pred[:, 1:, 0]) * mask
        advantage = val_future - val_pred[:, :-1, 0]
        val_loss = (val_pred[:, :-1, 0] - val_future.detach()) ** 2
        pol_loss = log_probs * advantage.detach() * mask
        entropy = - (probs * torch.log(probs + 1e-9)).sum(dim=-1)

        loss = (pol_loss + val_loss_coeff * val_loss - entropy_coeff * entropy) * mask
        pos_rewards = (reward > 0).to(torch.float32)


        loss = loss.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mean_reward", (rewards * mask).sum() / B, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mean_pos_reward", (pos_rewards * mask).sum() / B, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        gamma: float = 0.9,
        val_loss_coeff: float = 0.0001,
        entropy_coeff: float = 0.000,
        n_mask_steps: int = 5,

    ):

        logits, value, _ = self.network(batch["bottom_up"], batch["context_input"])
        device = logits.device
        B, T, N = logits.size()

        probs = nn.functional.softmax(logits, dim=-1)
        m = Categorical(logits=logits)
        action_idx = m.sample().detach()
        action_one_hot = nn.functional.one_hot(action_idx, num_classes=N)
        log_probs = - m.log_prob(action_idx)
        rewards = (batch["reward"] * action_one_hot).sum(dim=-1).detach()
        continue_trial = torch.ones(B).to(device=device).detach()
        mask = []

        for t in range(T):
            mask.append(continue_trial.clone())
            if t >= n_mask_steps:
                continue_trial = continue_trial * (rewards[:, t] == 0).to(torch.float32)

        val_pred = torch.cat((value, torch.zeros((B, 1, 1)).to(device=device)), dim=1)
        mask = torch.stack(mask, dim=1)

        # terminal = (rewards != 0).to(torch.float32)
        val_future = (rewards + gamma * val_pred[:, 1:, 0]) * mask
        advantage = val_future - val_pred[:, :-1, 0]
        val_loss = (val_pred[:, :-1, 0] - val_future.detach()) ** 2
        pol_loss = log_probs * advantage.detach() * mask
        entropy = - (probs * torch.log(probs + 1e-9)).sum(dim=-1)

        loss = (pol_loss + val_loss_coeff * val_loss - entropy_coeff * entropy) * mask

        pos_rewards = (rewards > 0).to(torch.float32)

        loss = loss.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mean_reward", (rewards * mask).sum() / B, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mean_pos_reward", (pos_rewards * mask).sum() / B, on_step=False, on_epoch=True, prog_bar=True)

        return loss

