from dataclasses import dataclass, field
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf2_utils.utils import train_step, test_step


@dataclass
class SmoothedLoss:
    smoothing: float = 1
    _losses: List[float] = field(init=False, default_factory=list)
    _smoothed_losses: List[float] = field(init=False, default_factory=list)
    _avg_loss: float = field(init=False, default=0)
    _best_loss: float = field(init=False, default=None)

    def update(self, loss):
        self._avg_loss = self.smoothing * self._avg_loss + (1 - self.smoothing) * loss
        smooth_loss = self._avg_loss / (
            1 - self.smoothing ** (len(self._smoothed_losses) + 1)
        )
        self._best_loss = (
            loss
            if len(self._losses) == 0 or loss < self._best_loss
            else self._best_loss
        )

        self._losses.append(loss)
        self._smoothed_losses.append(smooth_loss)

    @property
    def no_progress(self):
        return self._smoothed_losses[-1] > 4 * self._best_loss


@dataclass
class LrGenerator:
    min_lr: float
    max_lr: float
    n_steps: int
    _lrs: List[float] = field(init=False)

    def __call__(self):
        self._lrs = []
        for step in tqdm(range(self.n_steps)):
            learning_rate = self.min_lr * (self.max_lr / self.min_lr) ** (
                step / (self.n_steps - 1)
            )
            self._lrs.append(learning_rate)
            yield learning_rate

    @property
    def lrs(self) -> List[float]:
        return self._lrs


@dataclass
class Lr:
    lr: LrGenerator
    loss: SmoothedLoss
    _opt_idx_s: int = field(init=False, default=None)
    _opt_idx_m: int = field(init=False, default=None)

    @property
    def opt_idx_steep(self):
        cut = 3
        if self._opt_idx_s is None:
            sls = np.array(self.loss._smoothed_losses)
            self._opt_idx_s = np.argmin(sls[1 + cut :] - sls[cut:-1]) + 1 + cut
        return self._opt_idx_s

    @property
    def opt_idx_mag(self):
        if self._opt_idx_m is None:
            idx_ = np.argmin(self.loss._smoothed_losses)
            self._opt_idx_m = np.argmin(
                np.abs(np.array(self.lr.lrs) - self.lr.lrs[idx_] / 10)
            )
        return self._opt_idx_m

    @property
    def lr_opt_s(self):
        return self.lr.lrs[self.opt_idx_steep]

    @property
    def lr_opt_m(self):
        return self.lr.lrs[self.opt_idx_mag]

    def plot(self):
        self._plt(False)

    def plot_smoothed(self):
        self._plt(True)

    def _plt(self, smoothed: bool = False, cut: int = 5):
        name = "Smoothed Loss" if smoothed else "Loss"
        loss = self.loss._smoothed_losses if smoothed else self.loss._losses
        plt.plot(self.lr.lrs[cut:-cut], loss[cut:-cut])
        plt.axvline(x=self.lr_opt_s, color="r")
        plt.annotate(
            f"Optimal LR Steep {self.lr_opt_s:.4f}",
            xy=(self.lr_opt_s, loss[self.opt_idx_steep]),
        )
        plt.axvline(x=self.lr_opt_m, color="g")
        plt.annotate(
            f"Optimal LR Mag {self.lr_opt_m:.4f}",
            xy=(self.lr_opt_m, loss[self.opt_idx_mag]),
        )
        plt.xlabel("Learning Rate")
        plt.ylabel(name)
        plt.title(f"Learning Rate vs {name}")
        plt.xscale("log")
        plt.grid()
        plt.show()


@dataclass
class OneCycleLr:
    n_epochs: int
    n_batches: int
    max_lr: float
    lrs: List[float] = field(init=False)

    def __call__(self):
        self.lrs = []
        cycle_length = self.n_batches * self.n_epochs
        step_size = (cycle_length * 0.98) // 2
        min_lr = self.max_lr / 10
        rate = (self.max_lr - min_lr) / step_size

        lr = min_lr
        for step in tqdm(range(cycle_length)):
            self.lrs.append(lr)
            yield lr
            lr = lr + rate if step < step_size else lr - rate
            if step == (2 * step_size - 1):
                rate = (min_lr - self.max_lr / 100) / (cycle_length - step - 2)


def lr_finder(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    dataset,
    learn_rates: LrGenerator,
    losses: SmoothedLoss,
) -> Lr:
    for lr, (source, target) in zip(learn_rates(), dataset):
        tf.keras.backend.set_value(optimizer.lr, lr)
        loss = train_step(model, optimizer, loss_fn, source, target).numpy()
        losses.update(loss)
        if losses.no_progress:
            break

    return Lr(learn_rates, losses)


def learner(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    dataset,
    learn_rates: OneCycleLr,
    losses: Any,
    dataset_test=None,
):

    rates_iter = iter(learn_rates())
    for epoch in range(learn_rates.n_epochs):
        for src, trg in dataset:
            tf.keras.backend.set_value(optimizer.lr, next(rates_iter))
            loss = train_step(model, optimizer, loss_fn, src, trg).numpy()
            losses.update_train(loss)

        if dataset_test is not None:
            for s, t in dataset_test:
                test_loss = test_step(model, loss_fn, s, t)
                losses.update_test(test_loss)

        losses.finish_epoch(epoch)

    return losses
