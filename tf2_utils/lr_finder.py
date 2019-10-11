from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm


@dataclass
class Lr:
    min_lr: float
    max_lr: float
    n_steps: int
    smoothing: float = 1
    _opt_idx: int = field(init=False)
    _lrs: List[float] = field(init=False)
    _losses: List[float] = field(init=False)
    _smoothed_losses: List[float] = field(init=False)
    _avg_loss: float = field(init=False)
    _best_loss: float = field(init=False)

    def __call__(self):
        self._lrs, self._losses, self._smoothed_losses = [], [], []
        self._avg_loss = 0
        self._opt_idx, self._best_loss = None, None

        for step in tqdm(range(self.n_steps)):
            lr = self.min_lr * (self.max_lr / self.min_lr) ** (
                step / (self.n_steps - 1)
            )
            self._lrs.append(lr)
            yield lr

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

    @property
    def opt_idx(self):
        cut = 3
        if self._opt_idx is None:
            sls = np.array(self._smoothed_losses)
            self._opt_idx = np.argmin(sls[1 + cut :] - sls[cut:-1]) + 1 + cut
        return self._opt_idx

    @property
    def lr_opt(self):
        return self._lrs[self.opt_idx]

    def plot(self):
        self._plt(False)

    def plot_smoothed(self):
        self._plt(True)

    def _plt(self, smoothed: bool = False, cut: int = 5):
        name = "Smoothed Loss" if smoothed else "Loss"
        loss = self._smoothed_losses if smoothed else self._losses
        plt.plot(self._lrs[cut:-cut], loss[cut:-cut])
        plt.axvline(x=self.lr_opt, color="r")
        plt.annotate(
            f"Optimal LR {self.lr_opt:.4f}", xy=(self.lr_opt, loss[self.opt_idx])
        )
        plt.xlabel("Learning Rate")
        plt.ylabel(name)
        plt.title(f"Learning Rate vs {name}")
        plt.xscale("log")
        plt.grid()
        plt.show()


@tf.function
def train_step(
    model, optimizer, loss_fn, source: tf.Tensor, target: tf.Tensor, lr: float
) -> tf.Tensor:
    tf.keras.backend.set_value(optimizer.lr, lr)
    with tf.GradientTape() as tape:
        loss = loss_fn(target, model(source))
        grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def lr_finder(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    dataset,
    learn_rates: Lr,
) -> Lr:
    for lr, (source, target) in zip(learn_rates(), dataset):
        loss = train_step(model, optimizer, loss_fn, source, target, lr).numpy()
        learn_rates.update(loss)
        if learn_rates.no_progress:
            break
    return learn_rates
