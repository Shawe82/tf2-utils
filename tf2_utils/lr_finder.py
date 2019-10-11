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
    _smoothed_losses: float = field(init=False)
    _avg_loss: float = field(init=False)

    def update(self, lr, loss):
        self._avg_loss = self.smoothing * self._avg_loss + (1 - self.smoothing) * loss
        smooth_loss = self._avg_loss / (
            1 - self.smoothing ** (len(self._smoothed_losses) + 1)
        )
        self.best_loss = (
            loss if len(self._losses) == 0 or loss < self.best_loss else self.best_loss
        )

        self._lrs.append(lr)
        self._losses.append(loss)
        self._smoothed_losses.append(smooth_loss)

    @property
    def no_progress(self) -> bool:
        return self._smoothed_losses[-1] > 4 * self.best_loss

    def __call__(self):
        for step in range(self.n_steps):
            yield self.min_lr * (self.max_lr / self.min_lr) ** (
                step / (self.n_steps - 1)
            )
            if self.no_progress:
                break

    def reset(self):
        self._lrs = []
        self._losses = []
        self._smoothed_losses = []
        self._opt_idx = None
        self._avg_loss = 0

    @property
    def opt_idx(self):
        leave_out = 3
        if self._opt_idx is None:
            sls = np.array(self._smoothed_losses)
            self._opt_idx = (
                np.argmin(sls[1 + leave_out :] - sls[leave_out:-1]) + 1 + leave_out
            )
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
    learn_rates.reset()
    for lr, (source, target) in tqdm(
        zip(learn_rates(), dataset), total=learn_rates.n_steps
    ):
        loss = train_step(model, optimizer, loss_fn, source, target, lr).numpy()
        learn_rates.update(lr, loss)

    return learn_rates
