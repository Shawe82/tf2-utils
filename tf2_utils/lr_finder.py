from dataclasses import dataclass

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
    _opt_idx = None
    lrs = []
    losses = []
    smoothed_losses = []
    avg_loss = 0

    def update(self, lr, loss):
        self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
        smooth_loss = self.avg_loss / (
            1 - self.smoothing ** (len(self.smoothed_losses) + 1)
        )
        self.best_loss = (
            loss if len(self.losses) == 0 or loss < self.best_loss else self.best_loss
        )

        self.lrs.append(lr)
        self.losses.append(loss)
        self.smoothed_losses.append(smooth_loss)

    @property
    def no_progress(self) -> bool:
        return self.smoothed_losses[-1] > 4 * self.best_loss

    def lr(self, step: int) -> float:
        return self.min_lr * (self.max_lr / self.min_lr) ** (step / (self.n_steps - 1))

    def reset(self):
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        self._opt_idx = None
        self.avg_loss = 0

    @property
    def opt_idx(self):
        if self._opt_idx is None:
            sls = np.array(self.smoothed_losses)
            leave_out = 3
            self._opt_idx = (
                np.argmin(sls[1 + leave_out :] - sls[leave_out:-1]) + 1 + leave_out
            )
        return self._opt_idx

    @property
    def lr_opt(self):
        return self.lrs[self.opt_idx]

    def plot(self):
        self._plt(False)

    def plot_smoothed(self):
        self._plt(True)

    def _plt(self, smoothed: bool = False, cut: int = 5):
        name = "Smoothed Loss" if smoothed else "Loss"
        loss = self.smoothed_losses if smoothed else self.losses
        plt.plot(self.lrs[cut:-cut], loss[cut:-cut])
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
    lr_o: Lr
) -> Lr:
    lr_o.reset()
    for step, (source, target) in enumerate(tqdm(dataset, total=lr_o.n_steps)):
        lr = lr_o.lr(step)  # Step 1 and Step 4
        loss = train_step(model, optimizer, loss_fn, source, target, lr).numpy()
        lr_o.update(lr, loss)

        if step - 1 == lr_o.n_steps or lr_o.no_progress:
            print(
                "Stopping because of loss"
                if lr_o.no_progress
                else "Stopping because number of max steps have been reached"
            )
            break

    return lr_o
