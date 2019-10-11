from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Lr:
    _opt_idx = None
    lrs = []
    losses = []
    smoothed_losses = []

    def update(self, lr, loss, smooth_loss):
        self.lrs.append(lr)
        self.losses.append(loss)
        self.smoothed_losses.append(smooth_loss)

    @property
    def opt_idx(self):
        if self._opt_idx is None:
            sls = np.array(self.smoothed_losses)
            leave_out = 3
            self._opt_idx = np.argmin(sls[1 + leave_out:] - sls[leave_out:-1]) + 1 + leave_out
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

@dataclass
class LRFinder:
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    loss_fn: tf.keras.losses.Loss

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, source: tf.Tensor, target: tf.Tensor, lr: float) -> tf.Tensor:
        tf.keras.backend.set_value(self.optimizer.lr, lr)
        with tf.GradientTape() as tape:
            loss = self.loss_fn(target, self.model(source))
            grads = tape.gradient(loss, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def __call__(
        self,
        dataset,
        min_lr: float,
        max_lr: float,
        n_steps: int,
        smoothing: float = 1.0,
    ) -> Lr:
        lr_o = Lr()
        avg_loss = 0

        def exp_annealing(step: int) -> float:
            return min_lr * (max_lr / min_lr) ** (step / (n_steps - 1))

        for step, (source, target) in enumerate(tqdm(dataset, total=n_steps)):
            lr = exp_annealing(step)  # Step 1 and Step 4
            loss = self.train_step(source, target, lr).numpy()

            avg_loss = smoothing * avg_loss + (1 - smoothing) * loss
            smooth_loss = avg_loss / (1 - smoothing ** (step + 1))

            best_loss = loss if step == 0 or loss < best_loss else best_loss
            lr_o.update(lr, loss, smooth_loss)

            if step - 1 == n_steps or smooth_loss > 4 * best_loss:
                print(
                    "Stopping because number of max steps have been reached"
                    if step - 1 == n_steps
                    else "Stopping because of loss"
                )
                break

        return lr_o


    def __hash__(self):
        return hash(1)

    def __eq__(self, other):
        return True
