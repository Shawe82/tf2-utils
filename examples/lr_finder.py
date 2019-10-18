import tensorflow as tf
from tensorflow import keras

from tf2_utils.lr_finder import (
    LrGenerator,
    SmoothedLoss,
    OneCycleLr,
    lr_finder,
    learner,
)

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(64)
    dataset_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
        64
    )

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # With explicitly built loss objects and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-2)

    # With compiled model
    # model.compile(optimizer='adam',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    # loss_object = model.loss_functions[0]
    # optimizer = model.optimizer

    lrs = LrGenerator(min_lr=1e-4, max_lr=10, n_steps=200)
    losses = SmoothedLoss(0.98)
    lr = lr_finder(model, optimizer, loss_object, dataset, lrs, losses)
    # model.reset_states()
    # lr.plot_smoothed()

    rater = OneCycleLr(
        n_epochs=5,
        n_batches=tf.data.experimental.cardinality(dataset).numpy(),
        max_lr=lr.lr_opt_m,
    )

    class MyLossObject:
        def __init__(self):
            self.train_loss = tf.keras.metrics.Mean(name="train_loss")
            self.test_loss = tf.keras.metrics.Mean(name="test_loss")

        def finish_epoch(self, epoch):
            print(
                f"Epoch {epoch} Loss: {self.train_loss.result()} Test: {self.test_loss.result()}"
            )
            self.train_loss.reset_states()
            self.test_loss.reset_states()

        def update_train(self, loss):
            self.train_loss(loss)

        def update_test(self, loss):
            self.test_loss(loss)

    res = learner(
        model, optimizer, loss_object, dataset, rater, MyLossObject(), dataset_test
    )
