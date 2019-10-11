import tensorflow as tf
from tensorflow import keras

from tf2_utils.lr_finder import LrGenerator, lr_finder, SmoothedLoss

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(64)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    # With explicitly built loss objects and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-2)

    # With compiled model
    # model.compile(optimizer='adam',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    #loss_object = model.loss_functions[0]
    #optimizer = model.optimizer

    lrs = LrGenerator(min_lr=1e-6, max_lr=10, n_steps=100)
    losses = SmoothedLoss(0.96)
    lr = lr_finder(model, optimizer, loss_object, dataset, lrs, losses)
    lr.plot_smoothed()
