import tensorflow as tf


@tf.function
def train_step(
    model, optimizer, loss_fn, source: tf.Tensor, target: tf.Tensor
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        loss = loss_fn(target, model(source))
        grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


@tf.function
def test_step(model, loss_fn, source: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return loss_fn(target, model(source))

