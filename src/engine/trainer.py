import tensorflow as tf

def train_one_epoch(model, dataset, loss_fn, optimizer):
    """
    Trains the model for one epoch.

    Args:
        model: The Keras model to train.
        dataset: The tf.data.Dataset to train on.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
