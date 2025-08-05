import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean

def evaluate(model, dataset, loss_fn):
    """
    Evaluates the model on a dataset.

    Args:
        model: The Keras model to evaluate.
        dataset: The tf.data.Dataset to evaluate on.
        loss_fn: The loss function.

    Returns:
        A tuple of (average_loss, average_accuracy).
    """
    accuracy_metric = CategoricalAccuracy()
    loss_metric = Mean()

    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)
        loss = loss_fn(y_batch, logits)

        accuracy_metric.update_state(y_batch, logits)
        loss_metric.update_state(loss)

    avg_loss = float(loss_metric.result())
    avg_accuracy = float(accuracy_metric.result())

    return avg_loss, avg_accuracy
