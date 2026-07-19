import os
import torch
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())

def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy as a value between 0 and 1.

    Arguments
    ---------
    predictions : torch.Tensor
        The predicted class indexes.
        Shape should be [number_of_examples].
        Example: torch.tensor([1, 0, 1])

    labels : torch.Tensor
        The correct class indexes.
        Shape should be [number_of_examples].
        Example: torch.tensor([1, 0, 0])

    Returns
    -------
    accuracy : float
        The percentage of correct predictions in the batch.
        Example: 0.75 means 75% accuracy.
    """
    correct_predictions = (predictions == labels).sum().item()
    total_predictions = labels.size(0)
    return correct_predictions / total_predictions

def plot_training_history(history, output_path="figures/training_history.png"):
    """
    Plot training and validation loss/accuracy.

    Arguments
    ---------
    history : dict
        Training numbers collected after each epoch.
        Example:
        {
            "train_loss": [1.2, 0.9],
            "val_loss": [1.3, 1.0],
            "train_acc": [0.55, 0.70],
            "val_acc": [0.50, 0.65],
        }

    output_path : str
        Where the figure should be saved.
        Example: "figures/training_history.png"

    Returns
    -------
    output_path : str
        The path where the figure was saved.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")
