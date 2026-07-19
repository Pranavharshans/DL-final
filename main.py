import os
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation import (
    CountryImageDataset,
    prepare_training_data,
)
from model import MyModel
from train import train_one_epoch, evaluate, predict_holdout
from utils import get_device, count_parameters, plot_training_history

def main():
    # Files created after training finishes.
    FINAL_MODEL_PATH = "country_model.pt"
    FINAL_PREDICTIONS_PATH = "country_predictions.csv"

    # Main settings for the experiment.
    Config = {
        "data_dir": "geo_dataset",
        "image_size": 224,
        "validation_fraction": 0.1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 30,
        "weight_decay": 1e-4,
        "device": get_device()
    }

    data_dir = Config["data_dir"]
    labels_csv = os.path.join(data_dir, "train_labels.csv")
    train_image_dir = os.path.join(data_dir, "train")
    holdout_dir = os.path.join(data_dir, "holdout_public")

    # Read the CSV labels, create country indexes, and split train/validation rows.
    countries, country_to_index, train_rows, validation_rows = prepare_training_data(
        labels_csv,
        validation_fraction=Config["validation_fraction"],
    )

    # Dataset objects know how to load one image and convert its country to a number.
    train_dataset = CountryImageDataset(
        train_rows,
        train_image_dir,
        country_to_index,
        image_size=Config["image_size"],
    )
    validation_dataset = CountryImageDataset(
        validation_rows,
        train_image_dir,
        country_to_index,
        image_size=Config["image_size"],
    )

    # DataLoaders create batches of images for training and validation.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=Config["batch_size"],
        shuffle=False,
    )

    # Create the model and choose the training tools.
    model = MyModel(num_classes=len(countries)).to(Config["device"])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["learning_rate"], weight_decay=Config["weight_decay"])

    print(f"Countries: {countries}")
    print(f"Model parameters: {count_parameters(model):,}")
    epoch_bar = tqdm(range(Config["epochs"]), desc="Training", unit="epoch")

    # Keep a copy of the model with the best validation accuracy.
    best_model = None
    best_val_acc = 0.0

    # Store numbers from each epoch so we can plot them after training.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in epoch_bar:
        # One full pass over the training data.
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            Config["device"],
        )

        # Check performance on validation data without updating model weights.
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            Config["device"],
        )

        # Save the best model seen so far, not just the model from the final epoch.
        if validation_accuracy > best_val_acc:
            best_val_acc = validation_accuracy
            best_model = copy.deepcopy(model)

        # Save these values for the final training plot.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(validation_accuracy)

        # Show the newest numbers in the tqdm progress bar.
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_accuracy:.4f}",
            "val_loss": f"{validation_loss:.4f}",
            "val_acc": f"{validation_accuracy:.4f}"
        })

    # Plot loss and accuracy after all epochs are finished.
    figure_path = plot_training_history(history)
    print(f"Saved training plot to {figure_path}")

    if best_model is not None:
        # Save the model, config, and country names together.
        torch.save(
            {
                "model_state": best_model,
                "config": Config,
                "countries": countries,
            },
            FINAL_MODEL_PATH,
        )
        print(f"Saved model to {FINAL_MODEL_PATH}")

        # Use the best model to create predictions for the holdout images.
        predict_holdout(
            best_model,
            holdout_dir,
            countries,
            Config["image_size"],
            FINAL_PREDICTIONS_PATH,
            Config["device"],
        )
        print(f"Saved country predictions to {FINAL_PREDICTIONS_PATH}")

if __name__ == "__main__":
    main()
