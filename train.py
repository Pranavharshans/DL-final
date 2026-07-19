import csv
import os
import torch
from data_preparation import load_image_tensor
from utils import calculate_accuracy

def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    Train the model for one epoch.

    One epoch means that the model sees every batch in the training dataloader once.
    During training, the model makes predictions, calculates the loss, updates its
    weights, and records the training performance.

    This function should return the average training loss and training accuracy for
    this epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model that will be trained.

    dataloader : torch.utils.data.DataLoader
        The DataLoader that provides batches of training data.

    loss_function : torch.nn.Module
        The loss function used to measure how wrong the model predictions are.

    optimizer : torch.optim.Optimizer
        The optimizer used to update the model weights.

    device : torch.device or str
        The device where the model and tensors should be placed.

    Returns
    -------
    tuple
        The function should return two values:

        average_loss : float
            The average loss over all training images in this epoch.

        accuracy : float
            The training accuracy over all training images in this epoch.

            Example:
            If the model correctly predicts 80 out of 100 images, the accuracy
            should be 0.8.

    Steps
    -----
    1. Put the model into training mode.
    2. Create variables to keep track of the total loss, number of images,
    predictions, and labels.
    3. Loop through each batch from the dataloader.
    4. Move the images and labels to the selected device.
    5. Clear the old gradients from the optimizer.
    6. Pass the images through the model to get the outputs.
    7. Use the loss function to calculate the loss.
    8. Backpropagate the loss.
    9. Use the optimizer to update the model weights.
    10. Store the loss, predictions, and true labels.
    11. After all batches are finished, calculate the average loss.
    12. Calculate the accuracy.
    13. Return the average loss and accuracy.

    Note
    ----
    The model, images, and labels should be on the same device.

    For example, if the model is on the GPU, the images and labels should also be
    moved to the GPU before they are used by the model.

    In training mode, the model weights are updated.

    This is different from validation or testing, where the model should only make
    predictions and should not update its weights.
    """
    average_loss = None
    accuracy = None
    return average_loss, accuracy

def evaluate(model, dataloader, loss_function, device):
    """
    Evaluate the model on a validation dataset.

    Evaluation means that the model makes predictions, calculates the loss, and
    records the performance, but does not update its weights.

    This function should return the average evaluation loss and accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model that will be evaluated.

    dataloader : torch.utils.data.DataLoader
        The DataLoader that provides batches of validation or test data.

    loss_function : torch.nn.Module
        The loss function used to measure how wrong the model predictions are.

    device : torch.device or str
        The device where the model and tensors should be placed.

    Returns
    -------
    tuple
        The function should return two values:

        average_loss : float
            The average loss over all images in the dataset.

        accuracy : float
            The accuracy over all images in the dataset.

            Example:
            If the model correctly predicts 75 out of 100 images, the accuracy
            should be 0.75.

    Steps
    -----
    1. Put the model into evaluation mode.
    2. Create variables to keep track of the total loss, number of images,
    predictions, and labels.
    3. Turn off gradient calculation using `with torch.no_grad():`.
    4. Loop through each batch from the dataloader.
    5. Move the images and labels to the selected device.
    6. Pass the images through the model to get the outputs.
    7. Use the loss function to calculate the loss.
    8. Store the loss, predictions, and true labels.
    9. After all batches are finished, calculate the average loss.
    10. Calculate the accuracy.
    11. Return the average loss and accuracy.

    Note
    ----
    The model, images, and labels should be on the same device.

    For example, if the model is on the GPU, the images and labels should also be
    moved to the GPU before they are used by the model.

    During evaluation, the model weights should not be updated.

    Use `with torch.no_grad():` around the evaluation loop to turn off gradient
    calculation.

    Example:

        with torch.no_grad():
            for images, labels in dataloader:
                ...

    This saves memory and makes evaluation faster.

    This is different from training, where the model uses backpropagation and the
    optimizer to update its weights.
    """

    average_loss = None
    accuracy = None
    return average_loss, accuracy


def predict_holdout(model, holdout_dir, countries, image_size, output_csv, device):
    """
    Predict country labels for all holdout images and save the results to a CSV file.

    This function should use a trained model to predict the country label for each
    image in the holdout folder. The predictions should then be saved into a CSV
    file.

    The holdout images are images without labels. The goal is to use the model to
    predict their country labels.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model used to make predictions.

    holdout_dir : str
        The folder that contains the holdout images.

    countries : list of str
        A list of country names.

        The model will output a numeric prediction index. This index should be used
        to select the predicted country name from this list.

        Example:

        If the model predicts index 1 and countries is:

        ["France", "Germany", "Spain"]

        then the predicted country should be:

        "Germany"

    image_size : int
        The size used to resize each image before prediction.

        Example:
        If image_size is 224, each image should be resized to:

        224 x 224

    output_csv : str
        The path of the CSV file where the predictions should be saved.

    device : torch.device or str
        The device where the model and image tensors should be placed.

    Returns
    -------
    str
        The path of the saved CSV file.

        Example:

        "country_predictions.csv"

    Steps
    -----
    1. Get all image filenames from the holdout folder.
    2. Put the model into evaluation mode.
    3. Create an empty list to store the prediction results.
    4. Use `with torch.no_grad():` to turn off gradient calculation.
    5. For each holdout image, load the image as a tensor.
    6. Add a batch dimension to the image tensor.
    7. Move the image tensor to the selected device.
    8. Pass the image through the model to get the output.
    9. Get the predicted class index from the model output.
    10. Use the predicted class index to find the predicted country name.
    11. Store the filename and predicted country.
    12. Save all predictions to the output CSV file.
    13. Return the path of the saved CSV file.

    Note
    ----
    The model and image tensors should be on the same device.

    Use `with torch.no_grad():` during prediction because the model is only making
    predictions and should not update its weights.

    Each image tensor should include a batch dimension before it is passed into the
    model.

    Example:
    If one image tensor has shape:

        [3, image_size, image_size]

    then after adding the batch dimension, it should have shape:

        [1, 3, image_size, image_size]

    The output CSV file should contain two columns:

        filename,pred_country

    Example output CSV content:

        filename,pred_country
        image_001.jpg,Germany
        image_002.jpg,France
    """

    return output_csv
