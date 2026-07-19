import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def prepare_training_data(csv_path, validation_fraction=0.2, seed=42):
    """
    Read label data, create class indexes, and split the dataset into training and validation sets.

    This function should read a label file such as `train_labels.csv`. The file
    contains image filenames and their country labels. After reading the file, the
    function should create a list of all countries, build a mapping from country
    names to numeric class indexes, and split the data into training and validation
    rows.

    The label file is expected to contain at least two columns:

        filename,country

    Parameters
    ----------
    csv_path : str
        The path to the label file.

    validation_fraction : float
        The fraction of the data that should be used for validation.

        Example:
        If validation_fraction is 0.2, about 20% of the rows should be used
        for validation, and about 80% should be used for training.

    seed : int
        A random seed used to make the split reproducible.
        Using the same seed should create the same split each time.

    Returns
    -------
    tuple
        The function should return four values:

        countries : list of str
            A sorted list of all country names.

            Example:
            ["Germany", "Japan"]

        country_to_index : dict
            A dictionary that maps each country name to a numeric class index.

            Example:
            {
                "Germany": 0,
                "Japan": 1,
            }

        train_rows : list of dict
            The rows used for training.
            Each row should have the same structure as the rows read from the file.

            Example:
            [
                {"filename": "image_001.jpg", "country": "Japan"},
                {"filename": "image_002.jpg", "country": "Germany"},
            ]

        validation_rows : list of dict
            The rows used for validation.
            Each row should have the same structure as the rows read from the file.

            Example:
            [
                {"filename": "image_003.jpg", "country": "Japan"},
            ]

    Steps
    -----
    1. Read the label file.
    2. Store the label data as a list of dictionaries.

    Each dictionary should represent one image and its country label.

    Example:

    [
        {"filename": "image_001.jpg", "country": "Japan"},
        {"filename": "image_002.jpg", "country": "Germany"},
        {"filename": "image_003.jpg", "country": "Japan"},
    ]

    3. Create a sorted list of all unique country names.
    4. Create a dictionary that maps each country name to a numeric class index.
    5. Split the rows into training rows and validation rows.
    6. Return the countries, country-to-index mapping, training rows, and validation rows.

    Note
    ----
    When splitting the data, be careful about the distribution of each country.

    For example, if one country appears only in the training set and not in the
    validation set, the validation result may not properly show how well the model
    performs on that country.

    A better split should try to keep a similar country distribution in both the
    training set and the validation set.
    """
    countries = None
    country_to_index = None
    train_rows = None
    validation_rows = None

    return countries, country_to_index, train_rows, validation_rows


def load_image_tensor(image_path, image_size):
    """
    Load one image and convert it into a PyTorch tensor.

    This function should read an image file, resize it to the target image size,
    and convert it into a tensor that can be used as input for a neural network.

    The output tensor should have 3 color channels.

    Parameters
    ----------
    image_path : str
        The path to the image file.

    image_size : int
        The target width and height of the image.

        Example:
        If image_size is 224, the image should be resized to:

        224 x 224

    Returns
    -------
    torch.Tensor
        A tensor representing the image.

        The tensor should have shape:

        [3, image_size, image_size]

        This means:
        - 3 color channels
        - image_size pixels in height
        - image_size pixels in width

        The pixel values should be scaled to the range 0 to 1.

        Example:
        If image_size is 224, the returned tensor should have shape:

        [3, 224, 224]

    Note
    ----
    Different image files may have different sizes or color formats.

    Before returning the tensor, make sure:
    - the image has 3 channels
    - the image is resized to the target size
    - the tensor shape is [3, image_size, image_size]
    - the pixel values are between 0 and 1
    """
    image_tensor = None
    return image_tensor


class CountryImageDataset(Dataset):
    """Dataset for training and validating country classification."""

    def __init__(self, rows, image_dir, country_to_index, image_size=224):
        self.rows = rows
        self.image_dir = image_dir
        self.country_to_index = country_to_index
        self.image_size = image_size

    def __len__(self):
        """
        Return the number of samples in the dataset.

        This method tells PyTorch how many items are available in the dataset.

        Returns
        -------
        int
            The total number of rows in the dataset.

            Example:
            If the dataset contains 100 image-label rows, this method should return:

            100
        """
        return None

    def __getitem__(self, index):
        """
        Return one image and its label from the dataset.

        This method is called by PyTorch when it needs one training or validation sample.
        The input `index` tells the dataset which row to use.

        For the selected row, this method should:
        1. Get the filename and country label from the row.
        2. Build the full image path using the image directory and filename.
        3. Load the image as a tensor.
        4. Convert the country label into its numeric class index.
        5. Return both the image tensor and the label index.

        Parameters
        ----------
        index : int
            The position of the sample in the dataset.

        Returns
        -------
        tuple
            A tuple containing two values:

            image : torch.Tensor
                The image tensor.

                Expected shape:

                [3, image_size, image_size]

            label : int
                The numeric class index of the country label.

                Example:
                If the country is "Japan" and country_to_index is:

                {
                    "Germany": 0,
                    "Japan": 1,
                }

                then the label should be:

                1
        """

        image = None
        label = None

        return image, label
