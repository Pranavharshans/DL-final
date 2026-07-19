import os

from torch import nn


class MyModel(nn.Module):

    def __init__(self, num_classes):
        """
        Initialize the model.

        This method should define the layers or components of your neural network.

        Parameters
        ----------
        num_classes : int
            The number of output classes.

            This should match the number of countries in the dataset.

            Example:
            If there are 10 countries, num_classes should be 10.

        What to define here
        -------------------
        Define the layers or model components that will be used in `forward`.

        The model should take an image tensor as input and produce one score for each
        country class.

        Note
        ----
        The final layer of the model should output `num_classes` values for each image.

        For example, if num_classes is 10, the model should output 10 scores for each
        image.
        """
        super().__init__()

        self.model = None

    def forward(self, x):
        """
        Define how the input data passes through the model.

        This method should receive a batch of image tensors and return the model output.

        Parameters
        ----------
        x : torch.Tensor
            A batch of image tensors.

            Expected shape:

            [batch_size, 3, image_size, image_size]

        Returns
        -------
        torch.Tensor
            The model output.

            Expected shape:

            [batch_size, num_classes]

            Each row should contain the prediction scores for one image.

            Example:
            If batch_size is 4 and num_classes is 10, the output shape should be:

            [4, 10]

        What to do here
        ---------------
        Pass the input tensor through the layers or model components that were defined
        in `__init__`.

        The returned values should be raw prediction scores, not country names.
        """
        return None
