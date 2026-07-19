# Image Country Classification Challenge

You get street-level photos from different countries. Your model predicts the
country where each photo was taken.

---

## TL;DR

- **Input:** a street-level image.
- **Output:** one predicted country label.
- **Code:** We provide a simple code skeleton to help you get started. You
  may modify, replace, or restructure it however you like, as long as your final
  submission follows the competition rules.
- **Score:** classification **accuracy**. Higher is better.
- **Model:** a PyTorch image classifier.
- **You produce:** `country_predictions.csv`.

---

## The Data

The expected dataset folder is:

```text
geo_dataset/
├── train/                # labelled training images
├── train_labels.csv      # labels for train/
└── holdout_public/       # unlabelled images for prediction
```

`train_labels.csv` has one row per training image. It includes:

```text
filename,country,iso,lat,lng
```

For this project, the important target is:

```text
country
```

---

## Scoring

The scoring metric is **accuracy**:

```text
accuracy = correct country predictions / total images
```

Example:

```text
80 correct predictions out of 100 images
accuracy = 0.8000
```

The shared accuracy function is in `utils.py`:

```python
calculate_accuracy(predictions, labels)
```

It expects predicted class indexes and true class indexes, then returns one
number between `0` and `1`.

---

## Simple Codebase

```text
main.py              Train the model and create holdout predictions
model.py             Define the neural network model
data_preparation.py  Read labels, split data, and load images
train.py             Training, validation, and prediction functions
utils.py             Accuracy, plotting, device, and helper functions
requirements.txt     Python packages needed for the project
README.md            Project setup and usage instructions
```

This structure is only a starting point. You may change the files, model,
training loop, helper functions, or function descriptions if it improves your
solution.

---

## Environment Setup

Create a standard Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Train and Predict

Run:

```bash
python main.py
```

This will:

1. read `geo_dataset/train_labels.csv`
2. split the labelled data into training and validation sets
3. train the model
4. save the best model
5. save a training plot
6. predict countries for `geo_dataset/holdout_public/`

Expected output files:

```text
country_model.pt
country_predictions.csv
figures/training_history.png
```

---

## Prediction CSV Format

`country_predictions.csv` should have this format:

```csv
filename,pred_country
example_001.jpg,Germany
example_002.jpg,France
```

Each row should contain:

- the image filename
- the predicted country
