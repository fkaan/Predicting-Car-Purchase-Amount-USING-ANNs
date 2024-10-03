# Car Sales Prediction

The **Car Sales Prediction** project is designed to predict the purchase amount of cars based on various customer attributes. This project utilizes machine learning techniques to model the relationship between customer demographics and their expected car purchase behavior.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
The Car Sales Prediction project aims to predict the purchase amount of cars by leveraging customer attributes such as gender, age, annual salary, debt, and net worth.

## Features
- Predicts car purchase amounts using customer attributes such as gender, age, annual salary, debt, and net worth.
- Implements a neural network model using TensorFlow/Keras to train on customer data.
- Visualizes training progress, including loss curves for training and validation.

## Installation
To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/carsalesprediction.git
    ```



## Usage
Once the environment is set up, you can run the Jupyter Notebook to train the model and make predictions:

1. Open the notebook:

    ```bash
    jupyter notebook CarSalesPrediction.ipynb
    ```

2. Train the model on customer data:

   The model expects a dataset with customer attributes (gender, age, salary, debt, and net worth) for training.

3. Predict car purchase amounts:

   Example:

    ```python
    X_test = np.array([[0, 50, 50000, 10985, 600000]])
    y_predict = model.predict(X_test)
    print("Expected Purchase Amount: ", y_predict)
    ```

## Model Training
The model is a neural network built using TensorFlow/Keras. The training process includes:

1. Preparing the dataset of customer attributes.
2. Training the model with several epochs.
3. Visualizing the loss progress for both training and validation datasets.

Loss curves can be plotted as follows:

```python
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["val_loss"])
plt.title("Loss Progress")
plt.ylabel("Training and Validation Loss")
plt.xlabel("Epochs")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()
```

## Results
The model predicts the expected purchase amount based on customer data. An example prediction:

```python
Expected Purchase Amount: [[126894.98]]
```
## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

To install the dependencies, run:

```bash
pip install -r requirements.txt
```
