# Predicting Car Purchase Amount Using Artificial Neural Networks (ANNs)

## 1. Introduction

The prediction of car purchase amounts based on customer profiles has become an important aspect of data-driven marketing. Businesses use this information to predict customer spending behavior, tailor marketing strategies, and optimize sales. This project uses Artificial Neural Networks (ANNs) to predict car purchase amounts using a dataset that includes demographic and financial attributes of customers.

### 1.1 Background

Machine learning models, especially neural networks, have demonstrated superior performance in tasks requiring pattern recognition, such as predicting continuous variables like car purchase amounts. By analyzing customer features like age, gender, annual salary, credit card debt, and net worth, we aim to predict how much a customer is likely to spend on a car.

### 1.2 Problem Statement

Accurately predicting car purchase amounts can be challenging due to the non-linear relationships between customer features and purchase behavior. Traditional methods may not capture these complex relationships, which is where ANNs come into play. This project focuses on building a predictive model that can effectively learn and generalize from the data.

### 1.3 Objectives

The main objectives of this project are:
- Preprocess the dataset to remove irrelevant columns and handle missing data.
- Scale features to ensure optimal model performance.
- Build and train an Artificial Neural Network to predict car purchase amounts.
- Evaluate the model's performance and optimize it through tuning.

## 2. Materials and Methods

### 2.1 Data Collection

The dataset used for this project, `Car_Purchasing_Data.csv`, contains various customer features such as:
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
- Car Purchase Amount (Target variable)

### 2.2 Data Processing

To build an effective neural network, data preprocessing steps were carried out, including:
- Dropping unnecessary columns (Customer Name, e-mail, and Country).
- Scaling the features to a range between 0 and 1 using `MinMaxScaler` to prevent large feature values from dominating smaller ones.

#### 2.2.1 Dropping Irrelevant Columns

Certain columns like customer names and email addresses are irrelevant for predicting purchase amounts. These were removed to reduce noise in the data.

#### 2.2.2 Feature Scaling

Using the `MinMaxScaler`, we scaled all features to fall within the range [0, 1] to enhance the performance of the neural network.

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

### 2.3 Splitting the Dataset

To train and evaluate the ANN model effectively, the dataset was split into training and testing sets. This allows us to assess the model's performance on unseen data. We typically use an 80-20 split for training and testing.

```python
from sklearn.model_selection import train_test_split

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
## 2.4 Building the ANN Model

An Artificial Neural Network (ANN) was built using the Keras library. The model architecture consisted of:

- An input layer that accepts the scaled features.
- One or more hidden layers with activation functions (e.g., ReLU) to learn complex patterns.
- An output layer that predicts the car purchase amount.

### 2.4.1 Model Architecture

The chosen architecture for the ANN is as follows:

- **Input Layer**: 5 neurons (one for each feature)
- **Hidden Layer 1**: 10 neurons, ReLU activation
- **Hidden Layer 2**: 5 neurons, ReLU activation
- **Output Layer**: 1 neuron (predicting the purchase amount)

```python
from keras.models import Sequential
from keras.layers import Dense

# Creating the ANN model
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))  # Hidden layer 1
model.add(Dense(5, activation='relu'))                 # Hidden layer 2
model.add(Dense(1, activation='linear'))               # Output layer

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

