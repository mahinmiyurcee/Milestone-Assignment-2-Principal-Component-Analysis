# Milestone-Assignment-2: PCA and Logistic Regression on Breast Cancer Dataset

This project demonstrates the application of **Principal Component Analysis (PCA)** for dimensionality reduction and **Logistic Regression** for binary classification using the **Breast Cancer Wisconsin dataset** from scikit-learn. The dataset is reduced to two principal components using PCA, and a logistic regression model is trained to predict whether the tumor is benign or malignant.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset from 30 features to 2 components for visualization. After applying PCA, a logistic regression model is trained to classify the tumors as benign or malignant, based on these two components.

## Project Structure

The project consists of the following files:

- **pca_logistic_regression.py**: Python script that contains the full implementation.
- **README.md**: Project documentation (this file).
- **requirements.txt**: List of required Python libraries.

## Installation

To run this project, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pca-breast-cancer.git
   cd pca-breast-cancer
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the dependencies, you can run the Python script:

```bash
python pca_logistic_regression.py
```

### Script Overview:

- Loads the Breast Cancer dataset.
- Scales the features using `StandardScaler`.
- Applies PCA to reduce the dataset to 2 components.
- Visualizes the two principal components using a scatter plot.
- Splits the data into training and test sets.
- Trains a Logistic Regression model using the two PCA components.
- Evaluates the model's accuracy on the test set.
