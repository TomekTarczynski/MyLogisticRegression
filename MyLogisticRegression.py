import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, verbose=False):
        """
        Initializes the MyLogisticRegression class with learning rate, number of iterations,
        and an optional verbose parameter for debugging information.

        Parameters:
        - lr (float): Learning rate for gradient descent.
        - n_iters (int): Number of iterations to run gradient descent.
        - verbose (bool): If True, print debugging information during training.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.bias = None

        if self.verbose:
            print("Initial Parameters:")
            print(f"Learning Rate: {self.lr}")
            print(f"Number of Iterations: {self.n_iters}")
            print(f"Verbose Mode: {self.verbose}")
            print(f"Initial Weights: {self.weights}")
            print(f"Initial Bias: {self.bias}")
            print("="*50)

    def fit(self, X, y):
        """
        Fits the logistic regression model to the data using gradient descent.

        Parameters:
        - X (array-like): Input features with shape (n_samples, n_features).
        - y (array-like): Target labels with shape (n_samples,).
        """
        n_obs, n_vars = X.shape
        self.n_vars = n_vars
        self.n_obs = n_obs

        self.weights = np.zeros(n_vars)
        self.bias = 0

        if self.verbose:
            print("Start of fit:")
            print(f"Learning Rate: {self.lr}")
            print(f"Number of Iterations: {self.n_iters}")
            print(f"Verbose Mode: {self.verbose}")
            print(f"Initial Weights: {self.weights}")
            print(f"Initial Bias: {self.bias}")
            print(f"Number of observations: {self.n_obs}")
            print(f"Number of variables : {self.n_vars}")
            print("="*50)
            print(f"X\n{X}")
            print(f"Y\n{y}")
        for i in range(self.n_iters):
            y_pred = self.pred_proba(X)
            dw = 1.0 / n_obs * np.dot(X.T, (y_pred - y))
            db = 1.0 / n_obs * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print debugging information if verbose is True
            if self.verbose and i % (self.n_iters // 10) == 0:
                loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                print(f"Iteration {i}: Loss={loss}, Weights={self.weights}, Bias={self.bias}")

    def pred_proba(self, X):
        """
        Computes the predicted probabilities for the input data.

        Parameters:
        - X (array-like): Input features with shape (n_samples, n_features).

        Returns:
        - array-like: Predicted probabilities for each sample.
        """
        pred_lin = np.dot(X, self.weights) + self.bias
        pred_proba = sigmoid(pred_lin)
        return pred_proba

    def predict(self, X):
        """
        Predicts binary labels for the input data based on a 0.5 threshold.

        Parameters:
        - X (array-like): Input features with shape (n_samples, n_features).

        Returns:
        - array-like: Predicted binary labels (0 or 1) for each sample.
        """
        pred_proba = self.pred_proba(X)
        return (pred_proba >= 0.5).astype(int)
    

import torch

class MyTorchLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, verbose=False):
        """
        Initializes the MyLogisticRegression class with learning rate, number of iterations,
        and an optional verbose parameter for debugging information.

        Parameters:
        - n_features (int): Number of input features.
        - lr (float): Learning rate for gradient descent.
        - n_iters (int): Number of iterations to run gradient descent.
        - verbose (bool): If True, print debugging information during training.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.verbose = verbose
        
        # Initialize weights and bias
        self.weights = None
        self.bias = torch.zeros(1, requires_grad=True)

    def sigmoid(self, x):
        """
        Applies the sigmoid function.

        Parameters:
        - x (tensor): Input tensor.

        Returns:
        - tensor: Output tensor with sigmoid applied.
        """
        return 1 / (1 + torch.exp(-x))

    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        - X (tensor): Input features with shape (n_samples, n_features).
        - y (tensor): Target labels with shape (n_samples,).
        """

        self.weights = torch.zeros(X.shape[1], requires_grad=True)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        for i in range(self.n_iters):
            # Forward pass
            linear_model = torch.matmul(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute binary cross-entropy loss
            loss = -torch.mean(y * torch.log(y_pred + 1e-10) + (1 - y) * torch.log(1 - y_pred + 1e-10))

            # Backward pass
            loss.backward()

            # Update weights and bias
            with torch.no_grad():
                self.weights -= self.lr * self.weights.grad
                self.bias -= self.lr * self.bias.grad

                # Zero gradients after updating
                self.weights.grad.zero_()
                self.bias.grad.zero_()

            # Print debugging information if verbose is True
            if self.verbose and i % (self.n_iters // 10) == 0:
                print(f"Iteration {i}: Loss={loss.item()}")

    def predict_proba(self, X):
        """
        Predicts probabilities for the input data.

        Parameters:
        - X (tensor): Input features with shape (n_samples, n_features).

        Returns:
        - tensor: Predicted probabilities for each sample.
        """
        X = torch.tensor(X, dtype=torch.float32)
        linear_model = torch.matmul(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predicts binary labels for the input data based on a 0.5 threshold.

        Parameters:
        - X (tensor): Input features with shape (n_samples, n_features).

        Returns:
        - tensor: Predicted binary labels (0 or 1) for each sample.
        """
        return (self.predict_proba(X) >= 0.5).float()

