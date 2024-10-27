#################
# GENERATE DATA #
#################

import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 2 features
y = (0.5 + X[:, 0] + X[:, 1]  + X[:, 2]> 2).astype(int)  # Label based on the sum of features
X_train, X_test, y_train, y_test = train_test_split(X[:,:2], y, test_size=0.3, random_state=42)

###############################
# SKLEARN LOGISTIC REGRESSION #
###############################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Skikit logistic regression:\n")

skModel = LogisticRegression(penalty=None)
skModel.fit(X_train, y_train)

# Print the coefficients and intercept
print("Sklearn Coefficients:", skModel.coef_)
print("Sklearn Intercept:", skModel.intercept_)

y_pred_sk = skModel.predict(X_test)
y_proba_sk = skModel.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred_sk)
print(f"Accuracy: {accuracy}")
print("="*50)
print("\n"*2)

########################
# MyLogisticRegression #
########################

from MyLogisticRegression import MyLogisticRegression

print("My logistic regression:\n")

myModel = MyLogisticRegression(lr=1, n_iters= 10000, verbose=False)

myModel.fit(X_train, y_train)
print("MyModel Coefficients:", myModel.weights)
print("MyModel Intercept:", myModel.bias)

y_pred_my = myModel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_my)
print(f"Accuracy: {accuracy}")
print("="*50)
print("\n"*2)

#############################
# MyTorchLogisticRegression #
#############################

from MyLogisticRegression import MyTorchLogisticRegression

print("Torch logistic regression:\n")

myModel = MyTorchLogisticRegression(lr=1, n_iters= 10000, verbose=False)

myModel.fit(X_train, y_train)
print("Torch Coefficients:", myModel.weights.tolist())
print("Torch Intercept:", myModel.bias.tolist())

y_pred_my = myModel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_my)
print(f"Accuracy: {accuracy}")
print("="*50)
