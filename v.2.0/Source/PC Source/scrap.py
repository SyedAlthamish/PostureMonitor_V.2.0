import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Input the data file
df = pd.read_csv(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\combined_output.csv')

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train the SVM model
svm_clf = SVC(kernel='rbf', gamma='scale')
svm_clf.fit(X, y)

# Extract parameters
support_vectors = svm_clf.support_vectors_
dual_coef = svm_clf.dual_coef_
bias = svm_clf.intercept_
classes = svm_clf.classes_

# Print the extracted parameters
print("Support Vectors:\n", support_vectors)
print("Dual Coefficients:\n", dual_coef)
print("Biases:\n", bias)
print("Classes:\n", classes)
