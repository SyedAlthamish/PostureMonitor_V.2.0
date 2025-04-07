import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import sys

# Input the data file
df = pd.read_csv(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\combined_output.csv')

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train the SVM model using One-vs-Rest
ovr_clf = OneVsRestClassifier(SVC(kernel='rbf', gamma='scale'))
ovr_clf.fit(X, y)

total_bytes = []

# Now extract parameters from each binary classifier
for idx, clf in enumerate(ovr_clf.estimators_):
    print(f"\nClassifier for class: {ovr_clf.classes_[idx]}")
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_
    bias = clf.intercept_
    
    #print("Support Vectors:\n", support_vectors)
    #print("Dual Coefficients:\n", dual_coef)
    #print("Biases:\n", bias)
    
    # Get total memory sizes for NumPy arrays
    print("Total size of support_vectors (N):", support_vectors.nbytes, "bytes")
    print("Total size of dual_coef (N):", dual_coef.nbytes, "bytes")
    
    # Get object sizes (may not include data size for NumPy arrays)
    print("Size of support_vectors:", sys.getsizeof(support_vectors), "bytes")
    print("Size of dual_coef:", sys.getsizeof(dual_coef), "bytes")
    print("Size of bias:", sys.getsizeof(bias), "bytes")

    total_bytes.append(sys.getsizeof(support_vectors) + 
                       sys.getsizeof(dual_coef) + 
                       sys.getsizeof(bias))
    
# Additionally, print classes
print("\nClasses:\n", ovr_clf.classes_)
print("Size of classes:", sys.getsizeof(ovr_clf.classes_), "bytes")
