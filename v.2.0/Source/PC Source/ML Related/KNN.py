# -*- coding: utf-8 -*-
"""
    To input data, create a KNN model for classification, and export it as a pickle file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset from CSV file
# Ensure the correct file path is provided
dataset = pd.read_csv(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\PICO Source & Libraries\syed_finaldemo_MLReady.csv")

# Separate features (X) and target labels (y)
X = dataset.iloc[:, :-1].values  # Extract all columns except the last as features
y = dataset.iloc[:, -1].values   # Extract the last column as the target variable

from sklearn.model_selection import train_test_split
# Split the dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)  # Encode target labels to numeric values

# Uncomment the below section if feature scaling is required
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# Uncomment to visualize feature distributions using box plots
'''
import seaborn as sns
sns.boxplot(data=dataset)
plt.figure(figsize=(20,10))
sns.boxplot(data=dataset.iloc[:,:6])
plt.title('Box Plot of Features')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()
'''

def remove_outliers(dataset):
    """
    Function to remove outliers using the IQR (Interquartile Range) method.
    """
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove rows where any column has outliers
    dataset = dataset[~((dataset < lower_bound) | (dataset > upper_bound)).any(axis=1)]
    return dataset

# Uncomment to apply outlier removal
# dataset = remove_outliers(dataset.iloc[:, :6])

# Create a mapping of original labels to encoded values
label_mapping = dict(zip(label.classes_, label.transform(label.classes_)))
print("Label to encoded value mapping:")
for label, encoded_value in label_mapping.items():
    print(f"{label}: {encoded_value}")

# Uncomment for PCA (Principal Component Analysis) for dimensionality reduction
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''

from sklearn.neighbors import KNeighborsClassifier
# Initialize KNN classifier with 25 neighbors and Minkowski distance metric (Euclidean distance)
classifier = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p=2)
classifier.fit(X_train, y_train)  # Train the model

# Predict the test dataset
y_pred = classifier.predict(X_test)

# Print predicted vs actual labels
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Compute ROC-AUC score (Only applicable for multi-class classification using One-vs-Rest)
y_prob = classifier.predict_proba(X_test)  # Get predicted probabilities
roc_auc = roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(class_report)

import pickle
# Save the trained model to a pickle file
pickle.dump(classifier, open('model3.pkl', 'wb'))
print('Model saved successfully')

# Load the model back from the pickle file
model1 = pickle.load(open('model3.pkl', 'rb'))
