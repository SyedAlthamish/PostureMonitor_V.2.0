# -*- coding: utf-8 -*-
"""
    To input data and create a KNN model of classification and to export as a pickle file
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\PICO Source & Libraries\syed_finaldemo_MLReady.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 200)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y= label.fit_transform(y)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
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

    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1


    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    dataset = dataset[~((dataset < lower_bound) | (dataset > upper_bound)).any(axis=1)]

    return dataset

# Apply the function to remove outliers from all 6 columns
#dataset = remove_outliers(dataset.iloc[:, :6])

label_mapping = dict(zip(label.classes_, label.transform(label.classes_)))

# Print the mapping
print("Label to encoded value mapping:")
for label, encoded_value in label_mapping.items():
    print(f"{label}: {encoded_value}")




#from sklearn.decomposition import PCA

#pca = PCA(n_components=2)

#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

#explained_variance = pca.explained_variance_ratio_

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 25, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report

y_true= y_test

cm = confusion_matrix(y_true, y_pred)
print(cm)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Precision
precision = precision_score(y_true, y_pred, average='macro')

# Recall
recall = recall_score(y_true, y_pred, average='macro')

# F1 Score
f1 = f1_score(y_true, y_pred, average='macro')

# ROC-AUC Score (for binary classification)
y_prob = classifier.predict_proba(X_test) # Get predicted probabilities
roc_auc = roc_auc_score(y_true, y_prob,average='macro', multi_class='ovr') # Use predicted probabilities and set multi_class parameter

# Classification Report (contains precision, recall, f1-score, and support)
class_report = classification_report(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(class_report)

import pickle
pickle.dump(classifier,open('model3.pkl','wb'))
print('dumped')
model1 = pickle.load(open('model3.pkl','rb'))
