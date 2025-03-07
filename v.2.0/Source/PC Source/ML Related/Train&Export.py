'''{
    This file trains the selected classifiers on all the available data and then exports them to the appropriate directory
    }'''
    
    

'''
    File Description: This file trains multiple classifiers (SVM, RF, kNN, Custom ELM, and hpelm ELM)
    using the entire ML-Ready dataset and then exports the trained models to disk.
'''

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
from hpelm import ELM  # hpelm's ELM library
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# -----------------------
# Define a simple custom ELM classifier (for comparison)
# -----------------------
class ELMClassifier:
    def __init__(self, n_hidden=100, activation='sigmoid', random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _activation(self, X):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError("Unknown activation function.")

    def fit(self, X, y):
        # Encode labels to one-hot
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        n_samples = y_encoded.shape[0]
        n_classes = len(np.unique(y_encoded))
        T = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y_encoded):
            T[i, label] = 1

        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.W = np.random.randn(n_features, self.n_hidden)
        self.b = np.random.randn(self.n_hidden)

        H = self._activation(np.dot(X, self.W) + self.b)
        self.beta = np.dot(np.linalg.pinv(H), T)
        return self

    def predict(self, X):
        H = self._activation(np.dot(X, self.W) + self.b)
        Y = np.dot(H, self.beta)
        y_pred = np.argmax(Y, axis=1)
        return self.label_encoder.inverse_transform(y_pred)

# -----------------------
# Utility function to load CSV files from a folder
# -----------------------
def load_patient_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    patient_data = {}
    for file in csv_files:
        # Use filename (without extension) as patient ID
        patient_id = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        patient_data[patient_id] = df
    return patient_data

# -----------------------
# Function to filter out only the selected features and the label column
# -----------------------
def filter_features(patient_data, selected_features):
    filtered_data = {}
    for patient, df in patient_data.items():
        # Assume the label is in the last column
        label_column = df.columns[-1]
        # Keep only the selected features and the label
        df_filtered = df[selected_features + [label_column]]
        filtered_data[patient] = df_filtered
    return filtered_data

# -----------------------
# Train all classifiers on the entire dataset and export them
# -----------------------
def train_and_export_models(patient_data, output_dir, random_state=42):
    # Consolidate all patient data into one DataFrame
    all_data = pd.concat(list(patient_data.values()), ignore_index=True)
    # All columns except the last are features; last column is the label
    X = all_data.iloc[:, :-1].values
    y = all_data.iloc[:, -1].values
    
    ##### SVM
    # Define a parameter grid for SVM
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }

    # Initialize an SVM classifier with the RBF kernel
    svm = SVC(kernel='rbf')

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search_svm = GridSearchCV(estimator=svm,
                                param_grid=param_grid_svm,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1)

    # Fit the grid search to your dataset (X, y)
    grid_search_svm.fit(X, y)

    # Retrieve the best estimator
    best_svm = grid_search_svm.best_estimator_

    print("Best SVM parameters:", grid_search_svm.best_params_)
    
    ##### RF
    # --- Optimize Random Forest Classifier ---
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=random_state)
    grid_search_rf = GridSearchCV(estimator=rf, 
                                param_grid=param_grid_rf, 
                                cv=5, 
                                scoring='accuracy', 
                                n_jobs=-1)
    grid_search_rf.fit(X, y)
    best_rf = grid_search_rf.best_estimator_
    print("Best Random Forest parameters:", grid_search_rf.best_params_)

    ##### KNN
    # --- Optimize k-Nearest Neighbors Classifier ---
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(estimator=knn, 
                                param_grid=param_grid_knn, 
                                cv=5, 
                                scoring='accuracy', 
                                n_jobs=-1)
    grid_search_knn.fit(X, y)
    best_knn = grid_search_knn.best_estimator_
    print("Best kNN parameters:", grid_search_knn.best_params_)
    
    
    # --- Train Custom ELM Classifier ---
    elm_clf = ELMClassifier(n_hidden=100, activation='sigmoid', random_state=random_state)
    elm_clf.fit(X, y)
    
    # --- Train hpelm Classifier ---
    # Use LabelEncoder to encode labels and create one-hot targets
    label_encoder_h = LabelEncoder()
    y_encoded = label_encoder_h.fit_transform(y)
    classes = label_encoder_h.classes_
    n_classes = len(classes)
    n_features = X.shape[1]
    
    T_train = np.zeros((len(y_encoded), n_classes))
    for i, label in enumerate(y_encoded):
        T_train[i, label] = 1
    
    h_elm = ELM(n_features, n_classes, classification='c')
    h_elm.add_neurons(100, 'sigm')
    h_elm.train(X, T_train, 'c')
    
    # -----------------------
    # Export the models
    # -----------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    joblib.dump(svm_clf, os.path.join(output_dir, 'svm_model.pkl'))
    joblib.dump(rf_clf, os.path.join(output_dir, 'rf_model.pkl'))
    joblib.dump(knn_clf, os.path.join(output_dir, 'knn_model.pkl'))
    joblib.dump(elm_clf, os.path.join(output_dir, 'custom_elm_model.pkl'))
    
    # For hpelm, we export both the model and the label encoder since you'll need them together
    with open(os.path.join(output_dir, 'hpelm_model.pkl'), 'wb') as f:
        pickle.dump({'model': h_elm, 'label_encoder': label_encoder_h}, f)
    
    print("Models have been trained on the full dataset and exported to:", output_dir)

# -----------------------
# Example usage
# -----------------------
if __name__ == '__main__':
    # Update the folder_path to your CSV folder
    folder_path = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\Trash'
    # Load patient data from CSV files
    patient_data = load_patient_data(folder_path)
    
    # Define the specific features you want to keep (adjust these names to match your CSV columns)
    selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 
                         'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 
                         'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2']
    
    # Filter each dataframe to only retain the selected features and the label (last column)
    patient_data = filter_features(patient_data, selected_features)
    
    # Specify the directory where you want to save the trained models
    output_directory = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\models'
    
    # Train all classifiers on all rows and export them
    train_and_export_models(patient_data, output_directory)
