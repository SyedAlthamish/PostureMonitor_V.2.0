'''{
    File Description: This file contains the algorithm to take Ml-Ready datasets and estimate and display performance in 
    classification using multiple classifiers and multiple validation schemes including LOVO. SVm,knn,elm,rf,custom_elm are
    the classifiers used
    }'''

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from hpelm import ELM  # hpelm's ELM library

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
# Utility function to load CSV files from folder
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
# LOPO (Leave-One-Patient-Out) Validation function
# -----------------------
def lopo_validation(patient_data):
    # Store accuracies for each classifier
    acc_svm = []
    acc_rf = []
    acc_knn = []
    acc_elm = []      # custom ELM
    acc_hpelm = []    # hpelm classifier

    patient_ids = list(patient_data.keys())

    for test_id in patient_ids:
        # Prepare test set: current patient
        df_test = patient_data[test_id]
        X_test = df_test.iloc[:, :-1].values   # all columns except last (features)
        y_test = df_test.iloc[:, -1].values      # last column as label

        # Prepare training set: all other patients
        train_dfs = [patient_data[pid] for pid in patient_ids if pid != test_id]
        df_train = pd.concat(train_dfs, ignore_index=True)
        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values

        # --- SVM Classifier ---
        svm_clf = SVC(kernel='rbf', gamma='scale')
        svm_clf.fit(X_train, y_train)
        y_pred_svm = svm_clf.predict(X_test)
        acc_svm.append(accuracy_score(y_test, y_pred_svm))

        # --- Random Forest Classifier ---
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        acc_rf.append(accuracy_score(y_test, y_pred_rf))

        # --- k-Nearest Neighbors Classifier ---
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        knn_clf.fit(X_train, y_train)
        y_pred_knn = knn_clf.predict(X_test)
        acc_knn.append(accuracy_score(y_test, y_pred_knn))

        # --- Custom ELM Classifier ---
        elm_clf = ELMClassifier(n_hidden=100, activation='sigmoid', random_state=42)
        elm_clf.fit(X_train, y_train)
        y_pred_elm = elm_clf.predict(X_test)
        acc_elm.append(accuracy_score(y_test, y_pred_elm))

        # --- hpelm Classifier ---
        label_encoder_h = LabelEncoder()
        y_train_encoded = label_encoder_h.fit_transform(y_train)
        y_test_encoded = label_encoder_h.transform(y_test)
        classes = label_encoder_h.classes_
        n_classes = len(classes)
        n_features = X_train.shape[1]

        T_train = np.zeros((len(y_train_encoded), n_classes))
        for i, label in enumerate(y_train_encoded):
            T_train[i, label] = 1

        h_elm = ELM(n_features, n_classes, classification='c')
        h_elm.add_neurons(100, 'sigm')
        h_elm.train(X_train, T_train, 'c')
        Y_pred = h_elm.predict(X_test)
        y_pred_hpelm = np.argmax(Y_pred, axis=1)
        y_pred_hpelm = label_encoder_h.inverse_transform(y_pred_hpelm)
        acc_hpelm.append(accuracy_score(y_test, y_pred_hpelm))

        print(f"Patient {test_id}: SVM={acc_svm[-1]:.4f}, RF={acc_rf[-1]:.4f}, kNN={acc_knn[-1]:.4f}, Custom ELM={acc_elm[-1]:.4f}, hpelm ELM={acc_hpelm[-1]:.4f}")

    print("\n--- Average Accuracies (LOPO) ---")
    print(f"SVM: {np.mean(acc_svm):.4f}")
    print(f"Random Forest: {np.mean(acc_rf):.4f}")
    print(f"kNN: {np.mean(acc_knn):.4f}")
    print(f"Custom ELM: {np.mean(acc_elm):.4f}")
    print(f"hpelm ELM: {np.mean(acc_hpelm):.4f}")

# -----------------------
# Normal Split Validation function
# -----------------------
def normal_split_validation(patient_data, test_size=0.25, random_state=42):
    # Consolidate all patient data into one DataFrame
    all_data = pd.concat(list(patient_data.values()), ignore_index=True)
    X = all_data.iloc[:, :-1].values   # features
    y = all_data.iloc[:, -1].values      # label
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # --- SVM Classifier ---
    svm_clf = SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    # --- Random Forest Classifier ---
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    # --- k-Nearest Neighbors Classifier ---
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    
    # --- Custom ELM Classifier ---
    elm_clf = ELMClassifier(n_hidden=100, activation='sigmoid', random_state=random_state)
    elm_clf.fit(X_train, y_train)
    y_pred_elm = elm_clf.predict(X_test)
    acc_elm = accuracy_score(y_test, y_pred_elm)
    
    # --- hpelm Classifier ---
    label_encoder_h = LabelEncoder()
    y_train_encoded = label_encoder_h.fit_transform(y_train)
    y_test_encoded = label_encoder_h.transform(y_test)
    classes = label_encoder_h.classes_
    n_classes = len(classes)
    n_features = X_train.shape[1]
    
    T_train = np.zeros((len(y_train_encoded), n_classes))
    for i, label in enumerate(y_train_encoded):
        T_train[i, label] = 1
    
    h_elm = ELM(n_features, n_classes, classification='c')
    h_elm.add_neurons(100, 'sigm')
    h_elm.train(X_train, T_train, 'c')
    Y_pred = h_elm.predict(X_test)
    y_pred_hpelm = np.argmax(Y_pred, axis=1)
    y_pred_hpelm = label_encoder_h.inverse_transform(y_pred_hpelm)
    acc_hpelm = accuracy_score(y_test, y_pred_hpelm)
    
    print("\n--- Average Accuracies (Normal Split) ---")
    print(f"SVM: {acc_svm:.4f}")
    print(f"Random Forest: {acc_rf:.4f}")
    print(f"kNN: {acc_knn:.4f}")
    print(f"Custom ELM: {acc_elm:.4f}")
    print(f"hpelm ELM: {acc_hpelm:.4f}")

# -----------------------
# Example usage
# -----------------------
if __name__ == '__main__':
    folder_path = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\Trash'  # Update this path to your CSV folder
    # Load patient data from CSV files
    patient_data = load_patient_data(folder_path)
    
    # Define the specific features you want to keep (adjust these names to match your CSV columns)
    selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2']
    
    # Filter each dataframe to only retain the selected features and the label (last column)
    patient_data = filter_features(patient_data, selected_features)
    
    # Run Leave-One-Patient-Out cross-validation
    lopo_validation(patient_data)
    
    # Run normal train-test split validation on the consolidated dataset
    normal_split_validation(patient_data, test_size=0.25)
