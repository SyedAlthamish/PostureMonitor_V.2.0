#%% ############################################ SVM Classification Performance Check #########################################
'''{
    File Description: This file contains the algorithm to take ML-ready datasets and estimate and display performance in 
    classification using SVM with multiple validation schemes including LOPO.
}'''

import os
import glob
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------
# Utility function to load CSV files from folder
# -----------------------
def load_patient_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    patient_data = {}
    for file in csv_files:
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
        label_column = df.columns[-1]  # Assume the label is in the last column
        df_filtered = df[selected_features + [label_column]]
        filtered_data[patient] = df_filtered
    return filtered_data

# -----------------------
# LOPO (Leave-One-Patient-Out) Validation function
# -----------------------
def lopo_validation(patient_data):
    acc_svm = []
    patient_ids = list(patient_data.keys())

    for test_id in patient_ids:
        df_test = patient_data[test_id]
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

        train_dfs = [patient_data[pid] for pid in patient_ids if pid != test_id]
        df_train = pd.concat(train_dfs, ignore_index=True)
        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values

        svm_clf = SVC(kernel='rbf', gamma='scale')
        svm_clf.fit(X_train, y_train)
        y_pred_svm = svm_clf.predict(X_test)
        acc_svm.append(accuracy_score(y_test, y_pred_svm))

        print(f"Patient {test_id}: SVM={acc_svm[-1]:.4f}")

    print("\n--- Average Accuracy (LOPO) ---")
    print(f"SVM: {np.mean(acc_svm):.4f}")

# -----------------------
# Normal Split Validation function
# -----------------------
def normal_split_validation(patient_data, test_size=0.25, random_state=42):
    all_data = pd.concat(list(patient_data.values()), ignore_index=True)
    X = all_data.iloc[:, :-1].values
    y = all_data.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    svm_clf = SVC(kernel='rbf', gamma='scale')
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    print(f"\n--- TestSize:{test_size}; Average Accuracy (Normal Split) ---")
    print(f"SVM: {acc_svm:.4f}")

# -----------------------
# Example usage
# -----------------------
folder_path = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\Normalized'  # Update this path
patient_data = load_patient_data(folder_path)

selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2']
patient_data = filter_features(patient_data, selected_features)

lopo_validation(patient_data)
normal_split_validation(patient_data, test_size=0.25)
normal_split_validation(patient_data, test_size=0.5)
normal_split_validation(patient_data, test_size=0.75)


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
#%% ################################ Memory Cost Estimation (OVR) ############################
'''{
    This file trains the SVM classifier in One versus Rest strategy
    to figure out memory cost of classifier in pico
    }'''


import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Input the data file
file_name = 1
df = pd.read_csv(
    Path(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0')
    /"data"
    /"V.3.0"
    /"v.3.0 ML-Ready"
    /"combined_output.csv"
    )
    


# filter unnecessary features
df_filt = df[selected_features]
selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2','Posture']

# Separate features and target
X = df_filt.iloc[:, :-1].values
y = df_filt.iloc[:, -1].values

# Train the SVM model using One-vs-Rest
ovr_clf = OneVsRestClassifier(SVC(kernel='rbf', gamma='scale'))
ovr_clf.fit(X, y)

total_bytes = []
total_elements = []

# Now extract parameters from each binary classifier
for idx, clf in enumerate(ovr_clf.estimators_):
    print(f"\nClassifier for class: {ovr_clf.classes_[idx]}")
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_
    bias = clf.intercept_
    
    # calculating no of elements for each parameter
    NoEl_sup_vec = support_vectors.shape[0]*support_vectors.shape[1] #no of elements of support vector
    NoEl_dua_coe = dual_coef.shape[0]*dual_coef.shape[1] #no of elements of dual_coef
    NoEl_bias = bias.shape[0] #no of elements of bias
    
    print(idx,"No of elements in support vectors:", NoEl_sup_vec)
    print(idx,"No of elements in dual coeff:", NoEl_dua_coe)
    print(idx,"No of elements in bias:", NoEl_bias)
    
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

    #calculating total no of bytes per classifier in ovr
    total_bytes.append(sys.getsizeof(support_vectors) + 
                       sys.getsizeof(dual_coef) + 
                       sys.getsizeof(bias))
    
    #calculating total no of elements per classifier in ovr
    total_elements.append(NoEl_bias+NoEl_dua_coe+NoEl_sup_vec)

# Additionally, print classes
print("\nClasses:\n", ovr_clf.classes_)
print("Size of classes:", sys.getsizeof(ovr_clf.classes_), "bytes")
print("Absolute Total per Element:",total_bytes)
print("Absolute Total:",sum(total_bytes))

print("Total number of elements:",total_elements)
print("Absolute Total number of elements:",sum(total_elements))
print("Absolute Total number of elements[In Kb]:",(sum(total_elements)*4/1024), "Kb")


# %%################################ Memory Cost Estimation (OVO) ############################
'''{
    This file trains the SVM classifier in a standard configuration
    (without One-vs-Rest) to analyze memory cost and performance for
    direct deployment onto a constrained device like Raspberry Pi Pico.
    
}'''

import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Input the data file
file_name = 1
df = pd.read_csv(
    Path(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0')
    / "data"
    / "V.3.0"
    / "v.3.0 ML-Ready 5Lab"
    / "5Lab_abd_ees_mad_phar2_sar_syd_syd2_comb.csv"
)

# Select only the necessary features
selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2', 'Posture']
df_filt = df[selected_features]

# Separate features and target
X = df_filt.iloc[:, :-1].values
y = df_filt.iloc[:, -1].values

# Train the SVM model (multiclass natively handled by SVC)
clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')  # default is 'ovr', but could use 'ovo'
clf.fit(X, y)

# Extract learned parameters
support_vectors = clf.support_vectors_
dual_coef = clf.dual_coef_
bias = clf.intercept_

# Calculate element counts
NoEl_sup_vec = support_vectors.shape[0] * support_vectors.shape[1]
NoEl_dua_coe = dual_coef.shape[0] * dual_coef.shape[1]
NoEl_bias = bias.shape[0]

print("No of elements in support vectors:", NoEl_sup_vec)
print("No of elements in dual coeff:", NoEl_dua_coe)
print("No of elements in bias:", NoEl_bias)

# Memory sizes
print("Total size of support_vectors (N):", support_vectors.nbytes, "bytes")
print("Total size of dual_coef (N):", dual_coef.nbytes, "bytes")

# Object sizes (includes some metadata)
print("Size of support_vectors:", sys.getsizeof(support_vectors), "bytes")
print("Size of dual_coef:", sys.getsizeof(dual_coef), "bytes")
print("Size of bias:", sys.getsizeof(bias), "bytes")

# Total bytes and elements
total_bytes = sys.getsizeof(support_vectors) + sys.getsizeof(dual_coef) + sys.getsizeof(bias)
total_elements = NoEl_sup_vec + NoEl_dua_coe + NoEl_bias

print("Absolute Total bytes:", total_bytes)
print("Absolute Total number of elements:", total_elements)
print("Absolute Total Size [In Kb] with 4bytes per element:",(total_elements*4)/1024, "Kb")


# Print classes
print("Classes:", clf.classes_)
print("Size of classes:", sys.getsizeof(clf.classes_), "bytes")


# %%################################## Exporting Parameters of trained svm #################################
import numpy as np
import joblib

# Assuming 'clf' is your trained SVC model
support_vectors = clf.support_vectors_
dual_coef = clf.dual_coef_
intercept = clf.intercept_
classes = clf.classes_
gamma = clf._gamma  # or the value you set for gamma

# Save parameters to a file using joblib
joblib.dump({
    'support_vectors': support_vectors,
    'dual_coef': dual_coef,
    'intercept': intercept,
    'classes': classes,
    'gamma': gamma
}, 'Raw_svm_parameters.pkl')


