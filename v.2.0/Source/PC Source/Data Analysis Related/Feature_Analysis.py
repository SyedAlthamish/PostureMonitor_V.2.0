'''{
    File Description: This file contains the analysis of ML-Ready datasets' feature columns relevancy and redundancy
    with each other and with the labels respectively.
    }'''

import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder

# =======================
# Helper Functions
# =======================

def compute_entropy(labels):
    """Compute entropy of a discrete set of labels."""
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def compute_conditional_entropy(feature, y, n_bins=10):
    """
    Discretize the feature into n_bins and compute the conditional entropy H(Y|X).
    Here we use quantile-based binning.
    """
    # Create bins (using quantiles)
    try:
        bins = pd.qcut(feature, q=n_bins, duplicates='drop')
    except ValueError:
        # if the feature has too few unique values, use all unique values as bins
        bins = pd.cut(feature, bins=np.unique(feature))
    df_temp = pd.DataFrame({'bin': bins, 'y': y})
    cond_entropy = 0.0
    for b, group in df_temp.groupby('bin'):
        if len(group) == 0:
            continue
        H_y_given_bin = compute_entropy(group['y'].values)
        weight = len(group) / len(feature)
        cond_entropy += weight * H_y_given_bin
    return cond_entropy

# =======================
# Load Data
# =======================

# Replace 'data.csv' with your file name
df = pd.read_csv(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\NET_combined_filtered_labeled_output.csv')

# Assume all columns except the last are features, last column is multi-class label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# For methods that require numeric targets (e.g. Pearson, linear regression),
# encode the multi-class labels to numeric values.
if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y_numeric = le.fit_transform(y)
else:
    y_numeric = y.values

# Compute overall target entropy
target_entropy = compute_entropy(y)

# =======================
# Compute Metrics for Each Feature
# =======================

results = []

# mutual_info_classif and f_classif expect a 2D array for each feature.
for col in X.columns:
    feature = X[col].values
    # Ensure feature is numeric
    try:
        feature = feature.astype(float)
    except Exception as e:
        print(f"Skipping column {col} because it cannot be converted to float.")
        continue

    # Reshape feature to 2D array for scikit-learn methods
    feature_2d = feature.reshape(-1, 1)

    # --- 1. Pearson’s r ---
    try:
        pearson_r, pearson_p = pearsonr(feature, y_numeric)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan

    # --- 2. Spearman’s rank correlation ---
    try:
        spearman_r, spearman_p = spearmanr(feature, y_numeric)
    except Exception:
        spearman_r, spearman_p = np.nan, np.nan

    # --- 3. R² from linear regression ---
    lr = LinearRegression()
    lr.fit(feature_2d, y_numeric)
    r2 = lr.score(feature_2d, y_numeric)

    # --- 4. Mutual Information ---
    # Note: mutual_info_classif handles continuous features and a discrete target.
    mi = mutual_info_classif(feature_2d, y, discrete_features=False, random_state=42)[0]

    # --- 5. ANOVA F-statistic ---
    F_val, f_p = f_classif(feature_2d, y)
    F_val = F_val[0]
    f_p_val = f_p[0]

    # --- 6. Entropy Reduction (Information Gain) ---
    cond_entropy = compute_conditional_entropy(feature, y, n_bins=10)
    info_gain = target_entropy - cond_entropy

    results.append({
        'Feature': col,
        'Pearson r': pearson_r,
        'Pearson p': pearson_p,
        'Spearman r': spearman_r,
        'Spearman p': spearman_p,
        'R²': r2,
        'Mutual Info': mi,
        'ANOVA F': F_val,
        'ANOVA p': f_p_val,
        'Info Gain': info_gain
    })

# Convert results to a DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# =======================
# Redundancy Check between Features
# =======================

# Compute the absolute correlation matrix for the features
corr_matrix = X.corr().abs()

# Generate a mask for the upper triangle (excluding the diagonal)
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_corr = corr_matrix.where(mask)

# Define a threshold for high correlation (redundancy)
redundancy_threshold = 0.8

# Find feature pairs with correlation greater than the threshold
redundant_pairs = []
for i in range(upper_corr.shape[0]):
    for j in range(i + 1, upper_corr.shape[1]):
        if upper_corr.iloc[i, j] > redundancy_threshold:
            redundant_pairs.append((upper_corr.index[i], upper_corr.columns[j], upper_corr.iloc[i, j]))

print("\nRedundant feature pairs (correlation > {}):".format(redundancy_threshold))
if redundant_pairs:
    for pair in redundant_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
else:
    print("No redundant feature pairs found.")