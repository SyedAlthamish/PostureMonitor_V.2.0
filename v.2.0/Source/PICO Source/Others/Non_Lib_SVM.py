'''{
    This file is meant to run on PC and simulates 
    raw Non-Library SVM classification
    }'''
    
# svm_pico_ovo.py — MicroPython on RP2040

import json

# Load the JSON file
with open(r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\Source\PC Source\ML Related\Raw_svm_parameters.json", 'r') as f:
    params = json.load(f)

# Extract parameters
support_vectors = params['support_vectors']
dual_coef = params['dual_coef']
intercept = params['intercept']
classes = params['classes']
gamma = params['gamma']
support_vector_labels = params['support_vector_labels']



import math
'''
# ── Paste in your trained parameters here ────────────────────────────────────
# support_vectors: list of [n_features] floats, length = n_SV
support_vectors = [
    [ -44.71352,   -6.926781, -134.4893,  … ],
    [ -44.71822,   -6.922630, -134.4920,  … ],
    # …
]

# After fitting:
#   y: your original 1-D array of labels (shape = n_samples,)
#   clf.support_: indices of the support vectors in the original data
#support_vector_labels = y[clf.support_].tolist()
# support_vector_labels: class index for each support vector (length = n_SV)
support_vector_labels = [
    0, 0, 1, 2, 0, 4, …  # each entry is one of your original class labels
]

# dual_coef: shape = (n_classes-1, n_SV) as in sklearn’s SVC.dual_coef_ :contentReference[oaicite:0]{index=0}
dual_coef = [
    [ α₀₀, α₀₁, α₀₂, … ],   # row 0: opposing-class = classes_excl[0]
    [ α₁₀, α₁₁, α₁₂, … ],   # row 1: opposing-class = classes_excl[1]
    # …
]

# intercept: length = n_pairs = n_classes*(n_classes-1)/2, in lex order
#  (0 vs1), (0 vs2), … (0 vsN), (1 vs2), (1 vs3), …, (N-1 vs N) :contentReference[oaicite:1]{index=1}
intercept = [ b₀₁, b₀₂, b₀₃, …, b₁₂, b₁₃, … ]

# classes: sorted list of your class labels, length = n_classes
classes = [0, 1, 2, 3, 4]

# RBF gamma (must match your sklearn setting)
gamma = 0.05
# ─────────────────────────────────────────────────────────────────────────────
'''
# Precompute the list of class‐pairs in the same order as `intercept`
class_pairs = []
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        class_pairs.append((classes[i], classes[j]))
# Now len(class_pairs) == len(intercept)

def _rbf_kernel(x, sv):
    """Compute exp( -γ * ||x - sv||² )"""
    s = 0.0
    for xi, svi in zip(x, sv):
        print("DB:",xi,svi)
        d = xi - svi
        s += d*d
    return math.exp(-gamma * s)

def predict(x):
    """
    One-vs-One prediction:
     - For each pair (ci,cj), compute f_{ij}(x) = Σ α_{l,ij} K(x,sv_l) + b_{ij}
     - If f>0 vote for ci, else for cj
     - Return label with most votes
    """
    # 1) compute K_j = K(x, support_vectors[j]) once
    K = [ _rbf_kernel(x, sv) for sv in support_vectors ]

    # 2) initialize vote counts
    votes = {c: 0 for c in classes}

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) One‑vs‑One Voting Loop:
    #    For each pair of classes (ci, cj), we compute the decision function
    #    f_{ij}(x) = Σ_j [ α_{l,ij} * K(x, sv_l) ] + b_{ij}
    #    where α_{l,ij} is the dual coefficient for support vector l in the
    #    (ci vs cj) classifier, and b_{ij} is the corresponding intercept.
    #    We then cast a “vote” for ci if f_{ij}(x) > 0, else for cj.
    # ──────────────────────────────────────────────────────────────────────────────

    for pair_idx, (ci, cj) in enumerate(class_pairs):
        # Initialize the decision score for this binary classifier to zero
        s = 0.0

        # Loop over every support vector by index j
        for j, sv_label in enumerate(support_vector_labels):
            # Skip any support vectors that don't belong to either class in this pair
            if sv_label not in (ci, cj):
                continue

            # Determine which class is “other” in this pair relative to this SV’s label
            # If the SV’s label == ci, then the other class is cj; otherwise it’s ci
            other = cj if sv_label == ci else ci

            # Build a list of all classes except the SV’s own class, in the same
            # order used by dual_coef_ for that SV
            excl = [c for c in classes if c != sv_label]

            # Find the position (row) in dual_coef_ that corresponds to the “other” class
            # This gives us the correct α coefficient for SV j in the (ci vs cj) classifier
            r = excl.index(other)

            # Retrieve the dual coefficient α for this SV and this class‑pair
            α = dual_coef[r][j]

            # Only accumulate if α is non‑zero (saves a tiny bit of work)
            if α:
                # K[j] was precomputed as the RBF kernel between x and support_vectors[j]
                s += α * K[j]

        # After summing over all relevant support vectors, add the intercept b_{ij}
        s += intercept[pair_idx]

        # Cast a vote: if s > 0, the classifier leans toward class ci; otherwise cj
        if s > 0:
            votes[ci] += 1
        else:
            votes[cj] += 1


    # pick the class with highest votes
    best = max(votes, key=votes.get)
    return best

# ── Example usage ────────────────────────────────────────────────────────────
import pandas as pd
from pathlib import Path
df = pd.read_csv(
    Path(r'C:\Althamish\Project\PostureMonitor_V.2.0')
         /"v.2.0"
         /"data"
         /"V.3.0"
         /"v.3.0 ML-Ready 5Lab"
         /"5Lab_abd_ees_mad_phar2_sar_syd_syd2_comb.csv")
selected_features = ['tilt_x_S1', 'tilt_y_S1', 'tilt_z_S1', 'tilt_x_S2', 'tilt_y_S2', 'tilt_z_S2', 'xa_S1', 'ya_S1', 'xa_S2', 'ya_S2']
df_filt = df[selected_features]
listed_df = df_filt.to_csv(index=False,header=False)

for x in listed_df:
    lbl = predict(x)
    print("Input:", x, "→ Class:", lbl)
# ─────────────────────────────────────────────────────────────────────────────
