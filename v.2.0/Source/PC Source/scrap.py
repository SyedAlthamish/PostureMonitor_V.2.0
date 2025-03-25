'''{
    This file converts all csv files into one big file
    }'''

import pandas as pd
import glob
import os
df = pd.read_csv(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\combined_output.csv')
features = df.drop('Posture',axis=1)
std=features.std()
means = features.mean()
