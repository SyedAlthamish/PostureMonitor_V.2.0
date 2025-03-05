import pandas as pd

# Read both CSV files into DataFrames
df1 = pd.read_csv(r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\Trash\syd.csv")
df2 = pd.read_csv(r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\syd.csv")

# Compare the DataFrames
if df1.equals(df2):
    print("The CSV files are the same.")
else:
    print("The CSV files are different.")
