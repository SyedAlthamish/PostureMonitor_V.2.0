'''
{
    This file is used to globally normalize csv files contained within a folder,
    based on the collective mean and std.deviations.
} 
'''
import pandas as pd
import glob
import os


### Acquiring the global variables from a global file from combine_csv.py
global_df = pd.read_csv(r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\combined_output.csv')
features = global_df.drop('Posture',axis=1)
std=features.std()
means = features.mean()

### normalizing each file

#Folder Locations
inp_folder_name = r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready 5Lab" # folder containing unnormalized files
op_folder_name = r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready 5Lab\Normalized" # folder containing to-be-normalized files

#iterating through csv files
for item in os.listdir(inp_folder_name): # circulates through all items in folder
    
    if item.endswith(".csv"): # filters out non-csv files
        print("Running for:",item) #debugging code
        
        #declares file paths for each file iteration
        i_file_path = inp_folder_name + "\\" + item #path of the file in iteration
        op_file_path = op_folder_name + "\\" + item #path of the file in iteration
        
        i_df = pd.read_csv(i_file_path) # converts into pandas df 
        for col in i_df.columns[:-1]: # picking each feature col. at a time
            i_df[col] = (i_df[col] - means[col]) / std[col] # standardizing using basic formula of (x-mean/std)

        
        i_df.to_csv(op_file_path,index=False)   # saving back the normalized dataset
        
