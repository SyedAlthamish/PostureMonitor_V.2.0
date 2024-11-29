'''{
    File Desc: This is a basic draft file where relevancy of the features from the posture monitoring dataset 
        to its corresponding labels is found, specifically through the ANOVA test, and other relevant data
        including plotting it.
    }'''

#Importion
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Definitions
noofsensor = 3
nooffeatures = 11
feature_column_base = 'sensor-'
class_column = 'posture'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('datasets/combined_filtered_labeled_output_1.csv')

#main dataframes on which the analysis data is stored
anova_results = pd.DataFrame(columns=['Feature', 'F-statistic', 'Between-Group Variance (MSB)', 'Within-Group Variance (MSW)']) #the final analysis report storage variable as a pandas dataframe
feature_details = pd.DataFrame(columns=['feature','class','count','mean','std'])                                                       #details about each feature from each class

'''{
Func Name:  calculate_ANOVA_details(feature_column,class_column)
Input:  feature_column=string containing the name of a specific feature column
        class_column=string containing the name of the column containing labels/classes
        anova_df=pandas dataframe containing the columns 'Feature', 'F-statistic', 'Between-Group Variance (MSB)', 'Within-Group Variance (MSW)'
Output: returns a pandas dataframe(with above mentioned columns) containing a new row containing details from ANOVA calc
}'''
def calculate_ANOVA_details(feature_column, class_column,anova_df):
    # Group data by class
    grouped_data = [df[df[class_column] == class_value][feature_column] for class_value in df[class_column].unique()]
    overall_mean = np.mean(np.concatenate(grouped_data))
    
    # Number of groups and total sample size
    k = len(grouped_data)  # Number of groups
    N = sum(len(group) for group in grouped_data)  # Total number of samples
    
    # Calculate Between-Group Variance (MSB)
    SSB = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in grouped_data)
    df_between = k - 1
    MSB = SSB / df_between
    
    # Calculate Within-Group Variance (MSW)
    SSW = sum(sum((x - np.mean(group)) ** 2 for x in group) for group in grouped_data)
    df_within = N - k
    MSW = SSW / df_within
    
    # Manual F-statistic
    f_stat_manual = MSB / MSW
    
    # ANOVA Test using scipy for comparison
    f_stat, p_value = stats.f_oneway(*grouped_data)
    
    # Interpretation
    if p_value < 0.05:
        interpretation = "Statistically significant (reject null hypothesis)"
    else:
        interpretation = "Not statistically significant (fail to reject null hypothesis)"
    
    # Add the results to the DataFrame
    anova_df.loc[len(anova_results)] = [feature_column, f_stat, MSB, MSW]
    return anova_df

'''{
Func Name: calculate_Feature_details(feat_det,feature_col,class_col)
Input:  feat_det= pandas dataframe containing columns 'feature','class','count','mean','std'
        feature_col =  string containing the feature whose details need to be calculated
        class_col = string containing the name of the label/class column
Output: return a dataframe where N additional rows containing 'feature','class','count','mean','std' 
        of the feature_col feature for the N no of unique classes present in class_col column
}'''
def calculate_Feature_details(feat_det,feature_col,class_col):
    for class_value in df[class_col].unique():
            described_df = df[df[class_col] == class_value][feature_col].describe()
            feat_det.loc[len(feat_det)] = [feature_col,class_value,described_df.loc['count'], described_df.loc['mean'], described_df.loc['std']]
    return feat_det

'''{
Func Name:plotbox(y_parameter,feature_name)
Input:  y_parameter= string containing the value to be displayed on y axes {mean,std,count}
        feature_name =  string containing the name of the feature to be compared {'sensor-1-1',etc}
Output: returns a bar plot of 'feature_name' 's 'y_parameter' taken from the 'feature_details' dataframe
}'''
def plotbox(y_parameter,feature_name):        
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y=y_parameter, data=feature_details[feature_details['feature']==feature_name], width=0.5, palette="Set2")
    plt.title(f'Boxplot of \'{y_parameter}\' Values by Class for feature \'{feature_name}\'')
    plt.xlabel('Class')
    plt.ylabel(f'\'{y_parameter}\' Value')
    plt.show()
    
            
# Loop to run calculate_ANOVA_details(feature_column, class_column,anova_results) &...
# ...calculate_Feature_details(feat_det,feature_col,class_col) functions for all feature columns for all classes
for sensor_no in range(1, noofsensor + 1):
    for feature_no in range(1, nooffeatures + 1):
        feature_column = feature_column_base + str(sensor_no) + '-' + str(feature_no)
        feature_details = calculate_Feature_details(feature_details,feature_column,class_column)
        anova_results = calculate_ANOVA_details(feature_column, class_column,anova_results)


print(anova_results)    #prints the pandas dataframe containing results from ANOVA
print(feature_details)  #prints the pandas dataframe containing results from std,mean,count of each feature and class
plotbox('std','sensor-1-1')


