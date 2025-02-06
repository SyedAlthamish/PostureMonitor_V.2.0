'''{
File Desc: This is a basic draft file where relevancy of the features from the posture monitoring dataset 
           to its corresponding labels is found, specifically through the Pearson&Spearman Coefficients test, and other relevant data
           including plotting it.
}'''

#Importion
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pandas.api.types import CategoricalDtype



#Definitions
noofsensor = 3
nooffeatures = 11
feature_column_base = 'sensor-'
class_column = 'posture'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('datasets/combined_filtered_labeled_output_1.csv')
# Encode labels as numerical values

# Get unique class names in the order they first appear in the DataFrame
desired_order = df['posture'].unique()

# Define the categorical type with the specific order
cat_type = CategoricalDtype(categories=desired_order, ordered=True)

# Convert 'class_column' to categorical with the specified order and then encode
df['posture'] = df['posture'].astype(cat_type)
df['label_encoded'] = df[class_column].cat.codes



#main dataframes on which the analysis data is stored
anova_results = pd.DataFrame(columns=['Feature', 'F-statistic', 'Between-Group Variance (MSB)', 'Within-Group Variance (MSW)']) #the final analysis report storage variable as a pandas dataframe
feature_details = pd.DataFrame(columns=['feature','class','count','mean','std'])                                                       #details about each feature from each class
spear_pear_results= pd.DataFrame(columns=[ 'Feature','Pearson\'s Coeff','Spearman\'s Coeff']) #the final analysis report storage variable as a pandas dataframe


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

def plot_spearpear_results():
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y=spearpear_results['Pearson\'s Coeff'].abs(), data=spearpear_results, width=0.5, palette="Set2")
    plt.title(f'Boxplot of \'{y_parameter}\' Values by Class for feature \'{feature_name}\'')
    plt.xlabel('Class')
    plt.ylabel(f'\'{y_parameter}\' Value')
    plt.show()

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y=spearpear_results['Spearman\'s Coeff'].abs(), data=spearpear_results, width=0.5, palette="Set2")
    plt.title(f'Boxplot of \'{y_parameter}\' Values by Class for feature \'{feature_name}\'')
    plt.xlabel('Class')
    plt.ylabel(f'\'{y_parameter}\' Value')
    plt.show()


def calculate_SPEAR_PEAR_details(feature_name,dataframe):
    # Calculate Pearson correlation
    pearsoncorr, _ = pearsonr(dataframe[feature_name], dataframe['label_encoded'])

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(dataframe[feature_name], dataframe['label_encoded'])
    
    spear_pear_results.loc[len(spear_pear_results)]=[feature_name,pearsoncorr,spearman_corr]
    return spear_pear_results
        

# Loop to run calculate_ANOVA_details(feature_column, class_column,anova_results) &...
# ...calculate_Feature_details(feat_det,feature_col,class_col) functions for all feature columns for all classes
for sensor_no in range(1, noofsensor + 1):
    for feature_no in range(1, nooffeatures + 1):
        feature_column = feature_column_base + str(sensor_no) + '-' + str(feature_no)
        
        feature_details = calculate_Feature_details(feature_details,feature_column,class_column)
        anova_results = calculate_ANOVA_details(feature_column, class_column,anova_results)
        values_to_keep = ['NEURTRAL', 'SLOUCH_MILD', 'SLOUCH_MOD','SLOUCH_EXT']
        # Filter the DataFrame
        filtered_df = df[df['posture'].isin(values_to_keep)]
        spearpear_results = calculate_SPEAR_PEAR_details(feature_column,df)

print(anova_results)    #prints the pandas dataframe containing results from ANOVA
print(feature_details)  #prints the pandas dataframe containing results from std,mean,count of each feature and class
#plotbox('std','sensor-1-1')
print(spearpear_results)
#plot_spearpear_results()

#------------------------------------------------------------------------------------------
corr_matrix = df.drop(columns=['posture']).corr()


#----------------------------------------------------
from sklearn.feature_selection import mutual_info_classif

X = df.drop(columns=[class_column])  # Features (continuous)

y = df[class_column]                  # Target variable (categorical)

# Calculate mutual information scores for continuous features against the categorical target
mi_scores = mutual_info_classif(X, y, discrete_features=False)

# Create a DataFrame to hold the MI scores
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

# Sort the DataFrame by MI scores in descending order
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)

# Display the MI scores
print(mi_scores_df)

#---------------------------------------------------------------
import pandas as pd
import numpy as np
from itertools import combinations

def redundancy_index(X, Y):
    covariance = np.cov(X, Y)[0, 1]
    var_X = np.var(X)
    var_Y = np.var(Y)
    return (2 * covariance) / (var_X + var_Y)

# Get the feature columns (assuming all columns except the target are features)
feature_columns = df.drop(columns=['posture']).columns.tolist()  # Modify this if you have a target column to exclude

# Create an empty DataFrame to store redundancy index values
redundancy_results = pd.DataFrame(index=feature_columns, columns=feature_columns)

# Calculate redundancy index for each pair of features
for feature1, feature2 in combinations(feature_columns, 2):
    ri = redundancy_index(df[feature1], df[feature2])
    redundancy_results.loc[feature1, feature2] = ri
    redundancy_results.loc[feature2, feature1] = ri  # Symmetric, so fill both ways

# Display the redundancy index DataFrame
print(redundancy_results)
#---------------------------------------------------------

