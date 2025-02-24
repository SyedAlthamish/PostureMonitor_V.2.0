This folder contains the Structured,Labelled but Unclean Datasets that are extracted from raw datasets obtained directly from the MC's outputs with the irrelevant columns removed, however the irrelevant rows are kept as is for analysis. that are not needed for the machine learning models. but may be useful for analysis, diagnoses and debugging.

The format of each .txt file is as follows:
    6* 2 values of one sensor, 6 *2 values of another sensor, posture state (Each Column is also labelled)
    But the 'Transition' and 'Unknown' posture state are unremoved.