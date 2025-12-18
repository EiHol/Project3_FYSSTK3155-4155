# Project 3 in FYSSTK3155/4155 - Data Analysis and Machine Learning

**Niklas Vestskogen & Eirik Holm**




To install the required packages to run the project code, clone the repository and run the command ```pip install -r requirements.txt```


## Code
In the Code folder you will find all relevant code used in throughout this project:

The *Data.ipynb* file contains the steps used for data loading, preprocessing and augmentation of the CXR dataset used throughout the project. Alle these steps have been combined into one function in the *load_data.py* file, which is imported and used in subsequent notebooks.

The data itself can be found in the folders, *chest_xray_data* and *chest_xray_data_split* folders, where the first contains the raw dataset as imported through Kaggle. And the latter containing our custom reshuffled and split dataset.

The *CNN testing.ipynb* file contains all testing and results from our CNN implementation, including hyperparameter tuning and visualization. 

The *RNN_testing.ipynb* file contains all hyperparameter tuning and following visualizations for the RNN model, wheras the *RNN_final_testing.ipynb* file contains the final model implementation and visualizations. CSV files containing various results from the RNN analysis can be found in the folder *CSV results*.
