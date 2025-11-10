# Project 2 in FYSSTK3155/4155 - Data Analysis and Machine Learning

**Niklas Vestskogen & Eirik Holm**




To install the required packages to run the project code, clone the repository and run the command ```pip install -r requirements.txt```


## Code
In the Code folder you will find all relevant code used in throughout this project:

The  *FFNN.py* file contains the basis of our feed-forward neural network implementation, including various gradient descent methods and network training.

The *activation_functions.py* and *cost_functions.py* files contain all activation and cost functions used in the project, including some of their derivatives.

Lastly, all variants of *FFNN_..._testing.ipynb* jupyter nodebooks contain our testing and use of the previously mentioned implementations. Separate notebooks are found for general FFNN testing, regression problem testing and classification testing, as well as a final classification analysis using the full MNIST and fashion-MNIST datasets. CSV files containing results from classification testing can be found in the folder *Classification_CSV_Results*.
