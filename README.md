# Module 21 Deep Learning Challenge

## Project Files and components

- The program files are:
  - [ASCharityDeepLearning.ipynb](ASCharityDeepLearning.ipynb): The Jupyter Notebook file with the code for the analysis.
  - [ASCharityDeepLearning.py](ASCharityDeepLearning.py): The Python file with the code for the analysis.
- Output
  - [Output/AlphabetSoupCharity_Original_model.h5](Output/AlphabetSoupCharity_Original_model.h5)
  - [Output/AlphabetSoupCharity_Optimized_model.h5](Output/AlphabetSoupCharity_Optimized_model.h5)
  - [Output/AlphabetSoupCharity_Graph_Accuracy.png](Output/AlphabetSoupCharity_Graph_Accuracy.png)
  - [Output/AlphabetSoupCharity_Graph_Loss.png](Output/AlphabetSoupCharity_Graph_Loss.png)
  - [Output/AlphabetSoupCharity_Report.MD](Output/AlphabetSoupCharity_Report.MD)
  - [Output/AlphabetSoupCharity_Execution_Output.txt](Output/AlphabetSoupCharity_Execution_Output.txt)
- Other components
  - [README.md](README.md): This README file.
  - [https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv): : Input data files, in this case remotely stored in a URL.

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.

Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME            — Identification columns
- APPLICATION_TYPE        — Alphabet Soup application type
- AFFILIATION             — Affiliated sector of industry
- CLASSIFICATION          — Government organization classification
- USE_CASE                — Use case for funding
- ORGANIZATION            — Organization type
- STATUS                  — Active status
- INCOME_AMT              — Income classification
- SPECIAL_CONSIDERATIONS  — Special considerations for application
- ASK_AMT                 — Funding amount requested
- IS_SUCCESSFUL           — Was the money used effectively

## Step 1: Preprocess the Data

Use Pandas and scikit-learn’s to preprocess the dataset.

1.1. Read in the  charity_data.csv  to a Pandas DataFrame, and be sure to identify the following in your dataset:

- What variable(s) are the target(s) for your model? 
  - A: **_IS_SUCCESSFUL_**

- What variable(s) are the feature(s) for your model? 
  - A: **_ALL THE OTHER COLUMNS_**, except **_EIN_**, which is the identification column, and is discarded at the beginning of the preprocessing.

1.2. Drop the  EIN  and  NAME  columns.

1.3. Determine the number of unique values for each column.

1.4.For columns that have more than 10 unique values, determine the number of data points for each unique value.

1.5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value (Other) and then check if the binning was successful.

1.6. Use  pd.get_dummies()  to encode categorical variables.

1.7. Split the preprocessed data into a features array,  X , and a target array,  y . Use these arrays and the  train_test_split  function to split the data into training and testing datasets.

1.8 Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform  function.

## Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

2.1. Define parameters for the model

2.2. Compile the model

2.3. Train the model

2.4. Evaluate the model

## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%. Use any or all of the following methods to optimize your model:

  Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  Dropping more or fewer columns.
  
  Creating more bins for rare occurrences in columns.
  
  Increasing or decreasing the number of values for each bin.
  
  Add more neurons to a hidden layer.
  
  Add more hidden layers.
  
  Use different activation functions for the hidden layers.
  
  Add or reduce the number of epochs to the training regimen.
  
## Step 4: Report on the Neural Network Model

### Overview of the analysis

#### The purposed of the analysis is to create a model capable of determining  whether applicants will be successful if funded by Alphabet Soup

#### The target is to create a model that can have a verified accuracy of 75% or more

### Results

- Using bulleted lists and images to support your answers, address the following questions:

### Data Preprocessing
  
- What variable(s) are the target(s) for your model? 
  - **_IS_SUCCESSFUL_**

- What variable(s) are the features for your model? 
  - *ALL THE OTHER COLUMNS_**, except **_EIN_**, which is the identification column, and is discarded at the beginning of the preprocessing.

- What variable(s) should be removed from the input data because they are neither targets nor features?
  - As mentioned above **_EIN_** is a variable that does not add value to the analysis and can be discarded.

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why? Initially I used 2 hidden layers with 12 neurons respectively, and the activation function was **_relu_**. The output layer had 1 neuron and the activation function was **_sigmoid_**. The reason for this selection was to have a model with a good balance between accuracy and performance. But it was not performing as I expected so I made some adjustments: using "tanh" as the activation function

#### Were you able to achieve the target model performance? No, I was not on several "first" attempts. Changes hidden layers and neurons cause very odd results, but never close to 75%.

#### What steps did you take in your attempts to increase model performance? Besides changing hidden layers, neurons and activation functions, I noticed that the quantity and variety of features could be a factor. So I decided to apply binning to more categorical columns that I considered impactful for the analysis. I also increased the number of epochs to 100

### Summary

#### After several attempts to optimize the model, I was able to achieve the target model performance. The steps I took were: binning more categorical columns, increasing the number of epochs to 100, and changing the activation function to "tanh". The final accuracy was 75.5%

#### Is this final model something I could recommend? Yes, I would recommend this model because it has a good balance between accuracy and performance. But I would also recommend to keep trying to improve the model, maybe by adding more hidden layers and neurons, or changing the activation function to "relu".