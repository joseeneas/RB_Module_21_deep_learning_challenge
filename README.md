# Module 21 Deep Learning Challenge

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

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

- What variable(s) are the target(s) for your model? A: **IS_SUCCESSFUL**

- What variable(s) are the feature(s) for your model? A: **ALL THE OTHER COLUMNS**

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
  
## Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:

  **Overview of the analysis:**
    Explain the purpose of this analysis.

  **Results:**
    Using bulleted lists and images to support your answers, address the following questions:
  **Data Preprocessing**
  
      What variable(s) are the target(s) for your model?
      
      What variable(s) are the features for your model?
      
      What variable(s) should be removed from the input data because they are neither targets nor features?
  **Compiling, Training, and Evaluating the Model**
  
     How many neurons, layers, and activation functions did you select for your neural network model, and why?
     
     Were you able to achieve the target model performance?
     
     What steps did you take in your attempts to increase model performance?

  **Summary:
    Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then     explain your recommendation.
