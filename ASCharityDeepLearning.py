# %% [markdown]
# # Module 21 Deep Learning Challenge
# 

# %% [markdown]
# Step 1 - Preprocessing

# %% [markdown]
# Set up the environment, importing required modules

# %%
#
# Import our dependencies
#

import shutil                                          as shu
import warnings                                        as warn
import pandas                                          as pd
from   colorama                import Style            as st
from   colorama                import Fore             as fr
from   colorama                import Back             as bk
from   sklearn.model_selection import train_test_split as tts
from   sklearn.preprocessing   import StandardScaler   as sts
import tensorflow                                      as tf
from   tensorflow              import keras            as ker
from   keras.models            import Sequential       as seq
from   keras.layers            import Dense            as den

warn.filterwarnings('ignore')

# %% [markdown]
# Create Auxiliary Functions

# %%
#
# Auxiliary functions
#

w, h = shu.get_terminal_size()

def printSeparator():
    print(fr.GREEN + '-' * w + fr.WHITE)
    
def printStep(stepA, stepB):
    printSeparator()
    print(fr.BLUE   + stepA)
    print(fr.YELLOW + stepB)
    printSeparator()
    
def printDFinfo(name,dfName):
    printSeparator()
    print('Name: ',name)
    printSeparator()
    print(dfName.info())    
    printSeparator()
    print('Row Count :' + fr.RED)
    print(dfName.count(),fr.WHITE)
    printSeparator()
    print(dfName.head())
    printSeparator()
    
def printReport(reportName):
    printSeparator()
    print(fr.RED,'Classification Report',fr.WHITE)
    print(reportName)
    printSeparator()
    
def printBAS(basName):
    printSeparator()
    print(fr.WHITE + 'Balanced Accuracy Score : '+ fr.RED + str(basName))
    printSeparator()


# %% [markdown]
# Step 1   - Preprocess the Data
# 
# Step 1.1 - Read in the  charity_data.csv  to a Pandas DataFrame
# 
# Step 1.2 - Drop the  EIN. The column NAME cannot be removed, as it would make almost impossible to properly optimize the model

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.1 - Read in the charity_data.csv to a Pandas DataFrame')
printStep('Step 1   - Preprocess the Data',\
          'Step 1.2 - Drop the EIN column')

#
# read the charity_data.csv into a Pandas DataFrame
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
#

df_Application = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv") \
                   .drop(columns=['EIN'], axis=1)
printDFinfo('df_Application',df_Application)

# %% [markdown]
# 1.3. Determine the number of unique values for each column.

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.3 - Determine the number of unique values for each column.')
    
# 
# Determine the number of unique values in each column.
#

printSeparator()
print('df_application.nunique()')
printSeparator()
#print(df_Application.nunique())
print(df_Application.nunique())
printSeparator()

# %% [markdown]
#  Step 1.4 - For columns that have more than 10 unique values, determine the number of data points for each unique value.
# 

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.4 - For columns that have more than 10 unique values, determine the number of data points for each unique value.')
    

for column_Name in df_Application.columns:
    if df_Application[column_Name].nunique() > 10:
        print('Column Name', column_Name)
        printSeparator()
        print(df_Application[column_Name].value_counts())
        printSeparator()

# %% [markdown]
# Step 1.5 - Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value (Other) and then check if the binning was successful.

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.5 - Pick a cutoff point to bin "rare" categorical variables together in a new value (Other)')

# 
# Look at APPLICATION_TYPE value counts for binning
#

app_counts = df_Application['APPLICATION_TYPE'].value_counts()
printSeparator()
print('APPLICATION_TYPE value counts')
print(app_counts)

#
# Choose a cutoff value and create a list of application types to be replaced
#

application_types_to_replace = list(app_counts[app_counts < 500].index)
printSeparator()
print('application_types_to_replace')
print(application_types_to_replace)

#
# Replace in dataframe
#

for app in application_types_to_replace:
    df_Application['APPLICATION_TYPE'] = df_Application['APPLICATION_TYPE'].replace(app,"Other")

# 
# Check to make sure binning was successful
#

printSeparator()
print('After binning')
printSeparator()
print(df_Application['APPLICATION_TYPE'].value_counts())
printSeparator()

# %%
# 
# Look at CLASSIFICATION value counts for binning
#

class_counts = df_Application['CLASSIFICATION'].value_counts()
printSeparator()
print('CLASSIFICATION value counts')
print(class_counts)

# 
# You may find it helpful to look at CLASSIFICATION value counts >1
#

class_counts_gt1 = class_counts.loc[class_counts > 1]
printSeparator()
print('CLASSIFICATION value counts > 1')
class_counts_gt1.head()

# 
# Choose a cutoff value and create a list of classifications to be replaced
#

classifications_to_replace = list(class_counts[class_counts < 1000].index)
printSeparator()
print('classifications_to_replace')
print(classifications_to_replace)
#
# Replace in dataframe
#

for cls in classifications_to_replace:
    df_Application['CLASSIFICATION'] = df_Application['CLASSIFICATION'].replace(cls,"Other")

#    
# Check to make sure binning was successful
#
printSeparator()
print('After binning')
df_Application['CLASSIFICATION'].value_counts()


# %%
# 
# Look at ASK_AMT value counts for binning
#

app_counts = df_Application['ASK_AMT'].value_counts()
printSeparator()
print('ASK_AMT value counts')
print(app_counts)

#
# Choose a cutoff value and create a list of application types to be replaced
#

ask_amt_types_to_replace = list(app_counts[app_counts < 500].index)
printSeparator()
print('ask_amt_types_to_replace')
print(ask_amt_types_to_replace)

#
# Replace in dataframe
#

for app in ask_amt_types_to_replace:
    df_Application['ASK_AMT'] = df_Application['ASK_AMT'].replace(app,"Other")

# 
# Check to make sure binning was successful
#

printSeparator()
print('After binning')
printSeparator()
print(df_Application['ASK_AMT'].value_counts())
printSeparator()

# %%
# 
# Look at NAME value counts for binning
#

app_counts = df_Application['NAME'].value_counts()
printSeparator()
print('NAME value counts')
print(app_counts)

#
# Choose a cutoff value and create a list of application types to be replaced
#

names_to_replace = list(app_counts[app_counts < 100].index)
printSeparator()
print('names_to_replace')
print(names_to_replace)

#
# Replace in dataframe
#

for app in names_to_replace:
    df_Application['NAME'] = df_Application['NAME'].replace(app,"Other")

# 
# Check to make sure binning was successful
#

printSeparator()
print('After binning')
printSeparator()
print(df_Application['NAME'].value_counts())
printSeparator()

# %% [markdown]
# Step 1.6 - Use  pd.get_dummies()  to encode categorical variables.

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.6 - Use  pd.get_dummies()  to encode categorical variables.')
#
# Convert categorical data to numeric with `pd.get_dummies`
#

df_application_numeric = pd.get_dummies(df_Application)

# %% [markdown]
# Step 1.7 - Split the preprocessed data into a features array,  X , and a target array,  y . Use these arrays and the  train_test_split  function to split the data into training and testing datasets.

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.7 - Split the preprocessed data into features and a target arrays')

# 
# Split our preprocessed data into our features and target arrays
#

X = df_application_numeric.drop(['IS_SUCCESSFUL'], axis=1)
y = df_application_numeric['IS_SUCCESSFUL']

#
# Split the preprocessed data into a training and testing dataset
#

X_train, X_test, y_train, y_test = tts(X, y, random_state=58)

print('X_train.shape')
print(X_train.shape)
printSeparator()
print('X_test.shape')
print(X_test.shape)
printSeparator()
print('y_train.shape')
print(y_train.shape)
printSeparator()
print('y_test.shape')
print(y_test.shape)
printSeparator()

# %% [markdown]
# Step 1.8 - Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform  function.

# %%
#
# Log the processing progress
#

printStep('Step 1   - Preprocess the Data',\
          'Step 1.8 - Scale the training and testing features datasets')

#
# Create a Standard Scaler instance
# Fit the Standard Scaler
# Scale the data
#

scaler         = sts()
X_scaler       = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled  = X_scaler.transform(X_test)

#
# Log the processing progress
#

print('X_train_scaled.shape')
print(X_train_scaled.shape)
printSeparator()
print('X_test_scaled.shape')
print(X_test_scaled.shape)
printSeparator()

# %% [markdown]
# Step 2   - Compile, Train and Evaluate the Model
# 
# Step 2.1 - Define the model parameters
# 
# Step 2.2 - Compile the model
# 
# Step 2.3 - Train the model

# %%
#
# Log the processing progress
#

printStep('Step 2   - Compile, Train, and Evaluate the Model',\
          'Step 2.1 - Define the model parameters')

# 
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
#

number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1   = 12
hidden_nodes_layer2   = 12

nn_model = seq()

#
# First hidden layer
#

nn_model.add(den(units=hidden_nodes_layer1,input_dim=number_input_features, activation="tanh"))

#
# Second hidden layer
#

nn_model.add(den(units=hidden_nodes_layer2, activation="tanh"))

#
# Output layer
#

nn_model.add(den(units=1, activation="tanh"))

#
# Check the structure of the model
#

nn_model.summary()
printSeparator()
#
# Log the processing progress
#

printStep('Step 2   - Compile, Train, and Evaluate the Model',\
          'Step 2.2 - Compile the model')

# 
# Compile the model
#

nn_model.compile(loss      = 'binary_crossentropy', 
                 optimizer = 'adam', 
                 metrics   = ['accuracy'])
print('Model compiled')
printSeparator()

#
# Log the processing progress
#

printStep('Step 2   - Compile, Train, and Evaluate the Model',\
          'Step 2.3 - Train the model parameters')
# 
# Train the model
#

print('Model Training')
printSeparator()

fit_model = nn_model.fit(X_train_scaled,y_train,epochs=50)

printSeparator()
print('Model Training Complete')
printSeparator()

# %% [markdown]
# Step 2.4 - Evaluate the model

# %%
#
# Log the processing progress
#

printStep('Step 2   - Compile, Train, and Evaluate the Model',\
          'Step 2.4 - Evaluate the model parameters')
# 
# Evaluate the model using the test data
#

skip_optimization = False
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss:2.2f}, Accuracy: {model_accuracy:2.2f}")
accuracy = model_accuracy * 100
if (accuracy < 75):   
    print(F'Accuracy is {accuracy:2.2f}%, less than 75%')
    print("More optimization is required")
else:
    print(F'Accuracy is {accuracy:2.2f}%, greater than or equal to 75%')
    print("Model is optimized")
    #
    # Log the processing progress
    #
    printStep('Step 2   - Compile, Train, and Evaluate the Model',\
              'Step 2.5 - Export the model to a HDF5 file')
    # 
    # Export our model to HDF5 file
    #
    filename = 'Output/AlphabetSoupCharity_Original_model.h5'
    #
    # Save the model to a HDF5 file
    #

    nn_model.save(filename)
    printSeparator()
    print('Model saved to file : ',filename)
    printSeparator()
    print('End of processing')
    skip_optimization = True

# %% [markdown]
# Step 3   - Optimizing the Model
# 
# Step 3.1 - Attempt 1 - Add more neurons to a hidden layer
# 
#            (If automated attempts fail, then more code might be necessary to manually optimize)

# %%
#
# Log the processing progress
#
if skip_optimization == False:
  printStep('Step 3   - Optimizing the Model',\
           'Step 3.1 - Attempt 1 - Add more neurons to a hidden layer')
  number_input_features = len(X_train_scaled[0])
  print('Number of input features : ',number_input_features)
  for nodes in range(14, 140, 2):
    printSeparator()
    print('Number of hidden nodes   : ',nodes)
    hidden_nodes_layer1   = nodes
    hidden_nodes_layer2   = nodes
    hidden_nodes_layer3   = nodes
    hidden_nodes_layer4   = nodes
    hidden_nodes_layer5   = nodes
    nn_model2             = seq(name=f"Optimized_Model_{nodes}")
    nn_model2.add(den(units=hidden_nodes_layer1,input_dim=number_input_features, activation="tanh"))
    nn_model2.add(den(units=hidden_nodes_layer2, activation="tanh"))
    nn_model2.add(den(units=hidden_nodes_layer3, activation="tanh"))
    nn_model2.add(den(units=hidden_nodes_layer4, activation="tanh"))
    nn_model2.add(den(units=hidden_nodes_layer5, activation="tanh"))
    nn_model2.add(den(units=1,activation="tanh"))
    nn_model2.summary()
    printSeparator()
    print('Compile the model')
    nn_model2.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics   = ['accuracy'])
    print('Model compiled')
    printSeparator()
    print('Fit the model')
    fit_model2 = nn_model2.fit(X_train_scaled,y_train,epochs=50,verbose=0)
    print('Model fit')
    printSeparator()
    print('Evaluate the model')
    model_loss2, model_accuracy2 = nn_model2.evaluate(X_test_scaled,y_test,verbose=2)
    print(f"Loss: {model_loss2:2.2f}, Accuracy: {model_accuracy2:2.2f}")
    accuracy2 = model_accuracy2 * 100
    if (accuracy2 < 75):   
      print(F'Accuracy is {accuracy2:2.2f}, less than 75%')
      print("More optimization is required")
    else:
      print(F'Accuracy is {accuracy2:2.2f}, greater than or equal to 75%')
      print("Model is optimized")
      printStep('Step 3   - Optimizing the Model',\
               'Step 3.2 - Export the model to an HDF5 file')
      filename = 'Output/AlphabetSoupCharity_Optimized_model.h5'
      nn_model.save(filename)
      printSeparator()
      print('Model saved to file : ',filename)
      printSeparator()
      print('End of processing')
      break


