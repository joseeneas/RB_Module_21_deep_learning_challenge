# Module 21 Deep Learning Challenge

## Report on the Neural Network Model

### Overview of the analysis

#### The purposed of the analysis is to create a model capable of determining  whether applicants will be successful if funded by Alphabet Soup. The target is to create a model that can have a verified accuracy of 75% or more

### Results

- Using bulleted lists and images to support your answers, address the following questions:

### Data Preprocessing
  
- What variable(s) are the target(s) for your model? **_IS_SUCCESSFUL_**

- What variable(s) are the features for your model? **_ALL THE OTHER COLUMNS_**, except **_EIN_**, which is the identification column, and is discarded at the beginning of the preprocessing.

- What variable(s) should be removed from the input data because they are neither targets nor features? As mentioned above **_EIN_** is a variable that does not add value to the analysis and can be discarded.

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why? Initially I used 2 hidden layers with 12 neurons respectively, and the activation function was **_relu_**. The output layer had 1 neuron and the activation function was **_sigmoid_**. The reason for this selection was to have a model with a good balance between accuracy and performance. But it was not performing as I expected so I made some adjustments: using "tanh" as the activation function

#### Were you able to achieve the target model performance? No, I was not on several "first" attempts. Changes hidden layers and neurons cause very odd results, but never close to 75%

#### What steps did you take in your attempts to increase model performance? Besides changing hidden layers, neurons and activation functions, I noticed that the quantity and variety of features could be a factor. So I decided to apply binning to more categorical columns that I considered impactful for the analysis. I also increased the number of epochs to 100

### Summary

#### After several attempts to optimize the model, I was able to achieve the target model performance. The steps I took were: binning more categorical columns, increasing the number of epochs to 100, and changing the activation function to "tanh". The final accuracy was 75.5%

#### Is this final model something I could recommend? Yes, I would recommend this model because it has a good balance between accuracy and performance. But I would also recommend to keep trying to improve the model, maybe by adding more hidden layers and neurons, or changing the activation function to "relu"
