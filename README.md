# LeNet-5-CNN
This code implements a deep learning pipeline using PyTorch to train and evaluate the LeNet-5 convolutional neural network model on the MNIST dataset. Here’s a step-by-step breakdown of how it works:

1. Imports and Device Setup
- Import Libraries: The code imports various libraries required for building and training a neural network with PyTorch. These include PyTorch itself, Matplotlib for plotting, and some utility functions for data handling and neural network components.
- Set Device: It checks if a GPU is available and sets the computing device to GPU if available; otherwise, it uses the CPU. This is important for accelerating training.

2. Define Image Transformations
- Transform Function: A function called trans is defined to apply a series of transformations to the images. Specifically, it resizes the images to 32x32 pixels and converts them to tensors, which are the data format PyTorch uses.
  
3. Load the MNIST Dataset
- Download and Create Datasets: The code downloads the MNIST dataset, which consists of handwritten digits. It creates two datasets: one for training and one for testing.
- Data Loaders: It then creates DataLoader objects for these datasets. DataLoaders handle batching (grouping data into batches) and shuffling (randomizing the order of data), which are essential for training neural networks.
  
4. Define the LeNet-5 Model
- Model Architecture: The LeNet-5 model is defined as a class. This model is a type of convolutional neural network (CNN) that has several layers:
   --Three convolutional layers that apply filters to the input images to detect features.
  
   --Activation functions (Tanh) that introduce non-linearity.

   --Max-pooling layers that reduce the spatial dimensions of the feature maps.

Fully connected layers that combine all the features to make the final classification.

- Instantiate Model: An instance of the LeNet-5 model is created and moved to the selected device (CPU or GPU).

5. Define Training and Evaluation Functions
- Training Function: A function called train_model is defined to handle the training and evaluation of the model. It performs the following steps:
- Training Loop: For each epoch (one complete pass through the training data):
- Set the model to training mode.
- Loop over the training data in batches.
- Perform a forward pass (compute the output of the model).
- Compute the loss (difference between the predicted and true labels).
- Perform a backward pass (compute the gradients).
- Update the model parameters using the optimizer.
- Track Loss: Calculate and store the average loss for the epoch.
- Evaluation Loop: After each epoch, evaluate the model on the test data:
  
   -- Set the model to evaluation mode.
  
   -- Loop over the test data in batches.
  
   -- Perform a forward pass and compute the accuracy.
  
- Track and print the accuracy for the epoch.
  
The function returns lists of training losses and test accuracies for each epoch.

6. Training Loop
- Set Loss Function and Optimizer: Define the loss function (CrossEntropyLoss) and the optimizer (Adam). The loss function measures how well the model's predictions match the true labels. The optimizer updates the model parameters to minimize the loss.
- Train the Model: Call the train_model function to train the model for a specified number of epochs (10 in this case). This function will update the model parameters and track the performance metrics.
  
7. Save the Model
- Save Parameters: Save the trained model's parameters to a file. This allows you to load the model later without retraining it.
8. Plotting Results
- Plot Training Loss: Create a plot of the training loss over the epochs to visualize how the loss decreases as the model trains.
- Plot Test Accuracy: Create a plot of the test accuracy over the epochs to visualize how the model's performance on unseen data improves over time.
