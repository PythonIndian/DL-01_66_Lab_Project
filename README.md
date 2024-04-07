# DL-01_66_Lab_Project
Tom and Jerry Image Classification

# Context
Tom and Jerry is an American animated media franchise and series of comedy short films created in 1940 by William Hanna and Joseph Barbera. Best known for its 161 theatrical short films by Metro-Goldwyn-Mayer, the series centers on the rivalry between the titular characters of a cat named Tom and a mouse named Jerry.

This is one of the famous cartoon shows, that we would have never missed watching during our childhood. Now, its time to use our deep learning skills to detect our favorite characters - Tom and Jerry in the images extracted from some of the shows.

# Data Preparation
This dataset contains more than 5k images (exactly 5478 images) extracted from some of Tom & Jerry's show videos, that are available online.
The downloaded videos are converted into images with 1 frame per second (1 FPS).

Labeling for these images is done manually (by going through images one by one to tag them as 1 of the 4 outcomes), so the accuracy of ground_truth is 100%
Labeled images are separated into 4 different folders as given.

Folder - tom_and_jerry
SubFolder tom - contains images only with 'tom'
SubFolder jerry - contains images only with 'jerry'
SubFolder tom_jerry_1 - contains images with both 'tom' and 'jerry'
SubFolder tom_jerry_0 - contains images without both the characters

ground_truth.csv file contains labeled data against each image file.

# To overcome through challenges in dataset
Detect the presence of characters 'tom' and 'jerry' in any given image
Find the total screen time of Tom only, Jerry only, with both Tom & Jerry, without Tom & Jerry (Note: each image is extracted with 1FPS, which can help us to estimate the total screening time in seconds of Tom & Jerry after we detect these characters in these images)
Estimate maximum screen time showing both Tom & Jerry in the show (continuously) (Note: The image name is created based on the order of appearance, say file name - frame500.jpg indicates the frame extracted at 500th second of the video.

# Challenges in the data
There are images that can be challenged during training in image classification, as these images are distorted in the original size or shape and color of the characters. These image details are given in csv file - challenges.csv Doing error analysis on these images after model training, will help us understand how to improve the score.

# Internal-01 
Performed Preprocessing on dataset and done data augmentation on Kaggle explained the preprocessing and libraries to the subject teacher.

# Internal-02 
To perform 3 different models performed in prac-05 as given  perform various models on dataset and submit on GC.

# Models Training and gaining accuracy for prediction.
Here first we have unzip files from Google drive and local directory.
Move specific folder from extracted zip file to the data directory.
Install and import the split folder package for dataset splitting.
Split data into training and testing sets with 80-20% ratio.
Rename the validation set directory to 'test'.

Imports necessary libraries for plotting and working with images.
Reads an image file (frame1000.jpg) located at a specific path.
Displays the image using Matplotlib's imshow() function.
Shows the image plot inline in the Jupyter notebook.

The code imports TensorFlow, an open-source machine learning framework.
It imports ImageDataGenerator from tensorflow.keras.preprocessing.image, which generates batches of augmented image data for training deep learning models.
Sequential is imported from tensorflow.keras.models, allowing for the creation of sequential models.
Various layer classes such as Dense, Conv2D, Flatten, Dropout, MaxPooling2D, and Activation are imported from tensorflow.keras.layers, enabling the construction of neural network architectures.
image is imported from tensorflow.keras.preprocessing, providing utilities for image loading and preprocessing.
Matplotlib's pyplot module is imported as plt for creating plots and visualizations.
matplotlib.image is imported as mpimg for reading, writing, and displaying image data.

Defines the dimensions of images to be used for training and validation (150x150 pixels).
Specifies the directory path for the training data (train_data_dir) and the validation data (validation_data_dir).
Sets the number of training samples (nb_train_sample) and validation samples (nb_validation_samples) to 100 each.
Determines the number of training epochs (epochs) to be 20.
Sets the batch size (batch_size) to 20, indicating that the model will process 20 samples at a time during training.

train_datagen is created to generate augmented training images with rescaling, shearing, zooming, and horizontal flipping.
test_datagen is created to generate validation/test images with only rescaling.
train_generator is set up to generate batches of training data from the train_data_dir, with specified target size, batch size, and binary class mode.
validation_generator is set up to generate batches of validation/test data from the validation_data_dir, with specified target size and batch size, and binary class mode.

# CNN Model
Initializes a Sequential model, which represents a linear stack of layers.
Adds a 2D convolutional layer with 64 filters of size 3x3.
Introduces a ReLU activation function to introduce non-linearity to the model.
Incorporates a max pooling layer with a pool size of 2x2 to downsample the feature maps.
Adds a flattening layer to transform the 2D feature maps into a 1D vector.
Appends a fully connected layer with 64 neurons to the model.
Applies another ReLU activation function to the fully connected layer.
Includes an output layer with a single neuron, suitable for binary classification tasks.
Uses a sigmoid activation function in the output layer to produce class probabilities between 0 and 1.
Prints a summary of the model architecture, detailing the number of parameters in each layer and the total trainable parameters.

The compile() method configures the neural network model for training.
RMSprop optimizer is chosen for optimizing the model's parameters during training.
Binary cross-entropy loss function is selected, suitable for binary classification tasks.
Accuracy is chosen as the evaluation metric to monitor the model's performance during training and testing.
model.summary() prints a concise summary of the model architecture, including the type and size of each layer, the number of parameters, and the total trainable parameters in the model.

The fit_generator() method is used to train the model using data generated batch-by-batch by Python generators.
train_generator is the data generator for training images, which generates batches of augmented training data.
steps_per_epoch specifies the number of batches to yield from the training generator before completing one epoch. It's set to nb_train_sample, the total number of training samples divided by the batch size.
epochs specifies the number of complete passes through the entire training dataset during training.
validation_data specifies the data generator for validation/test images, which generates batches of validation/test data.
validation_steps specifies the number of batches to yield from the validation generator before completing one epoch. It's set to nb_validation_samples.
The fit_generator() function trains the model on the training data while validating it on the validation data after each epoch.
It returns a training object that contains information about the training process, including loss and accuracy metrics for both training and validation sets.

Matplotlib library is imported for creating plots and visualizations.
%matplotlib inline magic command ensures inline display of plots in the notebook.
training.history.keys() prints the keys of the history attribute, revealing available metrics.
Visualization of Model Accuracy:
Training and validation accuracy values are plotted across epochs.
The plot includes titles, axis labels, and a legend indicating "train" and "test" data.
Visualization of Model Loss:
Training and validation loss values are plotted across epochs.
The plot includes titles, axis labels, and a legend indicating "train" and "test" data.
plt.show() displays the plots showing the training and validation accuracy and loss values over epochs.

Matplotlib library is imported as plt for creating plots and visualizations.
%matplotlib inline magic command ensures that plots are displayed inline within the Jupyter notebook.
print(training.history.keys()) prints the keys of the history attribute of the training object to identify available metrics.
Visualization of Model Accuracy:
Training and validation accuracy values are plotted across epochs.
Title, axis labels, and legend are set to make the plot informative.
plt.show() displays the plot showing the training and validation accuracy values over epochs.
Visualization of Model Loss:
Training and validation loss values are plotted across epochs.
Title, axis labels, and legend are set to make the plot informative.
plt.show() displays the plot showing the training and validation loss values over epochs.

Image Loading and Preprocessing:

The image is loaded from a specified path and resized to 150x150 pixels.
It is converted to a NumPy array and expanded to include a batch dimension.
Prediction Process:

The neural network model predicts the class probabilities of the input image.
The predicted probabilities are printed to the console.
Final Prediction Determination:

The predicted probabilities are checked to determine the final class prediction.
If the probability is 1, the image is predicted to belong to class "jerry"; otherwise, it's predicted to belong to class "tom".
Image Display:

The loaded image is displayed using Matplotlib's functions.

Loads, preprocesses, and predicts the class of an image using a trained neural network model.
Determines the final prediction based on the predicted class probabilities.
Displays the image and prints the final prediction.

For further CNN models we have performed the same steps as done for the first CNN Model.
In this part,I have performed 3 models of CNN with different parameters and got different accuracy of three models:
CNN Model-01:
Accuracy: 60.97%
CNN Model-02:
Accuracy: 74.35%
CNN Model-03:
Accuracy: 39.1%



