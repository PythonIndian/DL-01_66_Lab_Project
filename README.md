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
