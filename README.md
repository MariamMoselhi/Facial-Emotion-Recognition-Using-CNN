Project Description: Facial Emotion Recognition Using CNN

Objective:
The goal of this project is to develop a Convolutional Neural Network (CNN) model that can accurately recognize facial emotions from grayscale images. The model is trained and validated using the "Face Expression Recognition Dataset" obtained from Kaggle.

Dataset:
The "Face Expression Recognition Dataset" consists of images labeled with different facial expressions, including emotions like Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The dataset was downloaded and split into training and validation sets.

Steps Involved:

Data Preparation:

The dataset was downloaded and unzipped.
A custom function was created to load images and their corresponding labels from the specified directories for training and validation data.
Visual analysis was performed to inspect the distribution of different emotion classes in the training data.
Data Preprocessing:

The images were resized to a fixed size of 48x48 pixels for consistency.
Normalization was applied to scale pixel values to a range of 0-1.
The training dataset was balanced using Random Oversampling to address class imbalance.
Model Architecture:

A CNN model was developed with multiple convolutional, batch normalization, pooling, and dropout layers to extract features and reduce overfitting.
The model ends with fully connected layers and a softmax activation function to classify the images into one of the seven emotion categories.
Model Training:

The model was compiled using the Adam optimizer and categorical cross-entropy loss function.
Early stopping and learning rate reduction callbacks were implemented to prevent overfitting and optimize the learning rate during training.
The model was trained on the balanced training set and validated on the split validation set.
Evaluation and Results:

The model’s performance was evaluated using classification reports on both validation and test sets.
Loss and accuracy curves were plotted to visualize the training progress.
Examples of predictions were displayed, showing true vs. predicted emotions on both validation and test sets.
Testing on New Images:

A function was created to test the model on new images by preprocessing them and predicting their corresponding emotions.
Predictions were displayed alongside the input images to visually assess the model’s performance.
Performance Metrics:

The model achieved significant accuracy in classifying emotions across all classes. Classification reports were generated to analyze precision, recall, and F1-scores for each class.
Conclusion: This project successfully built a robust CNN-based model capable of recognizing facial emotions. The model's ability to generalize was validated on unseen test images, demonstrating its potential for real-world applications, such as emotion analysis in human-computer interaction, customer experience enhancement, and mental health assessment.
