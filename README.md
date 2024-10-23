# Akbank-Deep-Learning-Bootcamp
This repository contains my bootcamp project and its explanation.
This project classifies images of fish species using deep learning techniques. 
The project explores different models, starting from a basic Artificial Neural Network (ANN), followed by two improved versions with enhanced techniques, and finally, a transfer learning approach using a pre-trained MobileNetV2 model.
#Project Overview
This project aims to classify fish species based on images using deep learning. It uses four models:

A basic ANN model as a starting point.
Two improved ANN models that use activation functions, batch normalization, and regularization techniques.
A pre-trained MobileNetV2 model to improve performance through transfer learning.
#Dataset
The dataset consists of 9000 images of 9 different fish species. Data augmentation techniques are used to prevent overfitting and increase the model's robustness.

#Models
Model 1: Basic ANN Model
This is the baseline model, which uses a simple artificial neural network (ANN). It flattens the image data and passes it through three fully connected layers, each followed by ReLU activation and dropout.

#Model 2: Improved ANN with LeakyReLU
This model introduces LeakyReLU activation, which improves the handling of the vanishing gradient problem, along with Batch Normalization to standardize the inputs to each layer and further prevent overfitting.

#Model 3: Further Improved ANN with Increased Layers and Regularization
The third model builds on the previous improvement by adding more layers (up to 1024 units), with increased dropout and L2 regularization, to capture more complex patterns.

#Model 4: Transfer Learning with MobileNetV2
The final model leverages a pre-trained MobileNetV2 model, which is a state-of-the-art architecture trained on ImageNet. The model's top layers are removed and replaced with custom dense layers specific to fish classification.

Special thanks to the Akbank Deep Learning Bootcamp for providing the resources to work on this project and to TensorFlow for their amazing libraries and tools.

Here is the kaggle link:https://www.kaggle.com/code/beyzacankurtaran/fish-classification-akbank-deep-learning-bootcamp
