# Classification-with-CNN.-Cifar-dataset
## This repository contains the implementation of various Convolutional Neural Network (CNN) architectures and their applications in image classification, object detection, and other computer vision tasks. The code is based on the comprehensive guide provided by Pinecone, which covers the fundamentals of CNNs, popular architectures, and practical examples using the CIFAR-10 dataset.

# Visual Guide to Applied Convolution Neural Networks

This repository contains the implementation of various Convolutional Neural Network (CNN) architectures and their applications in image classification, object detection, and other computer vision tasks. The code is based on the comprehensive guide provided by Pinecone, which covers the fundamentals of CNNs, popular architectures, and practical examples using the CIFAR-10 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Architectures](#architectures)
- [Data Preprocessing](#data-preprocessing)
- [Model Construction](#model-construction)
- [Training](#training)
- [Inference](#inference)
- [References](#references)

## Introduction
Convolutional Neural Networks (CNNs) have been the undisputed champions of Computer Vision (CV) for almost a decade. Their widespread adoption kickstarted the world of deep learning, enabling automatic feature extraction for a vast number of datasets and use cases.

## Architectures
This repository includes implementations of several popular CNN architectures:
- **LeNet**: The earliest example of a deep CNN, developed in 1998 by Yann LeCun et al.
- **AlexNet**: The catalyst for the birth of deep learning, winning the ImageNet ILSVRC challenge in 2012.
- **VGGNet**: Introduced in 2014, characterized by its depth with 16 or 19 layers.
- **ResNet**: The new champion of CV in 2015, with variants containing 34 or more layers.

## Data Preprocessing
The CIFAR-10 dataset is used for training and validation. The images are resized to 32x32 pixels and normalized using the mean and standard deviation values specific to the dataset.

## Model Construction
The `ConvNeuralNet` class defines the CNN architecture, including convolutional layers, activation functions, pooling layers, and fully connected layers.

## Training
The model is trained for 50 epochs using the CrossEntropyLoss function and the SGD optimizer. The training loop includes forward propagation, backward propagation, and optimization steps.

## Inference
The trained model is used for image classification on the CIFAR-10 test set. The predictions are compared with the actual labels to evaluate the model's performance.

## References
- [1] A. Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks (2012), NeurIPS
- [2] J. Brownlee, How Do Convolutional Layers Work in Deep Learning Neural Networks? (2019), Deep Learning for Computer Vision
- [3] J. Brownlee, A Gentle Introduction to Pooling Layers for Convolutional Neural Networks (2019), Deep Learning for Computer Vision
- [4] Y. LeCun, et. al., Gradient-Based Learning Applied to Document Recognition (1998), Proc. of the IEEE
- [5] K. Simonyan et al., Very Deep Convolutional Networks for Large-Scale Image Recognition (2014), CVPR
- [6] K. He et al., Deep Residual Learning for Image Recognition (2015), CVPR

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
