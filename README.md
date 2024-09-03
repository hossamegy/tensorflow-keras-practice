# **tensorflow-keras-practiceTensorFlow Custom Layers and Models**
This repository contains a collection of custom TensorFlow layers and models, as well as basic TensorFlow operations and examples. It demonstrates how to create and use custom layers, define complex models, and perform basic tensor operations.

## Contents
* Overview
* Files
* Models
* License
* Overview

## **Overview**
This repository includes:

* Basic TensorFlow tensor operations demonstrated in a Jupyter Notebook.
* Custom TensorFlow layers including residual layers for CNNs and DNNs.
* Various models including Residual Networks (ResNet), Simple Residual Networks, Wide and Deep models, and positional encoding for Transformers.

## **Files**
### **1. tensor_operations.ipynb**

This Jupyter Notebook demonstrates various TensorFlow operations and tensor manipulations. It includes:

* **Creating Tensors:** Different ways to create tensors, including constant, variable, zeros, ones, and random tensors.
* **Basic Tensor Operations:** Examples of basic tensor operations such as addition, squaring, summation, and tensor arithmetic using both TensorFlow and NumPy.
* **Conversion:** How to convert between NumPy arrays and TensorFlow tensors.
* **Advanced Operations:** Using functions like tf.range, tf.fill, tf.eye, and tf.linalg.diag to perform more complex tensor operations.

### **2. simple_custom_layer.py**
Contains a custom dense layer with trainable parameters and activation functions. This file includes:
* SimpleCustomLayer: A custom layer that implements a dense layer with trainable weights and biases, and applies an activation function.

### **3. dnn_residual_layer.py**
Defines a custom dense residual layer for deep networks. This file includes:
* **DNNResidualLayer:** A custom layer that applies a series of dense operations with residual connections.

### **4. cnn_residual_layer.py**
Defines a custom convolutional residual layer using TensorFlow. This file contains:

* CNNResidualLayer: A custom layer that applies a series of convolutional operations with residual connections.

### **5. simple_residual_net.py**
Defines a simple residual network combining custom CNN and DNN residual layers. This file includes:

* **SimpleResidualNet:** A model that uses a combination of custom CNN and DNN residual layers, followed by a dense output layer.
### **6. resnet_model.py**
Implements a Residual Network (ResNet) model using TensorFlow. This file includes:

* **IdentityBlock:** Defines an identity block with convolutional layers and batch normalization.
* **ResNet:** Constructs a ResNet model with multiple identity blocks, global average pooling, and a final classification layer.

### **7. wide_and_deep_model.py**
Implements a Wide and Deep model that combines dense layers with multiple outputs. This file includes:

* **WideAndDeepModel:** A model with separate hidden layers and auxiliary outputs, combining dense layers with concatenation.

### **8. huber_loss.py**
Defines a custom Huber loss function for regression tasks. This file includes:

* **huber_loss:** A function to compute Huber loss, which is robust to outliers and combines mean squared error with mean absolute error.

### **9. positional_encoding.py**
Implements a positional encoding layer for Transformer models. This file includes:

* **PositionalEncoding:** A custom layer that adds positional encoding to input tensors, helping models understand the order of sequences.

## **Models**
* **ResNet Model**
    A residual network model with identity blocks:

    ResNet: Demonstrates how to build a ResNet model with convolutional layers and residual connections.

* **SimpleResidualNet Model**
A simple network with custom residual layers:

    SimpleResidualNet: Combines custom CNN and DNN residual layers with a final dense output.

* **Wide and Deep Model**
    A model combining dense layers with multiple outputs:

    WideAndDeepModel: Demonstrates a model with parallel dense layers and concatenation of outputs.


# **License**
This project is licensed under the Apache License 2.0.

