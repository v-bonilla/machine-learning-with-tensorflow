# Machine Learning with TensorFlow - Bachelor Thesis

This repository contains the implementation for the Bachelor Thesis published [here](http://oa.upm.es/49683/) (in Spanish).

The content of the repository is divided in two parts: firstly, a solution to the classic problem of the XOR function, and afterwards, the implementation proposed to solve a [problem from Kaggle](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo) and an experiment comparing the optimizers and activation functions available in the high-level API of TensorFlow.

## Implementation of a classic problem: XOR function

The XOR function is an operation over two binary values, *x1* and *x2*. The function returns 1 if one of the binary values is 1. In any other case, when both values are 0 or 1, it returns 0.

*x1* | *x2* | *x1* XOR *x2*
:-----: | :------: | :------:
0 | 0 | 0
1 | 0 | 1
0 | 1 | 1
1 | 1 | 0

So XOR is a classification problem. Apparently it's a very simple problem, however, Minsky and Papert (1969) demostrated the difficulty to solve the problem by the architectures of the neural networks at that time. It is because, as shown in the image, the Perceptron can only classify linearly separable data and the XOR problem is not linear.

 ![Visual representation of non-linear XOR function](/readmeImages/XOR-SP.png)

 To solve this question a non-linear model is needed, therefore, the solution is a Multi-Layer Perceptron.

 To implement this design it can be used the high-level API of TensorFlow (`tf.estimator`) that simplifies the whole process of configure, train, evaluate the model and predict new examples. The process consists of four steps implemented in the Python file `xor/xorDNN.py`:

1. Load the CSV file with the training data to a TensorFlow `Dataset`.
2. Build the classifying Neural Network.
3. Train the model using the training data.
4. Classify new examples.

## Implementation of a Kaggle problem

Section working in progess.
