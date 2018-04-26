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

 ![Visual representation of non-linear XOR function](/images/XOR-SP.png)

 To solve this question a non-linear model is needed, therefore, the solution is a Multi-Layer Perceptron.

 To implement this design it can be used the high-level API of TensorFlow (`tf.estimator`) that simplifies the whole process of configure, train, evaluate the model and predict new examples. The process consists of four steps implemented in the Python file `xor/xorDNN.py`:

1. Load the CSV file with the training data to a TensorFlow `Dataset`.
2. Build the classifying Neural Network.
3. Train the model using the training data.
4. Classify new examples.

#### How to run

**Before proceed:**

- Make sure you have installed Python (I recommend to run with Python 2.7) and TensorFlow.
- If you haven't cloned the repository, do it with `git clone git@github.com:v-bonilla/machine-learning-with-tensorflow_bachelor-thesis.git`

**Steps:**

1. `cd machine-learning-with-tensorflow_bachelor-thesis/xor`
2. `python2.7 xorDNN.py`

## Implementation of a Kaggle problem

Kaggle is a web platform focused on Data Science in where, among other things, competitions and datasets are offered to the community. I decided to deal with the [problem proposed by the competition "Ghouls, Goblins, and Ghosts... Boo!"](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo). This competition consists in classify three types of enchanted creatures described by the features of the dataset.

The description of the dataset and files provided can be found [here](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo/data).

Given that it's a competition, to evaluate the model is needed to submit (to the Kaggle server) a file with the results given by the model and it will return the score or precision of the model.

To implement the neural network it can be used the high-level API of TensorFlow (`tf.estimator`) as for the XOR implementation. The process consists of four steps implemented in the Python file `ghouls-goblins-and-ghosts-boo/ggg_dnn.py`:

1. Load the CSV files and preprocess the training set with pandas.
2. Build the classifying Neural Network.
3. Train the model using the training data.
4. Classify new examples.

### Results

The difference between the architectures  used is the number of hidden layers and neurons. Every model had the same optimizer, activation function and number of train iterations. The results obtained are showed in the next table:

Models | Num of hidden layers | Num of neurons in each layer | Score (Max score: 1)
:-----: | :------: | :------: | :------:
Model 1 | 1 | [5] | 0.72400
Model 2 | 1 | [10] | 0.71833
Model 3 | 2 | [5,5] | 0.71833
Model 4 | 1 | [100] | 0.70132
Model 5 | 4 | [128,256,512,1024] | 0.68620
Model 6 | 7 | [128,256,512,1024,512,256,128] | 0.67296
Model 7 | 3 | [512,512,512] | 0.66918
Model 8 | 3 | [64,128,256] | 0.66351
Model 9 | 3 | [100,100,100] | 0.66162

The highest score is 0.724 by the model with five neurons in one layer.

However, in the leaderboard the average of the score by the community models is around 0.76. On the other hand, starting TensorBoard with the path of the model 1 directory as argument, it can be seen the chart describing the average error during the training.

![Average error during the training of the model 1](/images/GGG-average-error1.png)

In the chart, the average error is above 1 and the learning algorithm doesn't converge. Moreover, the highest score obtained is below the average of the community scores, that which motivate the next experiment.

### Experiment. Comparing optimizers and activation functions

Given that the highest score obtained is below the average, it is proposed the evaluation and comparison of models built with different optimizers and activation functions.

The experiment variables:

- The two first models with better results has been selected: model 1 and model 2, with five and ten neurons respectively in the hidden layer.
- Optimizers: Adadelta, Adagrad, Adam, Ftrl, GradientDescent, ProximalAdagrad, ProximalGradientDescent, RMSProp.
- Activation functions: ReLU, ReLU6, CReLU, ELU, SeLU, Softplus, Softsign, Sigmoid, Tanh.
- Every model has been trained during 10,000 iterations.

### Experiment. Results and conclusions

The comparison has 144 elements. The next table shows the top ten results:

Top ten models | Num of neurons in the hidden layer | Optimizer | Activation function | Score (Max score: 1)
:-----: | :------: | :------: | :------: | :------:
Model 1 | [5] | GradientDescent | SeLU | 0.73913
Model 2 | [5] | ProximalGradientDescent | Softplus | 0.73913
Model 3 | [5] | RMSProp | CReLU | 0.73534
Model 4 | [5] | GradientDescent | ELU | 0.73345
Model 5 | [5] | GradientDescent | ReLU | 0.73345
Model 6 | [5] | RMSProp | Softsign | 0.73345
Model 7 | [10] | Adam | Softplus | 0.73345
Model 8 | [10] | Adam | Softsign | 0.73345
Model 9 | [5] | Adagrad | CReLU | 0.73156
Model 10 | [5] | Adagrad | ELU | 0.73156

![Distribution of the scores](/images/experiment-results.png)

The specifications of the two best architectures are:

1. 5 neurons in the hidden layer, GradientDescent as optimizer and SeLU as activation function.
2. 5 neurons in the hidden layer, ProximalGradientDescent as optimizer and Softplus as activation function.

Once again, TensorBoard shows the charts of the average error in these models:

![Model 1. GradientDescent and SeLU.](/images/error-GradientDescent-SeLU.png)
Model 1. GradientDescent and SeLU.


![Model 2. ProximalGradientDescent and Softplus.](/images/error-ProximalGradientDescent-Softplus.png)
Model 2. ProximalGradientDescent and Softplus.

The score has increased slightly, that is an increment of 0.02%, and it's still below the average.

On the other hand, in the charts showing the average error it can be seen that the model converges between 0.4 and 0.6 in 1,000 iterations approximately.

In conclusion, the fact of changing the optimizer and the activation function has caused the increment in the learning speed and the reduction in the error, but it hasn't been enough because the increment in the score is insignificant.

To obtain a better result it's needed a more specific model developed with the low-level API of TensorFlow.

#### How to run - Predefined neural network

**Before proceed:**

- Make sure you have installed Python (I recommend to run with Python 2.7) and TensorFlow.
- If you haven't cloned the repository, do it with `git clone git@github.com:v-bonilla/machine-learning-with-tensorflow_bachelor-thesis.git`
- Change the variables "`GGG_TRAINING`" and "`GGG_TEST`" in `ghouls-goblins-and-ghosts-boo/ggg_dnn.py` with the location of the files you have downloaded from [the Kaggle competition](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo).

**To see help:** `python2.7 ggg_dnn.py -h`

**Steps:**

1. `cd machine-learning-with-tensorflow_bachelor-thesis/ghouls-goblins-and-ghosts-boo`
2. Execute with the number of neurons in each hidden layer. For example, a dnn with two hidden layers and 5 neurons in each: `python2.7 ggg_dnn.py -hl 5,5`
3. The submission file named `submission_dnn_HIDDEN_LAYERS_Adagrad_tf.nn.relu_5000.csv` will be created in `ghouls-goblins-and-ghosts-boo/`

#### How to run - Experiment

**Before proceed:**

- Make sure you have installed Python (I recommend to run with Python 2.7) and TensorFlow.
- If you haven't cloned the repository, do it with `git clone git@github.com:v-bonilla/machine-learning-with-tensorflow_bachelor-thesis.git`
- Change the variables "`GGG_TRAINING`" and "`GGG_TEST`" in `ghouls-goblins-and-ghosts-boo/ggg_dnn.py` with the location of the files you have downloaded from [the Kaggle competition](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo).

**Steps:**

1. `cd machine-learning-with-tensorflow_bachelor-thesis/ghouls-goblins-and-ghosts-boo/experiment`
2. `python2.7 ggg-comparison.py`
3. The submission files will be created in `ghouls-goblins-and-ghosts-boo/experiment/`
