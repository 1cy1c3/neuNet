NeuNet
============

# Description
Machine Learning is an artificial generation of knowledge from experience in the form of data. NeuNet represents an artificial neuronal network that is a mathematical replica of neurons in our brain. It learns from examples and can generalize after completion of the learning phase, whereby patterns are recognized in the learning data. Furthermore, it supports the creation of single layer or multi layer perceptrons and different activation functions like sigmoid or hyperbolic tangent.

## Prerequisites
+ [Python 3.6.4](https://www.python.org/downloads/release/python-364/) or later
+ [Matplotlib](https://matplotlib.org) to display labels
+ [scikit-learn](http://scikit-learn.org/stable/) to access the digits dataset

## Installation
At first, clone or download this project. Afterwards, go to the terminal and type `python3 setup.py install` to install this Python package. Furthermore, there are different ways to generate applications from it. For example, if you look at Mac OS X, it is enough to install *py2app* by `pip3 install py2app`. Afterwards, create your own *setup.py* via `py2applet --make-setup digits_learning.py` and build the app with `python3 setup.py py2app -A`. Moreover, type `./dist/digits_learning.app/Contents/MacOS/digits_learning` to start the app.

## Usage
Navigate to the folder `build/lib/com/runekrauss/neunet_test`. There are three test files that show how to work with the neuronal network:

+ **single\_layer\_perceptron.py**: Creates a simple single layer perceptron.
+ **multi\_layer\_perceptron.py**: Creates a multi layer perceptron with one hidden layer.
+ **digits\_learning.py**: Trains and tests the neuronal network with respect to the digits dataset. The learning process is backpropagation. Methods to increase the performance are the use of a momentum as well as batch training.

For example, type `python3 digits_learning.py` to start the gradient descent learning rule. The outputs are the result of the specific accuracy.

## More information
Generate the documentation of a file regarding the special comments with a command in your terminal, for example:

```
$ cd build/lib/com/runekrauss/neunet_test
$ pydoc3 -w digits_learning
```

Afterwards, you will get a website with helpful information about the code. Otherwise, import this module and call `__doc__` on it.