#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# In[3]:


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


# In[ ]:


class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a neural network
    """
    def __init__(self, reg=0.01, seed=0):
        self.reg = reg
        np.random.seed(seed)
        self.layers = []
    
    def addLayer(self, input_dim, output_dim, actFun_type=None):
        if actFun_type is None:
            self.layers.append(Layer(input_dim, output_dim))
        else:
            actFun = lambda o: self.actFun(o, actFun_type)
            diff_actFun = lambda o: self.diff_actFun(o, actFun_type)
            self.layers.append(Layer(input_dim, output_dim, actFun, diff_actFun))
    
    def feedforward(self, X):
        x = X
        for layer in self.layers:
            x = layer.feedforward(x)
    
    def calculate_loss(self, X, y):
        self.feedforward(X)
        output_layer = self.layers[-1]
        output_dim = output_layer.output_dim
        exp_a = np.exp(output_layer.a)
        self.probs = exp_a / np.sum(exp_a, axis=1, keepdims=True)
        data_loss = np.sum(-np.log(self.probs[range(len(X)), y]))
        # Add regulatization term to loss (optional)
        for layer in self.layers: 
            data_loss += self.reg / 2 * np.sum(np.square(layer.W))
                                 
        return (1. / len(X)) * data_loss
    
    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)
    
    def backprop(self, y):
        output_dim = self.layers[-1].output_dim
        delta = self.probs
        delta[y[:, np.newaxis], range(output_dim)] -= 1
        for layer in reversed(self.layers):
            delta = layer.backprop(delta)
            
    def update(self, step_size):
        for layer in reversed(self.layers):
            layer.update(step_size, self.reg)
    
    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            loss = self.calculate_loss(X, y)
            self.backprop(y)
            self.update(epsilon)
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, loss))


# In[ ]:


class Layer(object):

    def __init__(self, input_dim, output_dim, actFun=None, diff_actFun=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun = actFun
        self.diff_actFun = diff_actFun

        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim) # n1 x n2
        # input dim: N x n1
        self.b = np.zeros((1, output_dim))

    def feedforward(self, X):
        self.input = X
        self.z = X.dot(self.W) + self.b
        self.a = self.actFun(self.z) if self.actFun is not None else self.z

        return self.a

    def backprop(self, delta):
        self.delta = delta
        if self.diff_actFun is not None:
            self.delta *= self.diff_actFun(self.a)
        self.dW = (self.input.T).dot(delta) / len(self.input)    
        self.db = np.sum(delta, axis=0, keepdims=True) / len(self.input)

        return delta.dot(self.W.T)

    def update(self, step_size, reg=0):

        dW = self.dW + reg * self.W

        self.W -= step_size*dW
        self.b -= step_size*self.db


# In[ ]:


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
#     plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
#     plt.show()
    input_dim = 2
    output_dim = 2
    print(X.shape)
    print(y.shape)
    model = DeepNeuralNetwork(reg=0.01)
    model.addLayer(input_dim, 200, 'relu')
    model.addLayer(200, 400, 'relu')
    model.addLayer(400, 200, 'relu')
    model.addLayer(200, 2)
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()

