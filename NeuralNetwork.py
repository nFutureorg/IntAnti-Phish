import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict, OrderedDict
from urllib.parse import urlsplit
from tqdm import tqdm


#from sklearn import tree


def sigmoid(X):
    return 1 / (1 + np.exp(-X))
 
#print(sigmoid(np.array([1, 2, 1.1, 4])))

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_size=100, learning_rate=.1, epochs=1000, debug_print_epoch=10):
        assert hidden_layer_size >= 0
        self.hidden_layer_size_ = hidden_layer_size
        self.learning_rate_ = learning_rate
        self.epochs_ = epochs
        self.debug_print_epoch_ = debug_print_epoch
 
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)  # Makes sure the X and y play nice
 
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # In this particular case, we'll make sure the number of classes is 2
        assert n_classes == 2
 
        n_samples, n_features = X.shape
 
        self.binarizer_ = LabelBinarizer().fit(y)
        Y_binary = self.binarizer_.transform(y)
 
        # Compute the weight matrices sizes and init with small random values
 
        # Hidden Layer
        self.A1_ = np.random.randn(n_features, self.hidden_layer_size_)
        # Output Layer
        self.A2_ = np.random.randn(self.hidden_layer_size_, 1)
 
        # Training Process
        for epoch in range(self.epochs_):
            Y_hidden = X.dot(self.A1_)
            Y_output = sigmoid(Y_hidden.dot(self.A2_))
 
            error = Y_output - Y_binary
            d_A2 = error * Y_output * (1 - Y_output)
 
            hidden_error = d_A2.dot(self.A2_.T)
            d_A1 = hidden_error
 
            self.A1_ -= self.learning_rate_ * X.T.dot(d_A1)
            self.A2_ -= self.learning_rate_ * Y_hidden.T.dot(d_A2)
 
            if not epoch % self.debug_print_epoch_:
                score = self.score(X, y)
                print(f"Epoch={epoch} \t Score={score}")
 
    def predict_proba(self, X):
        """ Output probabilities for each sample"""
        # make sure X is of an accepted type
        X = check_array(X, accept_sparse='csr')  
 
        # Apply linear function at the hidden layer
        Y_hidden = X.dot(self.A1_)
 
        # Apply sigmoid at the output layer
        Y_output = sigmoid(Y_hidden.dot(self.A2_))
 
        return np.hstack((1 - Y_output, Y_output))
 
    def predict(self, X):
        """ Output only the most likely class for each sample """
        scores = self.predict_proba(X)
        indices = scores.argmax(axis=1)
        return self.binarizer_.inverse_transform(indices)


