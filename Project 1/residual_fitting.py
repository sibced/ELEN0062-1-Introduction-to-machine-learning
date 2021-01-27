"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary
from scipy.stats import pearsonr

class residual_fitting(BaseEstimator, ClassifierMixin):

    def __init__(self):
        w = np.zeros((0))
        nb_objects = 0
        nb_attributes = 0

    def fit(self, X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        
        self.nb_objects = X.shape[0]
        self.nb_attributes = X.shape[1]
        self.w =np.zeros(self.nb_attributes + 1)

        self.w[0] = np.mean(y)
    
        residuals = np.zeros(X.shape)

        for o in range(self.nb_objects):
            residuals[o][0] = y[o] - self.w[0]

        print(pearsonr(X[:][0], residuals[:][0]))
        self.w[1] = pearsonr(X[:][0], residuals[:][0])[0]
        self.w[1] *= np.std(residuals[:][0])

        for o in range(self.nb_objects):
            residuals[o][1] = y[o] - self.w[0] - (self.w[1]*X[o][0])

        self.w[2] = pearsonr(X[:][1], residuals[:][1])[0]
        self.w[2] *= np.std(residuals[:][1])

        # ====================

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        y = np.zeros(self.nb_objects)
        n=length(w)
        for o in range(self.nb_objects):
            y[o] = self.w[0]
            for i in range(1,n):
                y[o] += self.w[i]*X[o][i-1]

        return y
        # ====================

        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.

        # ====================

        pass

if __name__ == "__main__":
    from data import make_data1, make_data2
    from plot import plot_boundary, plot_boundary_extended
    datasets = [make_data1, make_data2]

    for data in datasets:
        X_train, y_train, X_test, y_test = data()
        classifier = residual_fitting()
        classifier.fit(X_train, y_train)
        classifier.predict(X_test, y_test)

