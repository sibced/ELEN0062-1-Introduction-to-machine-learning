"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from plot import plot_boundary
from data import make_data1, make_data2


if __name__ == "__main__":

    datasets = [make_data1, make_data2]
    max_depth = [1, 2, 4, 8, None]

# Question 1.1

    for i, data in enumerate(datasets):
        # divide data into training sample and test sample
        X_train, y_train, X_test, y_test = data()  
        # taking 20% of test samples for simpler plots
        size_test = int(0.2*len(X_test))
        X_test, y_test = X_test[:size_test], y_test[:size_test]

        for depth in max_depth:
            # buidling the decision tree classifier 
            classifier = DecisionTreeClassifier(max_depth = depth)
            classifier.fit(X_train, y_train)
            # saving figures
            fname = 'Q1_md' + str(i+1)  + '_depth_' + str(depth)
            title = 'Dataset ' + str(i+1) + ' : Decision Tree Boundary with depth = ' + str(depth)
            plot_boundary(fname, classifier, X_test, y_test, title = title)
            

# Question 1.2

    nb_generations = 5 # 5 generations
    accuracies = np.empty(nb_generations)

    for i, data in enumerate(datasets): 

        for j, depth in enumerate(max_depth): 

            for g in range(nb_generations):
                
                new_dt = [make_data1(random_state = g), make_data2(random_state = g)]
                X_train, y_train, X_test, y_test = new_dt[i] # divide data into training sample and test sample 
                # buidling the decision tree classifier 
                classifier = DecisionTreeClassifier(max_depth=depth)
                classifier.fit(X_train, y_train) 

                accuracies[g] = classifier.score(X_test, y_test)

            # Mean and std for the five generations (in percent)
            mean_acc = np.mean(accuracies)*100 
            std_acc = np.std(accuracies)*100

            # Print the results
            print("For dataset " + str(i+1) + " and depth " + str(depth) + " : accuracy mean = " + str(mean_acc) + " & std = " + str(std_acc))



