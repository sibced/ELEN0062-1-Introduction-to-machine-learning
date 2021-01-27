"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import make_data1, make_data2
from plot import plot_boundary

if __name__ == "__main__":
    pass # Make your experiments here


# Question 2.1

    datasets = [make_data1, make_data2]
    n_neighbors = [1, 5, 10, 75, 100, 150]
    
    for i, data in enumerate(datasets):
        # divide data into training sample and test sample
        X_train, y_train, X_test, y_test = data()  
        # taking 20% of test samples for simpler plots
        size_test = int(0.2*len(X_test))
        X_test, y_test = X_test[:size_test], y_test[:size_test]

        for n in n_neighbors:
            # buidling the KNeighbors classifier 
            classifier = KNeighborsClassifier(n_neighbors=n)
            classifier.fit(X_train, y_train)
            # saving figures
            fname = 'Q2_md' + str(i+1)  + '_neigh_' + str(n)
            title = 'Dataset ' + str(i+1) + ' : Nearest Neighbors Boundary with n = ' + str(n)
            plot_boundary(fname, classifier, X_test, y_test, title = title)
            

# Question 2.2    

    k = 5 # five - fold
    n_neighbors =  range(1, 150)  # test from 1 to 150 neighbors
    accuracies = np.zeros((k, len(n_neighbors))) # mat 5x149 : row for considered test sample ; column for nbr neighbors

    X_train, y_train, X_test, y_test = make_data2() # testing for second dataset
    X_train, y_train, X_test, y_test = X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist() # numpy array to list 

    n_samples = len(X_train)
    len_subsets = int(n_samples/k) # = 50

    mean_acc = np.zeros(( len(n_neighbors) )) # computed with fit and score
    true_mean_acc = np.zeros(( len(n_neighbors) )) # computed with cross_val_score

    for i in range(k):
        X_LS, y_LS = X_train.copy(), y_train.copy() # copy to get TS back
        X_TS = list()
        y_TS = list()

        for m in range(len_subsets):
            X_TS.append( X_LS.pop(i*len_subsets) )  # remove TS from dataset to only keep LS
            y_TS.append( y_LS.pop(i*len_subsets) )

        for j, n in enumerate(n_neighbors):
            # buidling the KNeighbors classifier 
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(X_LS, y_LS)
            accuracies[i][j] = neigh.score(X_TS, y_TS)

    mean_acc = np.mean(accuracies, axis = 0)
    plt.plot(n_neighbors, mean_acc)
    plt.title('Dataset 2 : Evolution of mean accuracies over the 5 folds ')
    plt.xlabel('number of neighbors')
    plt.ylabel('accuracy mean')
    plt.savefig('Q2_2_b')
    n_optimal = n_neighbors[np.argmax(mean_acc)]
    print("The maximum accuracy is %f for %d neighbors " % (mean_acc[n_optimal - 1], n_optimal))



# Question 2.3

    ls_size_list = [50, 200, 250, 500]
    ts_size = 500

    acc_1 = list() # for make_data1
    acc_2 = list() # for make_data2

    for k, ls_size in enumerate(ls_size_list):

        datasets = [make_data1(n_ts = ts_size, n_ls = ls_size), make_data2(n_ts = ts_size, n_ls = ls_size)]
        n_neighbors = range(1,int(ls_size/2)) # until half of LS
        # divide data into training sample and test sample for md1 and md2
        X_train_1, y_train_1, X_test_1, y_test_1 = datasets[0]
        X_train_2, y_train_2, X_test_2, y_test_2 = datasets[1]
        # temporary
        tmp_1 = list()
        tmp_2 = list()

        for n in n_neighbors:

            neigh1 = KNeighborsClassifier(n)
            neigh1.fit(X_train_1, y_train_1)
            tmp_1.append(neigh1.score(X_test_1, y_test_1))

            neigh2 = KNeighborsClassifier(n)
            neigh2.fit(X_train_2, y_train_2)
            tmp_2.append(neigh2.score(X_test_2, y_test_2))

        # evolution of acc for each ls_size for md1 and md2
        acc_1.append(tmp_1)
        acc_2.append(tmp_2)
        

    # plot results 
    n_opt_md1 = np.zeros(len(ls_size_list))
    n_opt_md2 = np.zeros(len(ls_size_list))

    for k, ls_size in enumerate(ls_size_list):

        n_neighbors = range(1,int(ls_size/2)) # until half of LS

        n_opt_md1[k] = n_neighbors[np.argmax(acc_1[k])]
        plt.figure(2*k+1)
        plt.plot(n_neighbors, acc_1[k])
        plt.title('Dataset 1 : Evolution of test accuracies with LS size of : ' +str(ls_size))
        plt.xlabel('number of neighbors')
        plt.ylabel('accuracy evolution ')
        plt.savefig('Q2_md1_LS_size_' + str(ls_size))

        n_opt_md2[k] = n_neighbors[np.argmax(acc_2[k])]
        plt.figure(2*k+2)
        plt.plot(n_neighbors, acc_2[k])
        plt.title(' Dataset 2 : Evolution of test accuracies with LS size of : ' +str(ls_size))
        plt.xlabel('number of neighbors')
        plt.ylabel('accuracy evolution')
        plt.savefig('Q2_md2_LS_size_' + str(ls_size))


    # 2.3. (b)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(ls_size_list))
    bar_width = 0.4
    opacity = 0.8

    plt.bar(index, n_opt_md1, bar_width, alpha=opacity, color='b', label='Dataset 1')
    plt.bar(index + bar_width, n_opt_md2, bar_width, alpha=opacity, color='g', label='Dataset 2')

    plt.xlabel('LS size')
    plt.ylabel('Optimal value of nbr neighbors')
    plt.title('Optimal value of nbr neighbors w.r.t LS size')
    plt.xticks(index + 0.5*bar_width, ('50', '200', '250', '500'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('Q2_3_b')


