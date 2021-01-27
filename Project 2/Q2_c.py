"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis

LIONE Maxime
SIBOYABASORE Cedric
"""
import numpy as np
from math import *
import matplotlib.pyplot as plt
from functions import f 

if __name__ == "__main__":

# Question 2.(c) : Bayes model and the Residual error

    # Parameters 
        N = 30000
        sigma = np.sqrt(0.1)
        # generate x_i values
        x_i = np.random.uniform(0, 2, N) 
        x_i = np.round(x_i, decimals=1)
        # generate y_i values
        eps_i = np.random.normal(0, sigma, N)
        eps_i = np.round(eps_i, decimals=1)
        y_i = np.zeros(N)
        for i, x in enumerate(x_i):
            y_i[i] =  f(x, noise = eps_i[i]) 
            y_i[i] = np.round(y_i[i], decimals=3)

    # Select samples with x = x0
        x0_values = np.array([0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
        bayes, residual = np.zeros(len(x0_values)), np.zeros(len(x0_values))
        for i, x0 in enumerate(x0_values): 
            y0_subset = list()
            for j, x in enumerate(x_i):
                if x == x0 :
                    y0_subset.append(y_i[j])
            bayes[i] = np.mean(y0_subset)
            residual[i] = np.var(y0_subset)

    # True bayes and residuals
        true_bayes, true_residual = np.zeros(len(x0_values)), np.zeros(len(x0_values))
        for i, x0 in enumerate(x0_values): 
            true_bayes[i] = f(x0) 
            true_residual[i] = sigma**2    

    # Print results 
        print("Estimated mean bayes value : " + str(np.round(np.mean(bayes), decimals = 6)) + " (theoretical value : " + str(np.round(np.mean(true_bayes), decimals = 6)) + ")")
        print("Estimated mean residual value : " + str(np.round(np.mean(residual), decimals = 6)) + " (theoretical value : " + str(np.round(np.mean(true_residual), decimals = 6)) + ")")

    # Saving figures
        plt.figure(dpi=200)
        plt.title("Bayes and Residual error estimates")
        plt.plot(x0_values, true_bayes, label="Real Bayes")
        plt.plot(x0_values, true_residual, label="Real Residual")
        plt.plot(x0_values, bayes, label="Estimated Bayes")
        plt.plot(x0_values, residual, label="Estimated Residual")
        plt.xlabel('x_i')
        plt.ylabel('y_i')
        plt.legend()
        plt.savefig('Q2_c.png')







