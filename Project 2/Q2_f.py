"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis

Authors : 
LIONE Maxime
SIBOYABASORE Cedric
"""
import numpy as np
from math import *
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from functions import f 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":

# Question 2.(f) : Ridge regression

    # Parameters 
        N = 30000 # dataset size
        k = 1000 # nbr LS 
        LS_size = int(N/k) # LS size
        sigma = np.sqrt(0.1)
        m = 5 # degree
        # generate x_i values
        x_i = np.random.uniform(0, 2, N) 
        x_i = np.round(x_i, decimals=4)
        # generate y_i values
        eps_i = np.random.normal(0, sigma, N)
        eps_i = np.round(eps_i, decimals=4)
        y_i = np.zeros(N) 
        for i, x in enumerate(x_i):
            y_i[i] = f(x, noise = eps_i[i]) 
            y_i[i] = np.round(y_i[i], decimals=4)

    # True bayes and residuals
        x0_values = np.linspace(0,2,41)
        true_bayes, true_residual = np.zeros(len(x0_values)), np.zeros(len(x0_values))
        for i, x0 in enumerate(x0_values): 
            true_bayes[i] = f(x0) 
            true_residual[i] = sigma**2   

    # Initialize data strucutres
        # for each x0
        y0_predict = np.zeros([len(x0_values), k])  
        y0_predict_means = np.zeros(len(x0_values)) 
        y0_predict_vars = np.zeros(len(x0_values)) 
        
        # for each x0
        squared_bias = np.zeros(len(x0_values))   
        variance = np.zeros(len(x0_values))
        expected_error = np.zeros(len(x0_values))

        # lambda values
        lambda_ = np.linspace(0,2,21)
        lambda_part = np.array([0,1,2])

        # means over all x0 (Q2.(e))
        bias_means = np.zeros(len(lambda_))      
        variance_means = np.zeros(len(lambda_))  
        error_means = np.zeros(len(lambda_))     

    # Entering main loop 
        for p, lmb in enumerate(lambda_):

            # For each LS 
            for i in range(k):
                x_subset = np.zeros(LS_size)
                y_subset = np.zeros(LS_size)
                
                for j in range(LS_size):
                    x_subset[j] = x_i[i*LS_size+j]
                    y_subset[j] = y_i[i*LS_size+j]

                # train the model 
                x_subset = x_subset[:, np.newaxis]
                y_subset = y_subset[:, np.newaxis]
                # polynomial fit (Ridge regression)
                model = make_pipeline(PolynomialFeatures(m), Ridge(alpha=lmb)) 
                model.fit(x_subset, y_subset)
                
                # test the model on x0 values 
                x0_values_tmp = np.linspace(0,2,41)
                x0_values_tmp = x0_values_tmp[:, np.newaxis]
                tmp = model.predict(x0_values_tmp)
                for n, x0 in enumerate(x0_values):
                    y0_predict[n][i] = tmp[n]   

            # Over all LS
            # calculate the bias, variance and error of the learning algorithm
            for n, x0 in enumerate(x0_values):
                y0_predict_means[n] = np.mean(y0_predict[n])
                y0_predict_vars[n] =  np.var(y0_predict[n])

                squared_bias[n] = (true_bayes[n] - y0_predict_means[n])**2
                variance[n] = y0_predict_vars[n] 
                expected_error[n] = 0.1 + squared_bias[n] +  variance[n]

            # Over all x0 (for each lambda)
            bias_means[p] = np.mean(squared_bias)
            variance_means[p] = np.mean(variance)
            error_means[p] = np.mean(expected_error)  
            
            # Plot quantities for lambda = 0, 1, 2
            if lmb in lambda_part:
                plt.figure(dpi=200)
                plt.title("Expected error of learning algorithm (Ridge regression with lambda = "+ str(lmb)+ ")")
                plt.plot(x0_values, squared_bias, label ="Squared Bias")
                plt.plot(x0_values, variance, label = "Variance")
                plt.plot(x0_values, expected_error, label = "Expected Error")
                plt.xlabel('x_i')
                plt.ylabel('y_i')
                plt.legend()
                plt.savefig('Q2f_lmb_'+str(lmb)+'.png')

        # Mean quantities w.r.t lambda
        plt.figure(dpi=200)
        plt.title("Mean quantities for ridge regression of degree 5")
        plt.plot(lambda_, bias_means, label ="Mean squared bias")
        plt.plot(lambda_, variance_means, label = "Mean variance")
        plt.plot(lambda_, error_means, label = "Mean expected Error")
        plt.xlabel('lambda')
        plt.ylabel('y_i')
        plt.legend()
        plt.savefig('Q2f_means.png')
            