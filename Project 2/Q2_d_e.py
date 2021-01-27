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
from sklearn import linear_model
from functions import f 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

if __name__ == "__main__":

# Question 2.(d) : Expected error 

    # Parameters 
        N = 30000 # dataset size
        k = 1000 # nbr LS 
        LS_size = int(N/k) # LS size
        sigma = np.sqrt(0.1)
        m = np.array([0,1,2,3,4,5]) # degrees
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

        # means over all x0 (Q2.(e))
        bias_means = np.zeros(len(m))      
        variance_means = np.zeros(len(m))  
        error_means = np.zeros(len(m))     

        # specific values of x0
        x0_particular = np.array([0, 0.5, 1, 1.75])
        squared_bias_part = list()
        variance_part = list()
        error_part = list()

    # Entering main loop 
        for degree in m: 
            
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
                # polynomial fit (Ordinary least-square) 
                polynomial_features = PolynomialFeatures(degree = degree) 
                x_subset_p = polynomial_features.fit_transform(x_subset)
                model = LinearRegression()
                model.fit(x_subset_p, y_subset)

                # test the model on x0 values 
                x0_values_tmp = np.linspace(0,2,41)
                x0_values_tmp = x0_values_tmp[:, np.newaxis]
                x0_poly = polynomial_features.fit_transform(x0_values_tmp)
                tmp = model.predict(x0_poly)
                for n, x0 in enumerate(x0_values):
                    y0_predict[n][i] = tmp[n]   

            # Over all LS
            # calculate the bias, variance and error of the learning algorithm
            squared_bias_part.append(list())
            variance_part.append(list())
            error_part.append(list())

            for n, x0 in enumerate(x0_values): 
                y0_predict_means[n] = np.mean(y0_predict[n]) 
                y0_predict_vars[n] =  np.var(y0_predict[n])

                squared_bias[n] = (true_bayes[n] - y0_predict_means[n])**2
                variance[n] = y0_predict_vars[n] 
                expected_error[n] = 0.1 + squared_bias[n] + variance[n]

                if x0 in x0_particular: 
                    squared_bias_part[degree].append( squared_bias[n] )
                    variance_part[degree].append( variance[n] )
                    error_part[degree].append( expected_error[n] )
                
            # Q2.(e)
            # Over all x0 
            bias_means[degree] = np.mean(squared_bias)
            variance_means[degree] = np.mean(variance)
            error_means[degree] = np.mean(expected_error)
                
            # Save fig for each degree
            plt.figure(dpi=200)
            plt.title("Expected error of the learning algorithm (degree : " + str(degree)  + ")")
            plt.plot(x0_values, squared_bias, label ="Squared Bias")
            plt.plot(x0_values, variance, label = "Variance")
            plt.plot(x0_values, expected_error, label = "Expected Error")
            plt.xlabel('x_i')
            plt.ylabel('y_i')
            plt.legend()
            plt.savefig('Q2d_m_'+str(degree)+'.png')
            

        # Bias, variance and error for particular values of x0
        squared_bias_part = np.array(squared_bias_part)
        variance_part = np.array(variance_part)
        error_part = np.array(error_part)
        print("\nSquared Biais (row : degree / column : x0) : ")
        print(squared_bias_part)
        print("\nVariance (row : degree / column : x0) : ")
        print(variance_part)
        print("\nExpected error (row : degree / column : x0) : ")
        print(error_part)

        for i, x0 in enumerate(x0_particular):
            plt.figure(dpi=200)
            plt.title("Impact of m on squared bias, variance, error (x0 = " + str(x0)  + ")")
            plt.scatter(m, squared_bias_part[:,i], label ="Squared Bias")
            plt.scatter(m, variance_part[:,i], label = "Variance")
            plt.scatter(m, error_part[:,i],label = "Expected Error")
            plt.xlabel('degree')
            plt.ylabel('y_i')
            plt.legend()
            plt.savefig("Q2d_complexity_x0_"+str(x0)+".png")

        # Plot Q2.(e)
        plt.figure(dpi=200)
        plt.title("Mean values of squared bias, variance, error")
        plt.plot(m, bias_means, label ="Mean squared bias")
        plt.plot(m, variance_means, label = "Mean variance")
        plt.plot(m, error_means, label = "Mean expected Error")
        plt.xlabel('degree')
        plt.ylabel('y_i')
        plt.legend()
        plt.savefig("Q2e.png")
        

    





