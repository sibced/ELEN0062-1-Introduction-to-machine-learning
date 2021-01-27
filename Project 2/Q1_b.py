"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis

Authors :
LIONE Maxime
SIBOYABASORE Cedric
"""

import numpy as np
from scipy.integrate import nquad

rho = 0.75

def f1(x0, x1):
	return 1/( 2*np.pi * np.sqrt(1-rho**2) ) * np.exp( -1/( 2*(1-rho**2) ) * (x0**2 + 2*rho*x0*x1 + x1**2)  )

def f2(x0, x1):
	return 1/( 2*np.pi * np.sqrt(1-rho**2) ) * np.exp( -1/( 2*(1-rho**2) ) * (x0**2 - 2*rho*x0*x1 + x1**2)  )

if __name__ == "__main__":

	#Analytic error computation

	#Integrals computation
	P_pos = nquad(f1,[ [-np.inf,0], [-np.inf, 0]])[0] + nquad(f1,[ [0, np.inf], [0, np.inf]])[0] # x0*x1 > 0
	P_neg = nquad(f2,[ [-np.inf,0], [0, np.inf]])[0] + nquad(f2,[ [0, np.inf], [-np.inf,0]])[0]  # x0*x1 < 0

	P = P_neg + P_pos
	error_emp = 1 / 2 * P
	print("generalization error = %f" % error_emp)


	#Emprical error computation

	N=3000	#number of samples
	mean = [0, 0] #mean matrix
	y_values =[-1, 1]
	y = np.random.choice(y_values,N)
	hb = np.zeros(N) #Bayes model

	for i in range(N):
		if y[i] == 1:
			rho = 0.75
		else:
			rho = -0.75
		cov = [[1 ,rho], [rho,1]] #covariance matrix
		x0_i, x1_i = np.random.multivariate_normal(mean, cov)
		if x0_i * x1_i > 0:
			hb[i] = 1
		else:
			hb[i] = -1
	
	same = list(hb == y) #transformation into list to use count
	error_an = same.count(False)/N
	print("\nAnalytic error = %f" % error_an)

