import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegressionImpl:

    def __init__(self, lamda=.01, threshold=0.5,  epsilon=1e-6, learning_rate=1e-4, max_steps=1000):
        self.lamda = lamda
        self.threshold = threshold
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    def sigmoid(self, X):
        """
        :param X: data matrix (2 dimensional np.array)
    
        """
        return 1.0/(1 + np.exp(-X)) 

    def logistic_func(self, beta, X):
        """
        :param X: data matrix (2 dimensional np.array)
        :param beta: value of beta (1 dimensional np.array)
        
        """
        Z = np.dot(X, beta.T)
        
        return self.sigmoid(Z)

    def fit(self, X, Y):
        return self.gradient_descent(X, Y)

    def gradient(self, beta, X, Y):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param beta: value of beta (1 dimensional np.array)
        :return: np.array i.e. gradient according to the data
        
        """
        transpose = (self.logistic_func(beta, X) - Y).T
        return np.dot(transpose, X)

    def cost_func(self, X, Y, beta):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param beta: value of beta (1 dimensional np.array)
        :return: numberic value of the cost function
        
        """
        n = X.shape[0]
        d = len(beta) - 1
        regular = self.lamda / 2 * d * (np.sum(beta**2))
        log_0 = np.log(1 - self.logistic_func(beta, X))
        log_1 = np.log(self.logistic_func(beta, X))   
        cost = -((Y * log_1) + ((1 - Y) * log_0)) 
        return cost + regular 

    def gradient_descent(self, X, Y):
        """
        :param X: data matrix (2 dimensional np.array)
        :param Y: response variables (1 dimensional np.array)
        :param epsilon: threshold for a change in cost function value
        :param max_steps: maximum number of iterations before algorithm will
            terminate.
        :return: value of beta (1 dimensional np.array)
        
        """
        beta = np.zeros(X.shape[1])
        cost = np.mean(self.cost_func(X, Y, beta))
        change_cost = 1
        d = X.shape[1]
        num_iter = 1
        
        while (change_cost > self.epsilon):  
            old_cost = cost
            step_size_0 = self.learning_rate * self.gradient(beta, X, Y)[0]
            regul = (self.lamda/d * beta[1:])
            step_size_1 = self.learning_rate * (self.gradient(beta, X, Y)[1:] + regul)
            beta[0] = beta[0] - step_size_0
            beta[1:] = beta[1:] - step_size_1 
            
            cost = np.mean(self.cost_func(X, Y, beta)) 
            change_cost = abs(old_cost - cost) 
            
            if (num_iter >= self.max_steps):
                break
            num_iter += 1
        return beta  

    def plot_logistic(self, X, y, beta):
      
        x_0 = X[np.where(y == 0.0)] 
        x_1 = X[np.where(y == 1.0)]  
        plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='g', label='y = 0') 
        plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 
        
        #decision boundary 
        x1 = np.arange(-2, 3, 0.1) 
        x2 = -(beta[0] + beta[1] * x1) / beta[2] 
        plt.plot(x1, x2, c='k', label='reg line') 
    
        plt.xlabel('x1') 
        plt.ylabel('x2') 
        plt.legend() 
        plt.show()

    def predict(self, X, beta):
        p_sigmoid = self.logistic_func(X, beta)
        return pd.Series(np.where(p_sigmoid > self.threshold, 1, 0))

    def matrix_data(self, X_data):
        k = X_data.shape[1] #independent feature count
        n = len(X_data)
        data_vector = np.array(X_data).reshape((n, k))
        #add ones column for intercept
        return np.c_[np.ones(data_vector.shape[0]), data_vector]
