#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:31:25 2017

@author: wty
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin

class LinearRegressor(BaseEstimator,RegressorMixin):  
    """A Linear Regressor for ML2017-lab-01"""
    
    def __init__(self, w=0, lamda=0, eta=0.1, max_epoch=50):
        """
        Called when initializing the classifier
        """
        self.w = w
        self.lamda = lamda
        self.eta = eta
        self.max_epoch = max_epoch
        
        self.w_history = []
        
    def __h(self,w,X):
        return X.dot(w)
    
    def h(self,X):
        return self.__h(self.w,X)
    
    def __error(self,w,X,Y):
        return self.__h(w,X) - Y
    
    def error(self,X,Y):
        return self.__error(self.w,X,Y)
    
    def L(self,X,Y):
        return self.__L(self.w,X,Y)
    
    def __L(self,w,X,Y):
        num_records,num_features  = np.shape(X)  
        lamda = self.lamda
        e = self.__error(w,X,Y)
        
        regulation_loss = 1.0/2 * lamda * w.transpose().dot(w)
        loss = 1.0/2 * 1.0/float(num_records) * e.transpose().dot(e)\
            + regulation_loss
        return loss[0][0]
        
    def g(self,X,Y):
        return self.__g(self.w,X,Y)
    
    def __g(self,w,X,Y):
        num_records,num_features  = np.shape(X)
        lamda = self.lamda
        e = self.__error(w,X,Y)
        # L2 norm
        return 1.0/num_records * X.transpose().dot(e) \
                + lamda * w    
    
    
    def fit(self, X, Y):
        """
        A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self.w_history.append(self.w)
        for epoch in range(self.max_epoch):
            self.w = self.w - self.eta * self.g(X,Y)
            self.w_history.append(self.w)
        
        return self    
    
    def __predict(self,w,X):
        return self.__h(w,X)
    
    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        return self.__predict(self.w,X)
    
    def __score(self,w,X,Y):
        return -np.sqrt(((self.__predict(w,X) - Y)**2).sum()/len(Y))
    
    def score(self, X, Y):
        # RMSE
        return self.__score(self.w,X,Y)
    
    def getLossHistory(self,X,Y):
        return [self.__L(w,X,Y) for w in self.w_history]
    
    def getScoreHistory(self,X,Y):
        return [self.__score(w,X,Y) for w in self.w_history]
