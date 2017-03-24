#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright 2017 Masten Space Systems, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Author: Jack Nelson <jnelson@masten.aero>
"""

import csv
import os
import sys
import logging
from time import time
from operator import itemgetter
from itertools import izip
import pickle

import numpy as np 
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import grid_search
from sklearn import metrics
from scipy import stats
import pandas as pd

"""
surrogate_modeler.py

We construct the surrogate model by performing Support Vector Regression (SVR) with a Radial Basis
Function (RBF) kernel using Scikit-learn's Support Vector Machine (SVM) library.

The inputs to the modeling algorithm serve as training data for the SVR algorithm. This includes the
geometric description of each design in the training run, that is the independent or "design" variables, 
and the corresponding CFD output metrics, the dependent variables or "performance metrics".

The algorithm trains a surrogate model on the design variables and performance metrics, cross-validates
the model, then returns the SVR object representing the surrogate model that can then be used to 
perform sensitivity analysis.

"""
######################################
# CONFIGURATION FILE SELECTION
# get the absolute path to config file
thispath = os.path.dirname(os.path.abspath(__file__))
config_relative_path = "/../"
config_abspath = thispath + config_relative_path
sys.path.append(config_abspath)

# choose the root config file to use
from config import *
# Data normalization flag. This should always be True
normalize = True
output_directory = './SM_outputs/'
######################################

logging.basicConfig(level = logging.DEBUG)

def load_data_from_csv(data_file):
    """
        Unpacks a generic csv of data and returns an array of header field names (assumed to be the
        first row of the csv) and an array of the data fields themselves.
    """
    with open(data_file) as f:
        data_header = f.readline().split(',')
    data_fields = [i.strip() for i in data_header]
    data = np.genfromtxt(os.path.join(data_file), delimiter=',')[1:]

    return data_fields, data

# Utility function to report best scores
# Gratuitously copied from the scikit-learn example "Comparing randomized search and grid search for
# hyperparameter estimation" at (http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html)
def report(grid_scores, n_top=3):
    print("\nModel cross-validation report")
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print ("Mean validation score: %1.4f (std: %1.4f)" %(score.mean_validation_score, np.std(score.cv_validation_scores)))
        #print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              #score.mean_validation_score,
              #np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def generate_optimized_surrogate(X_train, Y_train, label, C_range = [0.1, 5], epsilon_range = 0.01, grid_iter = 10, scoring = 'r2'):
    """
        Trains a surrogate model using Radial Basis Function (RBF) Support Vector Regression (SVR) 
        while cross-validating the model and searching the hyperparameter space for optimal values
        of C and epsilon. If a testing (evaluation) data set is provided, it evaluates the best model
        on the testing set.

        This function assumes/requires that the X and Y data inputs be scaled between 0.0 and 1.0.

        Returns the best surrogate model from cross-validation and (if applicable) evaluation.
    """
    # We're going to roll model training, cross-validation, and hyperparameter optimization
    # all into one function call. First, we need to set up our model to be trained (an SVR object), and
    # create a dictionary containing the range of values to search for optimal hyperparameters
    svr = SVR(kernel = 'rbf')
    params = {'C': np.linspace(C_range[0], C_range[1], grid_iter),
              'epsilon' : [np.std(Y_train) + i for i in np.linspace(-epsilon_range, epsilon_range, grid_iter)]}

    # initialize our grid search object. For all intents and purposes, this *will* be our 
    # surrogate model because, once we train it, we can make predictions on other data with it.
    # Explanation of parameters:
    #
    # param_grid: the dictionary of parameter settings to use for the gridsearch
    # n_jobs: the number of parallel jobs to run
    # scoringg: the scoring methodology to use to evaluate and compare models
    # cv: The cross-validation algorithm to use. Passing an integer K instructs the algorithm to
    #     divide the training data into K different folds and cross-validate on each. When K = N where
    #     N is the number of samples in the training data, it is essentially Leave-One-Out crossval.
    # 
    search = grid_search.GridSearchCV(estimator = svr,
                                     param_grid = params,
                                     n_jobs = parallel_jobs,
                                     scoring = scoring,
                                     cv = np.shape(Y_train)[0],
                                     verbose = 0)
    
    # run the grid search algorithm to simultaneously train and cross validate our 
    # SVR model while searching the hyper parameter spaces for the optimal parameter
    # values.
    start = time()
    print "X_train.values"
    print type(X_train.values)
    print np.shape(X_train.values)

    print "\nY_train.values"
    print type(Y_train.values)
    print np.shape(Y_train.values)

    search.fit(X_train.values, Y_train.values)
    print "GridSearchSearchCV took %1.2f seconds." %((time() - start))
    print "grid_scores_ shape: ", np.shape(search.grid_scores_)

    if DEBUG:
        # let's plot the gridsearch scores across the searched hyperparameter space.
        # first, we get the grid_scores_ array from our gridsearch object and extract
        # the number of hyperparameters we search
        scores = search.grid_scores_
        hparameters = params.keys()
        C_hparams = params['C']
        epsilon_hparams = params['epsilon']
        score_dimension = 2

        # create a new array to contain the mesh of scores
        score_map = np.empty(np.shape(scores)[0])

        for m, score in enumerate(scores):
            mean = score[1]
            score_map[m] = -mean

        score_map = np.reshape(score_map, (grid_iter, grid_iter))

        # now create the plot
        fig, ax = plt.subplots()

        if score_dimension == 2:
            heatmap = ax.pcolormesh(epsilon_hparams, C_hparams, score_map, cmap = 'viridis')
            ax.set_xlabel(hparameters[0])
            ax.set_ylabel(hparameters[1])
            ax.set_title("%s Hyperparameter score heatmap\n%s" %(label, data_name))
            fig.colorbar(heatmap)
            fig.savefig("%s%s_%s_hyperparameter_heatmap.png" %(output_directory, label, data_name))
            #plt.show()


    report(search.grid_scores_)

    best_model = search.best_estimator_

    print "Best estimator:"
    print best_model
    print "Best parameters: "
    print search.best_params_

    return best_model

def main():
    #picklef = open(config_file, 'r')
    #config_dict = pickle.load(picklef)

    print "\n========================="
    print "SURROGATE MODEL GENERATOR"
    print "========================="
    print "PARSE AND CLEAN DATA"
    print "========================="
    # load design and target data into a pandas dataframe from the input csv
    dataframe = pd.read_csv(input_data_file)

    # drop rows (samples) with NaNs in them
    dataframe = dataframe[dataframe.isnull() == False]

    # split the dataframe into design and target dataframes
    design_data = dataframe[features]
    design_labels = design_data.axes

    target_data = dataframe[targets]
    target_labels = target_data.axes

    if DEBUG:
        print "\nFeatures:\n", design_data
        print "\nTargets:\n", target_data

    print "\nParsed data shapes\n design data: ", np.shape(design_data), "\n target data: ", np.shape(target_data)
    print " #samples: %d\n #input parameters: %d" %(np.shape(design_data)[0], np.shape(design_data)[1])
    print " #output parameters: %d" %np.shape(target_data)[1]

    if DEBUG:
        print "design data:"
        print design_data
        print "target_data:"
        print target_data

    if test_split > 0.0:
        print "\n========================="
        print "SPLIT TRAIN AND TEST DATASETS"
        print "========================="
        # split the data into a training set and a testing set for validation later.
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(design_data, target_data, test_size = test_split)

        print "\nX_train, Y_train:", np.shape(X_train), np.shape(Y_train)
        print "X_test, Y_test:", np.shape(X_test), np.shape(Y_test)
        print "training sample size: %d" %np.shape(X_train)[0]
        print "testing sample size: %d" %np.shape(X_test)[0]
        if DEBUG:
            print "X_train:\n", X_train
            print "Y_train:\n", Y_train
    else:
        X_train = design_data
        Y_train = target_data
        X_test, Y_test = [], []
    # standardize the training data to mean 0 and variance 1
    if normalize is True:
        print "\n========================="
        print "DATA NORMALIZATION AND SCALING"
        print "========================="

        # initialize a StandardScaler object to calculate the means and scaling values of each design
        # parameter (that is, it calculates the means and stdevs over the columns).
        # We then use the scaler object to transform the entire input data set (except for the design ID 
        # number) to their normalized values.
        X_train_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaler.transform(X_train), columns = X_train.axes[1])
        if test_split > 0.0:
            X_test_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaler.transform(X_test), columns = X_test.axes[1])
        else:
            X_test_scaled = []

        print "\n feature min: ", X_train_scaler.data_min_
        print " feature max: ", X_train_scaler.data_max_
        print " feature range: ", X_train_scaler.data_range_
        print " feature scales: \n", X_train_scaler.scale_

        print "\nScaled training inputs:"
        print " shape: ", np.shape(X_train_scaled)
        
        if DEBUG:
            print "\n X_train_scaled:\n", X_train_scaled
            print "\nScaled testing inputs:"
            print " shape:", np.shape(X_test_scaled)
            print "\n X_test_scaled:\n", X_test_scaled

        Y_train_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(Y_train)
        Y_train_scaled = pd.DataFrame(Y_train_scaler.transform(Y_train), columns = Y_train.axes[1])
        if test_split > 0.0:
            Y_test_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(Y_test)
            Y_test_scaled = pd.DataFrame(Y_test_scaler.transform(Y_test), columns = Y_test.axes[1])
        else:
            Y_test_scaled = []

        print "\n output min: ", Y_train_scaler.data_min_
        print " output max: ", Y_train_scaler.data_max_
        print " output range: ", Y_train_scaler.data_range_
        print " output scales: \n", Y_train_scaler.scale_

        print "\nScaled training inputs:"
        print " shape: ", np.shape(Y_train_scaled)
        
        if DEBUG:
            print "\n Y_train_scaled:\n", Y_train_scaled
            print "\nScaled testing inputs:"
            print " shape:", np.shape(Y_test_scaled)
            print "\n Y_test_scaled:\n", Y_test_scaled
            #print "\nBefore scaling:"
            #print np.shape(X_train)
            #print X_train

        


        # This is just for visualizing the normalization transformations with histograms
        if DEBUG is True and 1:
            fig, axes = plt.subplots(np.shape(X_train)[1], sharex=True, sharey=True)
            for ax, label in izip(axes, X_train.axes[1]):
                ax.hist(X_train[label], bins = 7)
                ax.set_title(label)
            fig.suptitle("Distribution of design parameters before normalization")
            
            fig, axes = plt.subplots(np.shape(X_train_scaled)[1], sharex=True,sharey=True)
            print X_train_scaled.axes
            for ax, label in izip(axes, X_train_scaled.axes[1]):
                ax.hist(X_train_scaled[label], bins=7)
                ax.set_title(label)
            fig.suptitle("Distribution of design parameters after normalization")

            if len(Y_train) is not 0 and len(Y_train_scaled) is not 0:
                fig, axes = plt.subplots(np.shape(Y_train)[1], sharex=True,sharey=True)
                for ax, label in izip(axes, Y_train.axes[1]):
                    ax.hist(Y_train[label], bins=7)
                    ax.set_title(label)
                fig.suptitle("Distribution of performance parameters before normalization")

                fig, axes = plt.subplots(np.shape(Y_train_scaled)[1], sharex=True,sharey=True)
                for ax, label in izip(axes, Y_train_scaled.axes[1]):
                    ax.hist(Y_train_scaled[label], bins=7)
                    ax.set_title(label)
                fig.suptitle("Distribution of performance parameters after normalization")
            plt.show()
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    print "\n========================="
    print "SUPPORT VECTOR REGRESSION"
    print "========================="

    surrogate_models = [] # Array to hold the surrogate model objects for each output parameter
    
    # If gridsearch is True, use scikit-learn's gridsearch to systematically search for optimal
    # hyperparameter values. Else, we use hyperparameter values set by the user to construct and 
    # train surrogate models for each performance variable.
    if gridsearch:
        # construct a surrogate model for each output parameter (performance metric)
        print "My God... They're learning..."
        for n, target_parameter in enumerate(Y_train_scaled):
            print "\n------------------------"
            print target_parameter
            print "------------------------"
            if DEBUG: print Y_train_scaled[target_parameter]
            model = generate_optimized_surrogate(X_train_scaled, 
                                       Y_train_scaled[target_parameter], 
                                       label = target_parameter,
                                       C_range = C_range, 
                                       epsilon_range = epsilon_scale,
                                       grid_iter = optimize_iter,
                                       scoring = model_scoring)
            surrogate_models.append(model)
    else:
        for n, target_parameter in enumerate(Y_train_scaled):
            print "\n------------------------"
            print target_parameter
            print "------------------------"
            model = SVR(kernel='rbf', C = C_tuple[n], epsilon = epsilon_tuple[n], gamma = 'auto').fit(X_train_scaled, Y_train_scaled[target_parameter])
            surrogate_models.append(model)

    print "\nSurrogate models:\n", surrogate_models
    """
    print np.shape(surrogate_model)
    print surrogate_model
    # make predictions over the output surrogate data.
    #prediction_outputs = [model.predict(X_train_scaled) for model in surrogate_model]
    prediction_outputs = surrogate_model[1].predict(X_train_scaled)
    print np.shape(prediction_outputs)
    print prediction_outputs
    """

    # If the sampled data was split into training and testing sets, evaluate the generated models
    # on the testing data. Otherwise, compute cross-validated scores using the training data.

    # First, instantiate a list to hold our scaler (transformation) objects to transform the values
    # predicted by the models to the range of the performance metrics being modeled.
    Y_scalers = []
    for n, model in enumerate(surrogate_models):
        print "\n------------------------"
        print targets[n]
        print "------------------------"

        if test_split > 0.0:
            print "\n========================="
            print "MODEL EVALUATION"
            print "========================="
            predictions = model.predict(X_test_scaled)
            target_values = Y_test[targets[n]]
            # reverse-transform the outputs and predictions back to their original values
            Y_test_scaler = preprocessing.MinMaxScaler().fit(Y_test[targets[n]].reshape(-1,1))
            predictions = Y_test_scaler.inverse_transform(predictions.reshape(-1,1))

            #print Y_test[:,n]
            #print predictions
            #result_array = np.column_stack((Y_test[:,n].reshape(-1,1), predictions))

            print "test values, predicted values"
            print target_values, predictions
            print "model score:", metrics.mean_squared_error(target_values, predictions)
            #print "model score: ", model.score(target_values, predictions)
            print "model parameters:"
            parameters = model.get_params()
            print ' C: ', parameters['C']
            print ' epsilon: ', parameters['epsilon']
            #print ' gamma: ', parameters['gamma']

        # If a testing set was not set aside, use Leave-One-Out (LOO) cross-validation
        else:
            scaled_target_values = Y_train_scaled[targets[n]].values
            target_values = Y_train[targets[n]].values

            scores = cross_validation.cross_val_score(model, 
                                                      X_train_scaled.values, 
                                                      scaled_target_values,
                                                      scoring = 'mean_squared_error',
                                                      cv = len(Y_train_scaled))

            avg_score = np.mean(scores)
            score_std = np.std(scores)
            print "model avg score: %1.5f (+/-%1.5f)" %(-avg_score, score_std)

            predictions = cross_validation.cross_val_predict(model,
                                                             X_train_scaled.values,
                                                             scaled_target_values,
                                                             cv = len(Y_train_scaled))

            # Make a scaler and inverse transform the predictions back to their original, unscaled ranges
            Y_test_scaler = preprocessing.MinMaxScaler().fit(target_values)
            predictions = Y_test_scaler.inverse_transform(predictions)  
            Y_scalers.append(Y_test_scaler)
            print "Y_scalers[%d]: "%n, Y_scalers[n]

        # plot the predicted vs actual values
        fig, ax = plt.subplots()
        ax.scatter(predictions, target_values, marker = 'x')
        ax.plot(target_values, target_values, c='b', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Actual Values")
        ax.set_title("Predicted vs Actual Target Values: %s" %targets[n])

        fig.savefig('%s%s_%s_predicted_vs_actual.png' %(output_directory, data_title, targets[n]))     
    """
    if test_split > 0.0:
        print "\n========================="
        print "MODEL EVALUATION"
        print "========================="

        # step through each model and evaluate its performance on the testing data
        for n, model in enumerate(surrogate_models):
            print "\n------------------------"
            print targets[n]
            print "------------------------"
            predictions = model.predict(X_test_scaled)
            target_values = Y_test[targets[n]]
            # reverse-transform the outputs and predictions back to their original values
            Y_test_scaler = preprocessing.MinMaxScaler().fit(Y_test[targets[n]].reshape(-1,1))
            predictions = Y_test_scaler.inverse_transform(predictions.reshape(-1,1))

            #print Y_test[:,n]
            #print predictions
            #result_array = np.column_stack((Y_test[:,n].reshape(-1,1), predictions))

            print "test values, predicted values"
            print target_values, predictions
            print "model score:", metrics.mean_squared_error(target_values, predictions)
            #print "model score: ", model.score(target_values, predictions)
            print "model parameters:"
            parameters = model.get_params()
            print ' C: ', parameters['C']
            print ' epsilon: ', parameters['epsilon']
            #print ' gamma: ', parameters['gamma']

            # plot the predicted vs actual values
            fig, ax = plt.subplots()
            ax.scatter(predictions, target_values, marker = 'x')
            ax.plot(target_values, target_values, c='b', linestyle='--')
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.set_title("Predicted vs Actual Target Values: %s" %targets[n])

            fig.savefig('%s%s_predicted_vs_actual.png' %(output_directory, targets[n]))

    else:
        print "\n========================="
        print "MODEL CROSS-VALIDATION"
        print "========================="

        # Use cross-validation to evaluate the models created above
        for n, model in enumerate(surrogate_models):
            print "\n------------------------"
            print targets[n]
            print "------------------------"

            scaled_target_values = Y_train_scaled[targets[n]].values
            target_values = Y_train[targets[n]].values

            scores = cross_validation.cross_val_score(model, 
                                                      X_train_scaled.values, 
                                                      scaled_target_values,
                                                      scoring = 'mean_squared_error',
                                                      cv = len(Y_train_scaled))

            avg_score = np.mean(scores)
            score_std = np.std(scores)
            print "model avg score: %1.5f (+/-%1.5f)" %(-avg_score, score_std)

            predictions = cross_validation.cross_val_predict(model,
                                                             X_train_scaled.values,
                                                             scaled_target_values,
                                                             cv = len(Y_train_scaled))

            # Make a scaler and inverse transform the predictions back to their original, unscaled ranges
            Y_test_scaler = preprocessing.MinMaxScaler().fit(target_values)
            predictions = Y_test_scaler.inverse_transform(predictions)

            # plot the predicted vs actual values
            fig, ax = plt.subplots()
            ax.scatter(predictions, target_values, marker = 'x')
            ax.plot(target_values, target_values, c='b', linestyle='--')
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.set_title("Predicted vs Actual Target Values: %s" %targets[n])

            fig.savefig('%s%s_predicted_vs_actual.png' %(output_directory, targets[n]))
    """
    if save_models is True:
        model_file = data_title + "_surrogate_models.pkl"
        input_scaler_file = data_title + "_input_scalers.pkl"
        scaler_file = data_title + "_datascalers.pkl"
        models_savefile = output_directory + model_file
        input_scalers_savefile = output_directory + input_scaler_file
        scalers_savefile = output_directory + scaler_file
        #models_savefile = "%s%s_surrogate_models.pkl" %(output_directory, data_name)
        #scalers_savefile = "%s%s_datascalers.pkl" %(output_directory, data_name)

        with open(models_savefile, 'w') as f:
            pickle.dump(surrogate_models, f)

        with open(input_scalers_savefile, 'w') as f:
            pickle.dump(X_train_scaler, f)

        with open(scalers_savefile, 'w') as f:
            pickle.dump(Y_scalers, f)

    return surrogate_models, Y_scalers

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    start = time()
    main()
    print "\nSurrogate Modeling script took %2.2f seconds" %(time()-start)