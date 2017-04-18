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

    Classes for training and using surrogate models.
"""
import os
import sys
import pdb
import logging
import multiprocessing
import itertools
import json

import pandas as pd
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import grid_search
from sklearn import metrics
import numpy as np

# setup the relative path to the root config file
# This is a kludge. Need to figure out proper intra-package relative importing...
"""
thispath = os.path.dirname(os.path.abspath(__file__))
config_relative_path = "/../"
config_abspath = thispath + config_relative_path
sys.path.append(config_abspath)

import config
"""
import sm_pipe as Pipe

# suppress scikit-learn deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SingleModel():
    """
        A single-objective surrogate model class. Acts as a container object for attributes and data
        associated with a single surrogate model object.

        Class Attributes:

        X - 
            Y
            label
            svr_obj
            X_scaler
            Y_scaler
            X_scaled
            Y_scaled
            fitness_score
            hyperparameters
    """

    def __init__(self, X, Y, label):
        """
            Initialize the SingleModel class.
        """
        self.X = X
        self.Y = Y
        self.label = label

        # initialize attributes to be used later
        self.svr_obj = None
        self.X_scaler = None
        self.Y_scaler = None

        self.X_scaled = None
        self.Y_scaled = None

        self.crossval_predictions = []
        self.fitness_score = None
        self.hyperparameters = {'C': None, 'epsilon': None}

class SurrogateModel(Pipe.Filter):
    """
        SurrogateModel(configfile, datafile, feature_labels, target_labels, gridsearch = False, C_range = (1,30), epsilon_scale = 0.15, scoring_metric = 'mean_squared_error', parallel = 1, 1cross_validate = True, scoring_metric = 'mean_squared_error')

        Class that represents a surrogate model based on Radial Basis Function (RBF) Support Vector Regression (SVR).
        
        SurrogateModel acts as a wrapper for individual surrogate models, holding several scikit-learn SVR objects for a single dataset that
        correspond to the outputs of the overall model. For example, if a dataset has a multivariate input X but only a single
        output parameter Y, SurrogateModel.models will contain a single scitik-learn regression object used to predict the single output Y.
        If a dataset with inputs X has a multivariate output Y, then SurrogateModel.models will contain several regression objects
        that will each predict one of the outputs in Y. In this way, SurrogateModel can handle both single and multivariate outputs
        without changing the interface very much.

        Parameters:
        -----------
            configfile: The (absolute) path to a config file to use to create the SurrogateModel object.
                If a configfile is passed as an argument, all subsequent arguments will be ignored.

            datafile: The (absolute) path to and filename of the datafile to.

            label (String):
                The label for the surrogate model.

            feature_labels (List of Strings): 
                The names of the input features the model is trained on. In other words, the columns of X.

            target_labels (List of Strings):
                The names of the outputs the model predicts. That is, the columns of Y.

            gridsearch (Boolean): 
                Use scikit-learn's gridsearch tool to search for optimized hyperparameters to train the model with.

            optimize_iter (Integer): 
                The number of iterations for gridsearch to perform.

            C_range (Tuple of floats): 
                The start and end values of the range to run gridsearch over for C.

            epsilon_scale (Float): 
                The scaling factor of the range around the standard deviation of the standard deviation of the target data

            scoring_metric (String):
                A scikit-learn regression scoring parameter. 
                Choices are: 
                'mean_absolute_error'
                'mean_Squared_error'
                'median_absolute_error'
                'r2'

                See scikit-learn documentation for more information on scoring parameters.
                http://scikit-learn.org/stable/modules/model_evaluation.html

            parallel (Integer):
                The number of processes to run in parallel.
                Setting this to 1 (the default) runs a single process.

            verbosity (Integer):
                Set the verbosity of debugging outputs.
                The lower the number, the more verbose.
                Debug = 1, Info = 2, Warning = 3, Error = 4, Critical = 5

        Methods:
        --------
            train(datafile = None): Trains a surrogate model on data loaded in from datafile.
            cross_validate(): Cross-validates the trained models with 
            predict(X) : Make prediction outputs from the sample inputs X. X is an indexible array.
            

        Attributes:
        -----------
            name : The name of the performance metric being modeled.
            models: A dictionary of SingleModel objects indexed on target_labels
            fitness : The fitness score evaluated during model validation/cross-validation.
            scoring_metric : The scoring metric used to evaluate the fitness of the model.
            feature_labels : A list of the parameter names this model takes as input for predictions.
            target_labels : A list of the target (output) parameter names.
            hyperparameters (dict): The hyperparameters with which the model was trained.



    """
    def __init__(self,  
                 configfile = None,
                 datafile = None,
                 label = None,
                 feature_labels = None, 
                 target_labels = None,
                 optimize_iter = 20,
                 C_range = (1,30),
                 epsilon_scale = 0.15,
                 scoring_metric = 'mean_squared_error',
                 parallel = 1,
                 verbosity = 0):
        # initialize the filter superclass
        super(SurrogateModel, self).__init__(tag = label)

        

        # If no config file is passed, take the parameters from the constructor arguments
        if configfile is None:
            self.datafile = datafile
            self.C_range = C_range
            self.epsilon_scale = epsilon_scale
            self.optimize_iter = optimize_iter
            self.parallel = parallel
            self.verbosity = verbosity * 10 # convert integer in range 0-5 to logging level range 0-50

            # initialize class attributes
            self.label = label
            self.scoring_metric = scoring_metric
            self.feature_labels = feature_labels
            self.target_labels = target_labels
        
        # If a config file is passed, parse the file for the parameters
        else:
            self._parse_config(configfile)

        # Since an SVR object is trained for each output variable, we store these individual
        # model objects in a dictionary keyed by their variable name.
        if self.target_labels is not None:
            self.models = dict.fromkeys(self.target_labels)
        else:
            self.models = None
        self.X = None
        self.Y = None
        # scaling objects used to scale inputs and outputs
        self.X_scaler = None

        # scaled dataset
        self.X_scaled = None

        # cross-validation attributes
        self.crossval_predictions = []
        self.crossval_mean = None
        self.crossval_std = None
        self.crossval_plot = None

        # setup logging
        logging.basicConfig(level = self.verbosity)

        # if a datafile is provided in the init, load the dataset
        if self.datafile is not None:
            self._load_dataset(self.datafile)

    def _parse_config(self, config_filepath):
        """
            Parses a json config file
        """
        with open(config_filepath) as cfgson:
            cfgdat = json.load(cfgson)

        # get a list of the sections
        sections = [section for section in cfgdat]

        # Parse the section objects for specific parameter fields
        if "Pipeline" in sections:
            pipesec = cfgdat["Pipeline"]
            fields = [field for field in pipesec]

            if "debug" in fields:
                debug = pipesec["debug"]
                if debug is True:
                    self.verbosity = logging.DEBUG
                else:
                    self.verbosity = logging.INFO
            if "modeling_description" in fields:
                self.label = pipesec["modeling_description"]

        if "Data" in sections:
            datasec = cfgdat["Data"]
            fields = [field for field in datasec]

            if "input_data_file" in fields:
                self.datafile = datasec["input_data_file"]
            if "features" in fields:
                self.feature_labels = datasec["features"]
            if "targets" in fields:
                self.target_labels = datasec["targets"]
        
        if "HyperparameterOpt" in sections:
            hypoptsec = cfgdat["HyperparameterOpt"]
            fields = [field for field in hypoptsec]

            if "optimize_iter" in fields:
                self.optimize_iter = hypoptsec["optimize_iter"]
            if "parallel_jobs" in fields:
                self.parallel = hypoptsec["parallel_jobs"]
            if "C_range" in fields:
                self.C_range = hypoptsec["C_range"]
            if "epsilon_scale" in fields:
                self.epsilon_scale = hypoptsec["epsilon_scale"]
            if "model_scoring" in fields:
                self.scoring_metric = hypoptsec["model_scoring"]

        return


    def _load_dataset(self, input_file):
        """
            Loads feature input (X) and target output (Y) data from a csv dataset. X and Y is stored

            Args:
                - input_file: String. Required.
                    A fulle filepath to the file holding the dataset to be loaded. Assumed to be
                    a csv file.
        """
        logging.info("Loading dataset from %s" %self.datafile)
        try:
            # load csv data from file into a Pandas dataframe
            dataframe = pd.read_csv(input_file, delim_whitespace = True)
        except IOError as e:
            logging.error("IOError: File %s not found." %input_file)
            raise
        # drop rows (samples) with NaNs in them
        dataframe = dataframe[dataframe.isnull() == False]
        # split the dataframe into X and Y data based on the feature and target labels
        self.X = dataframe.filter(self.feature_labels)
        self.Y = dataframe.filter(self.target_labels)

        # initialize SingleModel objects stored in self.models
        for label, y_series in self.Y.iteritems():
            self.models[label] = SingleModel(X = self.X.values, Y = y_series.values, label = label)


    def _grid_report(self, grid_scores, n_top=3):
        """
            Utility function to report the best model fitness scores from hyperparameter grid search.

            Based on function from the scikit-learn example "Comparing randomized search and grid search
            for hyperparameter estimation" at http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
        """
        print("\nModel cross-validation report")
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print ("Mean validation score: %1.4f (std: %1.4f)" %(score.mean_validation_score, np.std(score.cv_validation_scores)))
            #print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  #score.mean_validation_score,
                  #np.std(score.cv_validation_scores)))
            print("Parameters: {0}\n".format(score.parameters))

    def _plot_predicted_vs_actual(self):
        """
            Produce a plot of the predicted outputs vs the actual (training) outputs for each output
        """
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(len(self.models))

        # iterate over each individual surrogate model and plot its cross-validated predictions
        # vs its training values
        for axis, (model_name, model) in itertools.izip(ax, self.models.iteritems()):

            axis.scatter(model.crossval_predictions, model.Y, marker = 'x')
            axis.plot(model.Y, model.Y, c='b', linestyle='--')
            axis.set_xlabel("Predicted Values")
            axis.set_ylabel("Actual Values")
            axis.set_title("%s" %model_name)

        plt.tight_layout()
        return fig

    def cross_validate(self, plot = True):
        """
            Cross-validate the (trained) SurrogateModel with the training data using Leave-One-Out (LOO)
            cross-validation. Returns a plot of the predicted outputs from the SurrogateModel vs the
            actual outputs from the real database.

            Args:
                None

            Returns:
                Matplotlib figure object
        """
        logging.info("Cross-validating surrogate models")
        # run Leave-One-Out (LOO) cross-validation on the training data and the trained models
        for model_name, model in self.models.iteritems():

            scores = cross_validation.cross_val_score(model.svr_obj,
                                                      model.X_scaled,
                                                      model.Y_scaled,
                                                      scoring = self.scoring_metric,
                                                      cv = len(model.Y_scaled))
            avg_score = np.mean(scores)
            score_std = np.std(scores)

            predictions = cross_validation.cross_val_predict(model.svr_obj,
                                                             model.X_scaled,
                                                             model.Y_scaled,
                                                             cv = len(model.Y_scaled))
            model.crossval_predictions = np.transpose(model.Y_scaler.inverse_transform(predictions))

        if plot:
            crossval_fig = self._plot_predicted_vs_actual()
            return crossval_fig
        else:
            return

    def train(self, datafile = None):
        """
            Train the SVR object(s) on a dataset. If a datafile is not passed to the constructor,
            it must be passed to train() as an argument.

            Passing a datafile to this function will overwrite any datafile set in the constructor.

        """
        logging.info("Training surrogate models")
        if datafile is not None:
            self.datafile = datafile
            self._load_dataset(self.datafile)

        if self.X is None:
            logging.error("Error: a datafile must already be loaded by SurrogateModel, or a datafile must be passed as an argument to train()")
            return

        # to support multi-objective learning, we must individually train the SVR objects
        # for each target variable. We do this by looping over the SVRs in self.models
        self.X_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(self.X.values)
        self.X_scaled = self.X_scaler.transform(self.X.values)


        for model_name, model in self.models.iteritems():
            logging.info("Training %s" %model_name)

            # create MinMaxScaler objects to normalize the training input and output data.
            # These will be used as the scaling objects for doing all future input/output scaling
            # for this surrogate model.
            model.X_scaler = self.X_scaler
            model.X_scaled = self.X_scaled
            model.Y_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(model.Y)

            # normalize the training data between 0 and 1 for training the SVM
            model.Y_scaled = model.Y_scaler.transform(model.Y)

            # set the search ranges for the learning hyperparameters C and epsilon
            params = {'C': np.linspace(self.C_range[0], self.C_range[1], self.optimize_iter),
                'epsilon': [np.std(model.Y_scaled) + i for i in np.linspace(-self.epsilon_scale, self.epsilon_scale, self.optimize_iter)]}

            # use gridsearch to find optimized hyperparameters of the SVR
            svr = SVR(kernel = 'rbf')
            search = grid_search.GridSearchCV(estimator = svr,
                                              param_grid = params,
                                              n_jobs = self.parallel,
                                              scoring = self.scoring_metric,
                                              cv = 3,
                                              #cv = np.shape(self.Y)[0], # sets the fold size for cross-val. cv = # of samples is essentially LOO CV.
                                              verbose = 0)

            search.fit(model.X_scaled, model.Y_scaled)

            model.svr_obj = search.best_estimator_
            model.fitness_score = search.best_score_
            model.hyperparameters = search.best_params_

            # update self.models with the trained version of SingleModel
            self.models[model_name] = model

        return self.models


    def predict(self, X, predict_list = None, norm= True):
        """
            Predict a model output based on the inputs in X.

            Args:
                X (list of floats): The input variables to the model. X must have length == len(feature_labels) and will be interpreted according to
                                    the order of the features in feature_labels.

                predict_list: List of output variable (target) labels. Optional.
                             Select which target variables to predict. If None, predict() will return
                             predictions for all the targets in a list.

                norm (Boolean): Turn normalization on or off. norm = False will assumes that the sample
                                passed in is scaled between 0 and 1.

            Returns:
                prediction (list of floats): The predicted output(s).
        """
        if type(X) is list:
            if norm is True:
                # scale the inputs
                X = self.X_scaler.transform(X)
                
            prediction = self.model.predict(X)

        return self.Y_scaler.inverse_transform(prediction)

    def run_filter(self, *args, **kwargs):
        """
            The "main" function of this filter class. This is called by a Pipeline instance to run the filtering
            sequence in this instance as part of the larger pipeline.
        """
        self.train()
        self.cross_validate()

        return self

def self_test():
    """
    # Test script to verify functionality of class.
    # Essentially re-implements the (deprecated) surrogate_modeler script.
    # To be replaced by a real test case at some point...

    print("Running self-test of SurrogateModel")
    # load sample data from file in a panda dataframe
    data_file = $(data file)
    config_file = $(config file)

    # Create a SurrogateModel object by passing a config file
    model = SurrogateModel(configfile = config_file)

    # train the model by passing it sample features and targets
    model.train()

    # cross-validate the trained model
    crossval_fig = model.cross_validate()
    crossval_fig.savefig("example_crossval_plot.png", show = True)
    """
if __name__ == "__main__":
    self_test()

