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
import sys
import os
import math
import time
import itertools 
import pdb
import random

from sklearn import preprocessing
import numpy as np 
import pickle
import scipy.optimize as optimize

import mvpa2.suite as pa
###########################
thispath = os.path.dirname(os.path.abspath(__file__))
config_relative_path="/../"
sys.path.append(thispath + config_relative_path)

import config
#import GenSAData as SA
###########################


class design_optimizer():
    """
        Class that constructs an optimization problem based on surrogate models, input parameters,
        and weights to control the optimization of output parameters. 
    """
    def __init__(self, **kwargs):
        """
            Constructor for an optimizer object with a cost function based on a set of surrogate
            models and output parameter weights.

            kwargs:
                models: 
                    list of sklearn SVR objects. Optional.
                    List of trained support-vector-regression (SVR) objects representing surrogate models.

                model_files: 
                    list of pickle files. Optional.
                    A list of pickle files containing surrogate models. Surrogate models are loaded from this file
                    into a list used to make predictions on input sample data.


                    Overrides 'models' parameter if set.
                
                scalers: 
                    list of sklearn MinMaxScaler objects. Optional.
                    Data scaler objects used to scale the surrogate model outputs to their proper ranges. These
                    scalers should be generated simultaneously with the surrogate models.

                datascaler_files: 
                    list of pickle files. Optional.
                    A list of pickle files containing scikit-learn MinMaxScaler objects used to inverse_transform
                    the surrogate model predictions back to their normal ranges.

                    Overrides 'scalers' parameter if set.

                output_weights: 
                    list of signed floats. Must be length N.
                    List of penalty/reward weights that drive optimization.

                target_goals: 
                    list of floats. Must be length N.
                    The goal values for each model.
    
                samples_file: 
                    pickle file
                    An array with samples arranged row-wise and features arranged column-wise

                optimize_title: 
                    string
                    The name to use for this optimizer object. Used to title plots, results, output files...

                target_names: 
                    list of strings.
                    A list of the target (surrogate model output) names. Each string in the list corresponds
                    to the outputs of the surrogate model with the same index in the list of surrogate models.
                    loaded from file.

                costf_params: 
                    list of strings
                    A list of the names of features (input parameters) or targets (output parameters) that
                    will be used in cost function evaluation.

                target_weights: 
                    list of floats
                    A list of weight values to use for cost function evaluation. Each weight in the list corresponds
                    to the weight value for the parameter with the same index in costf_parameters (see above)

                som_iter: integer. Optional. Default 200
                    The number of training iterations to use for training Self-Organizing Maps (SOMs).

                N_sample_skip: 
                    integer. Optional. Default 1
                    Downsample the input samples further by setting this to > 1. Sets how many samples to skip
                    over when generating the SOMs or optimizing. If N_sample_skip = 100, the number of samples used
                    will be the number of input samples loaded from samples_file divided by 100.

                output_dir: string. 
                    Optional. Default './optimize_outputs'
                    Directory in which all output data products and plots will be written.

                param_bounds_file: 
                    file. Optional. Default config.param_file

                som_dimensions:
                    integer. Optional. 
                    Default is the sqrt of the number of samples being mapped.
                    The square dimension size of the som map.

                debug:
                    Boolean. Default False.

        """ 
        # set class attributes with the kwargs
        self.samples_file = kwargs.get('samples_file', None)
        self.optimize_title = kwargs.get('optimize_title', config.optimize_title)
        self.model_files = kwargs.get('model_files', config.model_files)
        self.inscaler_file = kwargs.get('inscaler_file', config.inscaler_file)
        self.outscaler_files = kwargs.get('datascaler_files', config.datascaler_files)
        self.features = kwargs.get('feature_names', config.features)
        self.targets = kwargs.get('target_names', config.target_names)
        self.cost_function = kwargs.get('cost_function', config.cost_function)
        self.som_iter = kwargs.get('som_iter', config.som_iter)
        self.N_sample_skip = kwargs.get('N_sample_skip', config.N_sample_skip)
        self.outputdir = kwargs.get('output_dir', './optimize_outputs')
        self.param_bounds_file = kwargs.get('param_bounds_file', config.param_file)
        self.debug = kwargs.get('debug', config.DEBUG)
        self.som_dimensions = kwargs.get('som_dimensions', config.som_dimensions)
        self.som_learning_rate = kwargs.get('som_learning_rate', config.som_learning_rate)

        # setup attributes loaded from file
        self.models = self._load_models()
        self.inscaler = self._load_inscaler()
        self.outscalers = self._load_outscalers()
        if self.samples_file is not None:
            self.samples = self.load_samples(self.samples_file)

        # initialize attributes to be set later
        self.som_plot = None 

        # stuff for versioning results (not yet implemented)
        self.version_list = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omnicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
        self.N_version = 0

        if self.debug == True:
            print "Number of models: ", np.shape(self.models)
            print "Number of weights: ", len(self.cost_function.values())
            print "Number of features: ", len(self.features)
            print "Number of targets: ", len(self.targets)

    def _load_models(self):
        """
            Loads a list of saved surrogate models from file.
        """
        surrogate_models = []
        for fname in self.model_files:
            with open("../SurrogateModeling/SM_outputs/%s" %fname, 'r') as pf:
                temp_models = pickle.load(pf)
            for model in temp_models:
                surrogate_models.append(model)
        print "surrogate models loaded: ", len(surrogate_models)

        return surrogate_models
    def _load_inscaler(self):
        with open("../SurrogateModeling/SM_outputs/%s" %self.inscaler_file, 'r') as pf:
            scaler = pickle.load(pf)

        print "Loading inscaler"
        return scaler

    def _load_outscalers(self):
        datascalers = []
        for fname in self.outscaler_files:
            with open("../SurrogateModeling/SM_outputs/%s" %fname, 'r') as pf:
                temp_scalers = pickle.load(pf)

            for scaler in temp_scalers:
                datascalers.append(scaler)
        print "outscalers loaded: ", len(datascalers)
        return datascalers

    def load_samples(self, sample_file):
        # load the sampling data from file
        with open(sample_file, 'r') as pf:
            samples = pickle.load(pf)
            print "samples shape: ", np.shape(samples)
        return samples

    def costf(self, input_sample):
        """
            Evaluates the optimizer's cost function for a set of given inputs. Assumes the input_sample
            is scaled between 0.0 and 1.0.

            Returns:
                cost: The cost value of the models for the given sample

                model_outputs: A list of the predicted target values of each model for this sample
        """
        model_outputs = []
        cost = 0

        for param in self.cost_function:
            weight = self.cost_function[param]
            if param in self.targets:
                model_index = self.targets.index(param)
                model = self.models[model_index]
                model_output = model.predict(input_sample)

                cost += weight * model_output
                model_outputs = np.append(model_outputs, model_output)

            elif param in self.features:
                feature_index = self.features.index(param)
                cost += weight * input_sample[feature_index]

        return cost, model_outputs

    def optimization_costf(self, input_sample):
        """
            Evaluates the optimizer's cost function for a set of given inputs. Assumes the input_sample
            is scaled between 0.0 and 1.0. Evaluates identically to costf, but returns just the cost 
            function evaluation and not the individual model outputs.

            Returns:
                cost: The cost value of the models for the given sample
        """
        cost = 0
        for param in self.cost_function:
            weight = self.cost_function[param]
            if param in self.targets:
                model_index = self.targets.index(param)
                model = self.models[model_index]
                model_output = model.predict(input_sample)

                cost += weight * model_output

            elif param in self.features:
                feature_index = self.features.index(param)
                cost += weight * input_sample[feature_index]
        return cost

    def make_SOM(self):
        """
            Generates a self-organizing map (SOM) from
        """
        pass

    def evaluate_design_space(self, samples = None, scale_outputs = True):
        """
            Evaluates the optimizer's cost function across an array of sample inputs
            that define a design space.

            First, the optimizer's cost function is evaluated across the entire design space as
            defined by the input samples for all of its surrogate models. The cost function evaluations
            are then mapped using a Self-Organizing Map (SOM) to visualize the design space.

            Args:
                samples: 2D array
                    Samples arranged row-wise. Input parameters arranged by column.

                scale_outputs: Boolean. Default True.
                    Scales the predictions from the models back to their normal ranges if True.
            Returns:

        """
        if samples is None and self.samples is None:
            print "Error. No samples to evaluate. Need to load samples from file using load_samples, or pass an array of samples in with the kwarg 'samples'."
            return
        elif samples is None and self.samples is not None:
            samples = self.samples

        # downsample the samples for faster debugging.
        if self.N_sample_skip > 1:
            new_samples = []
            for i in range(0,len(samples), self.N_sample_skip):
                new_samples.append(samples[i])
            samples = new_samples

        # scale the inputs to normalized ranges to be used in the surrogate models
        scaled_samples = self.inscaler.transform(samples)

        # Evaluate the cost function across the entire sample space
        print "Number of samples to evaluate: %d" %(len(scaled_samples))
        print "Evaluating the design space..."
        scores = []
        model_outputs = []
        start = time.time()
        for i, sample in enumerate(scaled_samples):
            sample_score, model_evals = self.costf(sample)
            scores.append(sample_score)
            model_outputs.append(model_evals)
        scores = np.array(scores)
        model_outputs = np.array(model_outputs)

        # the outputs from the surrogate models we get from costf() are normalized. We need to scale them to their
        # real ranges using the datascalers attribute, which is a list of MinMaxScalers for each target parameter
        for i, column in enumerate(model_outputs.T):
            model_outputs[:,i] = self.outscalers[i].inverse_transform(column)

        print "Evaluating the design space... Complete. Elapsed time: %f" %(time.time() - start)

        print "samples shape: ", np.shape(scaled_samples)
        print "model_outputs shape: ", np.shape(model_outputs)
        print "scores shape: ", np.shape(scores)
        # combine the inputs, model outputs, and fitness scores for each sample into a single array
        training_data = np.hstack((samples, model_outputs, np.reshape(scores, (len(scores), 1))))

        # create a SimpleSOMMapper object to generate SOMs
        if self.som_dimensions is None:
            self.som_dimensions =  math.floor(math.sqrt(len(samples)))
        print "\nSOM dimensions: %dx%d" %(self.som_dimensions, self.som_dimensions)
        print "SOM training iterations : %d" %self.som_iter

        som = pa.SimpleSOMMapper((self.som_dimensions, self.som_dimensions), self.som_iter, learning_rate = self.som_learning_rate)
        print "Training the SOM..."
        train_start = time.time()
        som.train(training_data)
        train_stop = time.time()
        train_elapsed = train_stop - train_start
        print "Training the SOM... Complete. Time elapsed: %f" %(train_elapsed)

        print "K shape: ", np.shape(som.K)

        # check if the output directory for this SOM generation run exists. If not, create it.
        som_output_dir = "%s/%s" %(self.outputdir, self.optimize_title)
        if not os.path.exists(som_output_dir):
            os.makedirs(som_output_dir)

        import matplotlib.pyplot as plt
        title_list = self.features
        for target in self.targets:
            title_list.append(target)
        title_list.append('Cost_Function')
        #print "SOM plot titles:\n", title_list

        for i, som_plot_title in enumerate(title_list):
            print "Mapping %s" %som_plot_title
            img = plt.imshow(som.K[:,:,i], origin='lower')
            ax=plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            mapped = som(training_data)
            #print "Map shape: ", np.shape(mapped)
            #for i, m in enumerate(mapped):
            #    plt.text(m[1], m[0], self.features[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
            plt.title('%s'%(som_plot_title))

            plt.colorbar(orientation='vertical')
            plt.draw()
            plt.savefig("%s/%s_%s.png" %(som_output_dir, self.optimize_title ,som_plot_title))
            plt.close()

        # save the configuration parameters that were used for this generation to a text file
        with open("%s/%s_parameters.cfg" %(som_output_dir, self.optimize_title), 'w') as f:
            f.write("Generation Title: %s\n" %self.optimize_title)
            f.write("model files: ")
            for model in self.model_files:
                f.write('\t' + model)
            f.write("\nscaler files: ")
            for scaler in self.outscaler_files:
                f.write('\t' + scaler)
            f.write("\ntarget names:")
            for name in self.targets:
                f.write('\n\t' + name)
            f.write("\ntarget weights: ")
            for weight in self.cost_function.values():
                f.write("%1.1f" %weight)
            f.write('\n')

            f.write("\nSOM Training Parameters:\n")
            f.write("\tdimensions: %dx%d\n" %(self.som_dimensions, self.som_dimensions))
            f.write("\ttraining iterations: %d\n" %self.som_iter)
            f.write("\tn_samples: %d\n" %len(scaled_samples))
            f.write("\ttraining duration: %f\n" %train_elapsed)

    def load_param_bounds(self):
        """
            Load the lower and upper bounds (constraints) for the input parameters from the parameter
            bounds file set by param_file in the root config.
        """
        param_bounds = []
        with open(self.param_bounds_file, 'r') as param_file:
            for line in param_file:
                param_bounds.append(line.rstrip('\n').split('\t'))
        return param_bounds

    def optimize_design(self, costf = self.optimization_costf, samples = None, maxiter = None, method = 'gradient'):
        """
            Optimizes the design defined by the surrogate models across the design space defined
            by the samples using nonlinear optimization method.

            Any arbitrary cost function to optimize can be passed in with costf, so long as it's callable
            and accepts a list of floats.
            
            A constrained optimization problem is created using the constraints in the file param_file
            set in the root config. These are the constraints that were used to generate the samples
            which form the design space which is to be optimized using the cost function defined in costf
            above.

            Use the 'method' argument to select which optimization algorithm is used. 'gradient' selects
            a constrained gradient-based optimization, while 'diffevo' selects a constrained 
            differential evolution algorithm.
        """
        # TODO - need to scale parameter bound ranges and inverse transform the optimized result
        # load the input parameter bounds from file into a list of tuples and normalize them
        
        
        print "After scaling: ", bounds

        if method == 'gradient':
            bounds = self.load_param_bounds()
            print "Normalizing parameter constraints"
            bounds = [[int(param[1]), int(param[2])] for param in bounds]
            print "Before scaling: ", bounds
            bounds = self.inscaler.transform(bounds)    


            # set the initial guess to be a random sample
            x0 = np.array(samples[random.randint(0,len(samples))])
            x0_scaled = self.inscaler.transform(x0)
            #opt_result = optimize.fmin(self.costf, x0 = x0, disp = True, full_output = True)
            opt_method = 'CG'
            opt_result = optimize.fmin(costf, x0 = x0_scaled, maxfun = 10000, maxiter = 10000, full_output = True, disp = True, retall = True)
            #opt_result = optimize.minimize(self.optimization_costf, x0, method = opt_method, options = {'disp' : True})
            print "\nConvergence success"
            #print "Optimization method: ", opt_method
            #print "Optimization status: ", opt_result.status
            
            opt_param = self.inscaler.inverse_transform(opt_result[0])
            print "{:<20}{:<20}{:<20}".format("Param Name", "Initial Guess", "Optimized Param")
            for param_name, param_guess, param_opt,  in itertools.izip(self.features, x0, opt_param):
                print "{:<20}{:<20.2f}{:<20.2f}".format(param_name, param_guess, param_opt)
                #print "%s\t\t%1.2f\t\t\t%1.2f" %(param_name, param_guess, param_opt)
            print "Costf value at minimum: ", opt_result[1]
            #print "Termination message: ", opt_result.message
            print "Iterations: ", opt_result[2]
            #print "allvecs:"
            #print opt_result[5]

        elif method == 'diffevo':
            bounds = self.load_param_bounds()
            xmin=[]
            xmax=[]
            for b in range(len(bounds)):
                xmin.append(bounds[b][1])
                xmax.append(bounds[b][2])

            param_ranges=zip(self.inscaler.transform(np.array(xmin)),self.inscaler.transform(np.array(xmax)))
            # Run the optimization search
            opt_result=optimize.differential_evolution(self.optimization_costf,param_ranges,popsize=100)
            opt_param = self.inscaler.inverse_transform(opt_result['x'])
            print "{:<20}{:<20}".format("Param Name", "Optimized Param")
            for param_name, param_opt,  in itertools.izip(self.features, opt_param):
                print "{:<20}{:<20.4f}".format(param_name, param_opt)
                #print "%s\t\t%1.2f\t\t\t%1.2f" %(param_name, param_guess, param_opt)
            print "Costf value at minimum: ", opt_result['fun']
            print str(list(opt_param))

if __name__ == "__main__":
    """
    #Example psuedo-code main loop implementation:

    # create a design_optimizer object
    samplesf = $(samples_file)

    # define a cost function to use
    cost_function = $(Cost function of the model target variables)

    dopt = design_optimizer(samples_file = samplesf,
                            optimize_title = $(data title), 
                            cost_function = cost_function,
                            N_sample_skip = 500,
                            som_iter = 100,
                            som_dimensions = 200)

    #dopt.evaluate_design_space()
    # Have the optimizer evaluate the design space defined by the samples and build an SOM
    #dopt.evaluate_design_space(samples)
    dopt.optimize_design()
    """



