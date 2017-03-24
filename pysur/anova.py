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
from itertools import izip
import pickle
import datetime as dt
import csv
import math

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.util import read_param_file
from sklearn import preprocessing
from sklearn.svm import SVR

import numpy as np

class anova(object):
    """
        ANOVA - ANalysis Of VAriance
    """
    def __init__(self, sm, param_file, nsamples = 1e6, second_order = False, samples_file = None):
        """
            Initialize an anova instance.

            kwargs:
                sm - a trained SurrogateModel, or a filepath for a pickle file containing a trained SurrogateModel.

                param_file - File of the lower and upper bounds of the anova ranges for each target parameter in the surrogate model. 

                    Files should be tab-separated and organized with each parameter in its own row, with
                    the parameter name followed by the lower, then upper bounds.

                second_order - Boolean. Toggle the computation of second-order sensitivities.

                samples_file - String. The pickle file in which generated Saltelli samples are saved or used. If this file doesn't exist,
                    the samples will be generated and saved to the new file. All subsequent runs will use the samples in this file.


        """
        self.sm = sm
        self.param_file = param_file
        self.second_order = second_order
        self.nsamples = nsamples

    def _load_SM(self, sm):
        if type(sm) == 'str':
            # assume it's a filename and try to load the pickle file
            with open(sm, 'r') as pf:
                loaded_model = pickle.load(pf)
            return loaded_model
        else:
            # assume it's an instance of a SurrogateModel object
            return sm

    def _generate_samples(self, problemdef, savefile = None):
        """
            Generate a set of evenly-spaced Saltelli samples to use for variance analysis.
        """
        # Generate evenly-distributed samples within the defined parameter bounds using Saltelli Sampling
        if self.second_order == True:
            k = 2
        else:
            k = 1

        if savefile is not None:
            if os.path.isfile(savefile) is True:
                with open(savefile, 'w') as pf:
                    sample_list = pickle.load(savefile)
            else:
                # Generate the samples
                sample_list = saltelli.sample(problemdef, int(round(float(self.nsamples)/(k * problem['num_vars'] + 2))), calc_second_order = self.second_order)
                # Save the generated samples to a pickle file for later use
                with open(savefile, 'w') as pf:
                    pickle.dump(param_list, pf)
        else:
            # Generate the samples
                sample_list = saltelli.sample(problemdef, int(round(float(self.nsamples)/(k * problem['num_vars'] + 2))), calc_second_order = self.second_order)

        return sample_list

    def run_anova(self):
        """
            Generate Saltelli-sequence samples of model inputs and run variance-based Sobol sensitivity analysis on the models.

            Args:
                n_samples: Optional integer. Number of samples to generate for sensitivity analysis.

            Returns:
                SA_list: A list containing SAData dictionaries of the sensitivity analysis results.

        """
        # First, generate a problem definition for the saltelli sampling from the param file
        # TODO - the param file should be merged into the config file to streamline setup and interaction
        problem = read_param_file(self.param_file)

        # Generate or load the samples to be used for variance analysis
        samples = self._generate_samples(problem, self.samples_file)
        
        # Now we evaluate the surrogate models on the samples. 
        # First, we must scale the samples.


    def plot_anova(self):
        pass
