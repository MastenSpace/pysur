"""
Module for running iterative sweeps of sensitivity analyses on surrogate models over a range of sample sizes
to test for confidence interval convergence.

OVERVIEW
==========
This module provides functions to run sample size convergence tests on surrogate model 
sensitivity analysis. So what does all that BS actually mean? The module GenSAData.py is
used to run sensitivity analysis on surrogate models generated and trained by surrogate_modeler.py.
In order to test the dependence on certain input parameters or combinations of input parameters,
we have to generate an array of input parameter samples across ranges of each input parameter to
the model (see the text file DB10_ParamRanges.txt for an example of how parameter ranges are specified). 
The outputs of GenSAData is a sensitivity of the model to each input parameter and,
(if second-order sensitivities are being calculated), the sensitivities of the model to every
combination of input parameters. With each sensitivity, there is an associated confidence interval,
or error margin. Generally, the lower the confidence interval, the higher the certainty of the
sensitivity value is, which is desirable for finding with parameter or combination of parameters
the model is most sensitive to.

One of the important hyperparameters we can tweak when running sensitivity analysis is the number
of samples generated. Theoretically, the more samples we generate to run SA on, the more certain
we will be of the sensitivities. The purpose of this model, then, is to run lots and lots of iterations
of GenSAData (well, as many as you specify with n_iter), collect the sensitivity confidences for 
each iteration, and plot them as a function of n_iter. We should expect to see confidence intervals
approaching zero as n_iter -> infinity, which would indicate the error margin of our 
sensitivities is decreasing with increasing sample size.

USAGE
==========
The main function of this module is run_convergence_tests(). As the name suggests, this function
does the core work of sweeping over a range of sample sizes and calls run_sensitivity_analysis()
from GenSAData for each sample size, and writes the first-order (S1) and total (ST) sensitivity 
confidences for each parameter to a text csv. 

The sample range tested on is specified by min_nsamples and max_nsamples. The number of samples
tested in that range is set by n_iter, and the sample list Nsample_list is generated using a simple
linear interpolation across the specified range.

The next most important function in this module is plot_convergence which, as the name once
again suggests, plots the results of the convergence test. This function plots an array of
confidence values across the sample space they were generated with. It expects an MxN array, where
the rows are the confidence values calculated with a certain number of samples.
M is the number of iterations run_convergence_tests() did, so basically n_iter. N is twice the 
number of input parameters to the model, plus one for the first column which contains the number
of samples for that iteration. The rest of the columns contains the confidence intervals. If there
are P input parameters, column 0 is the sample number, columns 1:P are the first order sensitivity
confidences, and columns P+1:2*P+1 are the total confidences. 

In the future, if second order sensitivity confidences are added, then there will be 2*P + P^2 +1 columns.
Columns 1:P will still be first order confidences, columns P+1 : P^2 will be second-order confidences,
and columns P^2+1:P^2+P+1 will be the total confidences.

In order to run the convergence tests and plot the results, you can setup a main() function to first
run the convergence test by calling run_convergence_test(), then reading the confidences csv that
is generated into an array that is then passed to plot_convergence().

TIPS AND TRICKS
==========
Some tricky aspects of using these functions:
- The surrogate model run_convergence_test() actuall tests is NOT set in this module. Whichever 
    model GenSAData is set to generate sensitivities for is what model run_convergence_tests() will
    run on. You can specify which file to load the surrogate model from in the project root config file.
- The name of the file run_test_convergence() saves its results to is prefixed with the string set with
    the variable "outfile" and postfixed with
"""
#!/usr/bin/env python
import time 
import sys
import os
import csv
from itertools import izip

import numpy as np 

import GenSAData as genSA 

###################################################################
# Setup the test. Sepcify a list of Nsample sizes to sweep through
min_nsamples = 100
max_nsamples = 1e6
n_iter = 20
Nsample_list = np.linspace(min_nsamples, max_nsamples, n_iter)
outfile = "testconverge_"

# get the absolute path to config file
thispath = os.path.dirname(os.path.abspath(__file__))
config_relative_path = "/../"
sys.path.append(thispath + config_relative_path)

# choose the root config to use
import config as config
###################################################################
def plot_convergence(conf_array, model_label):
    """
        Plot the first-order and total confidences over a range of different sample sizes.

        Confidence data is passed in through an array with columns for the confidence intervals
        of each feature and rows for different sample sizes. If sensitivities for 16 features
        in a problem are generated, then conf_array should have 37 columns. Column 0 is the 
        number of samples for that run of SA, the next 16 cols are the S1 confidences, and the
        last 16 are the ST confidences.

        Args:
            conf_array: Required. Array of floats with shape n_iter x 2*P+1, where P is the number of 
                        surrogate model input parameters.
            model_label: Required String. Name or description for the plot.

        Returns:
            None. Plots a figure in a new window.
    """
    # get the number of features from the number of confidence values reported
    # for each row. Since ST confs are appended to S1 confs, this will be half
    # the total length of the row, excluding the sample size number in the first column
    print np.shape(conf_array)
    n_features = (np.shape(conf_array)[1] - 1)/2

    import matplotlib.pyplot as plt 
    fig, axes = plt.subplots(1,2, sharex=True)
    samples_list = conf_array[:,0].T
    labels = load_feature_labels()
    S1_data = conf_array[:,1:n_features]
    ST_data = conf_array[:,n_features+1:-1]

    S1_ax = axes[0]
    ST_ax = axes[1]

    S1_ax.plot(samples_list, S1_data)
    S1_ax.set_title("First-Order Confidence Intervals")
    ST_ax.plot(samples_list, ST_data)
    ST_ax.set_title("Total Confidence Intervals")

    fig.suptitle(model_label)
    #plt.legend()
    plt.show()

def run_convergence_test():
    """
        Run iterative sensitivity analyses over a range of sample sizes to test for sensitivity confidence convergence.

        Args:
            None.

        Returns:
            None. Writes an array of first-order and total sensitivity confidences to a csv file.
    """
    if os.path.isfile(outfile):
        userinput = raw_input("Outfile already exists. Type 'w' to overwrite. Type 'a' to append.\n")
        if userinput == 'w':
            print "You typed 'w'"
            # open the outfile in
            f = open(outfile, 'w')
            f.close()
        elif userinput == 'a':
            pass
        else:
            print "Not a valid input. Exiting."
            sys.exit()

    # Sweep through each sample size, run the sensitivity analysis, and store
    # the first-order interaction confidences values.
    for samplesize in Nsample_list:
        print "Running sample size %d... " %samplesize
        start = time.time()
        SAData_list = genSA.run_sensitivity_analysis(samplesize)
        stop = time.time()

        # BUG - why the fuck am I sweeping over the different targets?
        for SAData, target_name in izip(SAData_list, config.targets):
            S1_confs = SAData['S1_conf']
            ST_confs = SAData['ST_conf']

            dt = stop - start
            print "Running sample size %d... Done. Time elapsed: %3.2f" %(samplesize, dt)

            # 
            #sample_confs = np.append(sample_confs, S1_confs, axis = 0)
            # save outputs to a csv
            with open("%s%s.csv" %(outfile, target_name), 'a') as f:
                writer = csv.writer(f)
                row = np.insert(S1_confs, 0, samplesize)
                row = np.append(row, ST_confs)

                writer.writerow(row)

        #print "sample_confs:\n", sample_confs

def load_feature_labels():
    """
        Utility function to get input feature labels from the root config.
    """
    sys.path.append('../')
    import config

    feature_labels = config.features
    return feature_labels

def main(file):
    """
        Main function to combine running convergence tests and plotting them.

        This function can be changed and adapted to run any combination of the above functions.
    """
    # load an existing convergence test run output file
    i = 0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if i is 0:
                conf_array = np.zeros((0,np.shape(row)[0]))
                i += 1
            conf_array = np.vstack((conf_array,row))

    print conf_array

    # plot the confidences over the iterated sample sizes
    plot_convergence(conf_array, file)
    
if __name__ == "__main__":
    return