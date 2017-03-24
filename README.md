# PySur Surrogate Modeling and Optimization Toolkit

## Overview
Pysur is an engineering design and optimization toolkit written in Python. Pysur uses surrogate modeling techniques to evaluate a design space that would normally be expensive and time-consuming to model using multiphysics simulations and high-performance computers (HPCs). Pysur is **not** a replacement for multiphysics simulation and real-world testing. It is a design tool that speeds up the iterative loop between design, simulation, testing, and evaluation.

Pysur has three main components:
* Surrogate modeling
* Analysis of Variance (ANOVA)
* Design optimization and evaluation

The goal of this repository is to provide an open-source solution for end-to-end engineering design surrogate modeling and optimization. This project is built on top of open-source libraries such as scikit-learn, scipy, numpy, and more...
## Modules
### SurrogateModel
From [Wikipedia](https://en.wikipedia.org/wiki/Surrogate_model):
> A surrogate model is an engineering method used when an outcome of interest cannot be easily directly measured, so a model of the outcome is used instead. Most engineering design problems require experiments and/or simulations to evaluate design objective and constraint functions as function of design variables.... For many real world problems, however, a single simulation can take many minutes, hours, or even days to complete. As a result, routine tasks such as design optimization, design space exploration, sensitivity analysis and what-if analysis become impossible since they require thousands or even millions of simulation evaluations.

SurrogateModel contains classes for modeling a model of a design. The main class, SurrogateModel, is trained on a dataset that includes input and output parameters. Input parameters include design parameters and feature data of a model. For example, an input parameter could be an aircraft's wingspan, or the engine displacement of a car. Output parameters include performance and target variables. Output parameters could be the lift to drag ratio of an aircraft, or the gas mileage of a car.

The SurrogateModel class trains Radial Basis Function (RBF) Support Vector Regression (SVR) objects on the supplied data set to model the relationship between the input and output parameters. While this is not the same as running CFD simulations on an aircraft model or putting the model in a windtunnel, SurrogateModel creates a lower-fidelity model of the model (a meta-model, if you will) that can be more easily and quickly evaluated than CFD simulation or wind tunnel testing.

### ANOVA
Once again from [Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_variance):
> Analysis of variance (ANOVA) is a collection of statistical models used to analyze the differences among group means and their associated procedures (such as "variation" among and between groups)... In the ANOVA setting, the observed variance in a particular variable is partitioned into components attributable to different sources of variation.

The anova module provides tools to analyze the variance of the outputs of the surrogate models and their sensitivity to variation in their inputs. This is what makes surrogate modeling powerful. Because we can easily evaluate our surrogate models, we can easily and quickly tweak the inputs to the models to see how the outputs change, and statistically characterize the effect of the inputs on the outputs.

### DesignOptimizer
The DesignOptimizer is, as the name suggests, an optimization class for the design modeled by SuroogateModel. This is the least developed of the modules, and will be the focus of future developement.

## Development Notes
PySur is effectively in a pre-v1.0 state. Initial development effort in the June-July 2016 timeframe produced a clunky but working prototype of the entire pipeline from surrogate modeling to optimization. Much of the work from July-August 2016 has focused on refactoring the surrogate modeling code to make it more flexible and adaptable to different data inputs. **The first prototype of the code still works and exists split between the *SurrogateModeling* and *SensitivityAnalysis* directories.** All future development work will consolidate the source code into the *pysur* directory as the main source code directory. The *SurrogateModeling* and *SensitivityAnalysis* directories will become depreceated in favor of *pysur* as the source code directory. The functionality of surrogate_modeler.py has been refactored into a more streamlined object-oriented form in SurrogateModel.py. Most of the sensitivity analysis and optimizer functionaility still needs to be refactored into a new object-oriented form that plays nice with the new surrogate model code.

## Todo
- Finish anova.py refactor and integrate it w/ SurrogateModel class.
- Switch configuration from using config.py to a generic text config file (i.e. using simpleconfigparser) that can be passed as an argument to a constructor/script.
- Refactor design optimization into object-oriented form that integrates with SurrogateModel.