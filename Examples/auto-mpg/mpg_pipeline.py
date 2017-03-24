"""
    Example pipeline script using car mpg data.

"""
import sys, os

thispath = os.path.dirname(os.path.abspath(__file__))
pysur_relative_path = "/../../"
pysur_abspath = thispath + pysur_relative_path
sys.path.append(pysur_abspath)

from pysur import SurrogateModel, anova, design_optimizer

data = 'auto-mpg.data' # car mpg datafile
# first, create some data filters for the pipeline. We'll make a surrogate modeler
# filter and a design optimization filter
#
# Our analysis pipeline will look like the following:
#
#                  (Filter #1)         (Filter #2)
#   [data] --> (surrogate modeler) --> (optimizer) --> [optimized output models]
#                     |                     |
#                     V                     V
#               [saved models]    [Self Organizing Maps]
#
modeler = SurrogateModel.SurrogateModel()
#optimizer = opt.opt()

# connect
#modeler.downconnect(optimizer)
#filter_list = [modeler, model_opt]

# now we create a pipeline instance and add our filters to it
pipeline = sm_pipe.Pipeline(name = "mpg pipeline", rootdir = '.', filters = filter_list, data_sources = [data])