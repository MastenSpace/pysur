import sys, os

thispath = os.path.dirname(os.path.abspath(__file__))
pysur_relative_path = "/../../"
pysur_abspath = thispath + pysur_relative_path
sys.path.append(pysur_abspath)

from pysur import SurrogateModel

data = 'auto-mpg.data'

modeler = SurrogateModel.SurrogateModel()

# Create a surrogate model object
configfile = "config.py"
model = SurrogateModel.SurrogateModel(configfile = configfile)
model.train()