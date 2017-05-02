import sys, os

thispath = os.path.dirname(os.path.abspath(__file__))
pysur_relative_path = "/../../"
pysur_abspath = thispath + pysur_relative_path
sys.path.append(pysur_abspath)

from pysur import SurrogateModel

data = 'auto-mpg.data'

modeler = SurrogateModel.SurrogateModel()

# Create a surrogate model object
configfile = "autompg.cfg.json"
model = SurrogateModel.SurrogateModel(configfile = configfile)
model.train()

predict_data = [150.0, 6, 71]
targets = ["mpg", "horsepower"]
predictions = model.predict(predict_data, targets)

print predictions