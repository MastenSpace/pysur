##############################################
# TOP-LEVEL PIPELINE CONTROL
##############################################
MAKE_MODELS = True
RUN_SA = True
DEBUG = True

output_directory = "../Test_Outputs"

modeling_description = "Car fuel-efficiency modeling"
##############################################
# DATA DESCRIPTION
##############################################
data_title = 'auto-mpg'
input_data_file = 'auto-mpg.data'
features = ['displacement', 'cylinders', 'model_year'] 
targets = ['mpg', 'horsepower', 'acceleration'] 

##############################################
# SURROGATE MODEL CONFIG
##############################################
test_split = 0.0
gridsearch = True
save_models = True

C_tuple = (5.0749, 5.0749)
epsilon_tuple = (0.09177, 0.09018)

##############################################
# SURROGATE MODEL HYPERPARAMETER OPTIMIZATION
##############################################
optimize_iter = 10
parallel_jobs = 1
C_range = [1, 30]
epsilon_scale = 0.15
model_scoring = 'mean_squared_error'

##############################################
# SENSITIVITY ANALYSIS CONFIG
##############################################
param_file = "mpgParamRanges.txt"
Nsamples = 1000000
compute_second_order = False
n_cores = 4

##############################################
# DESIGN OPTIMIZATION CONFIG
##############################################
optimize_title = 'autoMPG'
model_files = ['']
target_names = ['mpg', 'horsepower','acceleration']
cost_function = {}

##############################################
# DESIGN OPTIMIZATION PARAMETERS
##############################################
som_iter = 200
N_sample_skip = 1
som_dimensions = None
som_learning_rate = 0.05
