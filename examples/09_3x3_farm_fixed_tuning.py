import numpy as np
import matplotlib.pyplot as plt

from deasc import WfModel
from deasc import Tuning
from deasc.utils_floris import (
    floris_extract_parameter,
    floris_param_change_object
)

"""
This example shows a fixed tuning of the Jensen wake expansion parameter k with GCH
model power predictions for a 3x3 wind farm of NREL 5 MW turbines.
"""

# %% Higher fidelity dataset

# Initialise trainer and set farm layout
path = "./inputs/"
input_file_trainer = "gch.yaml"
trainer = WfModel(input_file_trainer, path)
trainer.set_aligned_layout(3, 3, 7, 5)

# Define training condition/s
training_conditions = 1
ws_list = [8.0]
wd_list = [270]
ti_list = [0.05]
shear_list = [0]
yaw_list = [np.full(shape=trainer.n_turbs, fill_value=0)]

# Produce high-fidelity power measurements
wt_pow_training_list = []
for i in range(training_conditions):
    _, wt_pow_training, _, _ = trainer.farm_eval(yaw=yaw_list[i],
                                                 wd=wd_list[i],
                                                 ws=ws_list[i],
                                                 ti=ti_list[i],
                                                 shear=shear_list[i])
    wt_pow_training_list.append(wt_pow_training)

# %% Parameter tuning

# Initialise trainee and set farm layout
path = "./inputs/"
input_file_trainee = "jensen.yaml"
trainee = WfModel(input_file_trainee, path)
trainee.set_aligned_layout(3, 3, 7, 5)

# Parameters to tune
param_class_list = ['wake_velocity_parameters']
param_name_list = ['we']
param_bounds_list = [(0.0, 0.1)]

# Initialise parameter tuning object
tune_obj = Tuning(wf_model=trainee,
                  variables_class_list=param_class_list,
                  variables_names_list=param_name_list,
                  variables_bounds_list=param_bounds_list,
                  obj_func_name='RMSE',
                  opt_method='TURBO_1',
                  opt_options=None)

# Specify higher-fidelity tuning conditions
tune_obj.tuning_conditions(yaw_angles_list=yaw_list,
                           wind_directions_list=wd_list,
                           wind_speeds_list=ws_list,
                           turbulence_intensities_list=ti_list,
                           wind_shear_list=shear_list)

# Specify higher-fidelity turbine power measurements
tune_obj.tuning_data(data_power_list=wt_pow_training_list)

# Tune parameters, extract tuned dictionary, reinitialise wf_model object
trainee, trainee_dict_opt = tune_obj.tune_parameters()
trainee = floris_param_change_object(trainee, trainee_dict_opt)

# Extract and print original k parameter and tuned one
k_original = floris_extract_parameter(tune_obj.wf_model_dict_original,
                                      'wake_velocity_parameters',
                                      'we')
k_tuned = floris_extract_parameter(trainee_dict_opt,
                                   'wake_velocity_parameters',
                                   'we')
print("Original k parameter = %.4f" % (k_original))
print("Tuned k parameter = %.4f" % (k_tuned))
