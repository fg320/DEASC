import numpy as np

from deasc import WfModel
from deasc import WSOpt
from deasc import GPWrap
from deasc import TuningDyn_Turbines

from deasc.utils_floris import (
    floris_extract_object_dict,
    floris_param_change_object_dict,
    floris_param_change_object
)

"""
This example shows wake steering optimisation on a 3x1 wind farm of NREL 5 MW turbines.
Dynamic parameter tuning is introduced in the optimisation for the wake expansion
parameter k of the Jensen wake model. The tuning variables are the yaw angles of the two
most upstream turbines.
"""

# Initialise and set layout for wind farm model
path = "./inputs/"
input_file = "jensen.yaml"
wf_model = WfModel(input_file, path)
wf_model.set_aligned_layout(3, 1, 7, 5)

# Set kd deflection parameter
wf_model_dict = floris_extract_object_dict(wf_model)
wf_model_dict = floris_param_change_object_dict(wf_model_dict,
                                                'wake_deflection_parameters',
                                                'kd',
                                                0.3)
wf_model = floris_param_change_object(wf_model, wf_model_dict)

# Specify atmopheric conditions
ws = 8.0
wd = 270
ti = 0.05
shear = 0.0

# Wake steering optimisation inputs
yaw_initial = np.full(shape=(3), fill_value=0)
inflow = (yaw_initial, wd, ws, ti, shear)
variables = [1, 2]
var_bounds = (-25, 25)
var_initial = np.full(shape=(len(variables)), fill_value=0)

# %% Dynamic tuning object

# Parameter info
parameter_class = 'wake_velocity_parameters'
parameter_name = 'we'

# Import optimal parameter dataset and extract GP input
dataset_path = "./optimal_parameter_datasets/"
dataset_import = np.load(dataset_path+'we_3x1_2dim.npy', allow_pickle=True)
optimal_parameter_dataset = dataset_import.item()
yaw_data = []
param_data = []
for key in optimal_parameter_dataset.keys():
    # Last turbine yaw angle is fixed, not a GP dimension
    yaw_data.append(key[:-1])
    param_data.append([optimal_parameter_dataset[key]])

# Construct Gaussian Process (GP)
GP_obj = GPWrap(parameter_class=parameter_class,
                parameter_name=parameter_name,
                dimensions=2)
GP_model = GP_obj.GP_so(yaw_data, param_data, num_restarts=50, noise=0.05)

# Tuning object initialisation
tuning_dyn_obj = TuningDyn_Turbines(param_class=parameter_class,
                                    param_name=parameter_name,
                                    tuning_turbines=[1, 2],
                                    GP_model=GP_model)

# %% Optimisation with dynamic tuning

# Initialise wake steering object
wso_obj_tuning = WSOpt(wf_model=wf_model,
                       inflow=inflow,
                       variables=variables,
                       var_bounds=var_bounds,
                       var_initial=var_initial,
                       opt_method="SLSQP",
                       opt_options=None,
                       obj_function="Farm Power",
                       grouping=False,
                       tuning_dynamic=True
                       )

# Assign dynamic tuning to wake steering optimisation
wso_obj_tuning.tuning_dyn_initialize([tuning_dyn_obj])

# Optimise and print yaw angles
opt_yaw_angles_vars, opt_yaw_angles_all = wso_obj_tuning.optimize_yaw()
print('Optimal farm yaw angles with dynamic parameter tuning:')
print(opt_yaw_angles_all)

# %% Optimisation without dynamic tuning
wso_obj_notuning = WSOpt(wf_model=wf_model,
                         inflow=inflow,
                         variables=variables,
                         var_bounds=var_bounds,
                         var_initial=var_initial,
                         opt_method="SLSQP",
                         opt_options=None,
                         obj_function="Farm Power",
                         grouping=False,
                         tuning_dynamic=False
                         )
_, opt_yaw_angles_all_notuning = wso_obj_notuning.optimize_yaw()
print('Optimal farm yaw angles with no dynamic parameter tuning:')
print(opt_yaw_angles_all_notuning)
