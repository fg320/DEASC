import numpy as np

from deasc import WfModel
from deasc import WSOpt
from deasc import Tuning
from deasc import GPWrap
from deasc import TuningDyn_Grouping
from deasc import TuningDyn_Looping_Turbine

from deasc.utils_floris import (
    floris_extract_object_dict,
    floris_extract_parameter,
    floris_param_change_object_dict,
    floris_param_change_object
)

"""
This example shows wake steering optimisation on a 5x1 wind farm of NREL 5 MW turbines.
Dynamic parameter tuning with the looping approach is implemented to refine the results
achieved with grouping. Tuning is introduced in the optimisation for the wake expansion
parameter k of the Jensen wake model. The tuning variables are the yaw angles of all wind
turbines in the farm, excluding the most downstream one.
"""

# %% Initial wake steering optimisation - Grouping approach for dynamic parameter tuning

# Initialise and set layout for wind farm model
path = "./inputs/"
input_file = "jensen.yaml"
wf_model = WfModel(input_file, path)
wf_model.set_aligned_layout(5, 1, 7, 5)

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
yaw_initial = np.full(shape=(5), fill_value=0)
inflow = (yaw_initial, wd, ws, ti, shear)
variables = [1, 2, 3, 4]
var_bounds = (-25, 25)
var_initial = np.full(shape=(len(variables)), fill_value=0)

# Dynamic tuning object

# Parameter info
parameter_class = 'wake_velocity_parameters'
parameter_name = 'we'

# Import optimal parameter dataset and extract GP input
dataset_path = "./optimal_parameter_datasets/"
dataset_import = np.load(dataset_path+'we_5x1_2dim_grouping.npy', allow_pickle=True)
optimal_parameter_dataset = dataset_import.item()
yaw_data = []
param_data = []
for key in optimal_parameter_dataset.keys():
    yaw_data.append([key[0], key[2]])  # Extract group yaw
    param_data.append([optimal_parameter_dataset[key]])

# Construct Gaussian Process (GP)
GP_obj = GPWrap(parameter_class=parameter_class,
                parameter_name=parameter_name,
                dimensions=2)
GP_model = GP_obj.GP_so(yaw_data, param_data, num_restarts=100, noise=0.05)

# Tuning object initialisation
tuning_dyn_obj = TuningDyn_Grouping(param_class=parameter_class,
                                    param_name=parameter_name,
                                    tuning_groups=[[1, 2], [3, 4]],
                                    GP_model=GP_model)

# Optimisation with dynamic tuning

# Initialise wake steering object
wso_obj_tuning = WSOpt(wf_model=wf_model,
                       inflow=inflow,
                       variables=variables,
                       var_bounds=var_bounds,
                       var_initial=var_initial,
                       opt_method="SLSQP",
                       opt_options=None,
                       obj_function="Farm Power",
                       tuning_dynamic=True
                       )

# Assign dynamic tuning to wake steering optimisation
wso_obj_tuning.tuning_dyn_initialize([tuning_dyn_obj])

# Optimise and print yaw angles
opt_yaw_angles_vars, opt_yaw_angles_all = wso_obj_tuning.optimize_yaw()
print('Optimal farm yaw angles with dynamic parameter tuning:')
print(opt_yaw_angles_all)

# Extract wind farm power without any yaw
wf_pow_noyaw = wso_obj_tuning.wf_pow_noyaw

# %% Looping refinement

yaw_initial = opt_yaw_angles_all

# Number of loops for each turbine
n_iterations = 1

# One loop for each turbine variable
for turbine in [1, 2, 3, 4]*n_iterations:

    # Wake steering optimisation inputs - single turbine
    inflow = (yaw_initial, wd, ws, ti, shear)
    variables = [turbine]
    var_initial = [yaw_initial[turbine-1]]

    # %% Looping GP dataset

    # Higher fidelity dataset

    # Initialise trainer and set farm layout
    path = "./inputs/"
    input_file_trainer = "gch.yaml"
    trainer = WfModel(input_file_trainer, path)
    trainer.set_aligned_layout(5, 1, 7, 5)

    # Define training set
    yaw_list = []
    for yaw_var in np.linspace(-25, 25, 7):
        yaw_single = yaw_initial.copy()
        yaw_single[turbine-1] = yaw_var
        yaw_list.append(yaw_single)

    # Produce high-fidelity power measurement for each training condition
    wt_pow_training_list = []
    for i in range(len(yaw_list)):
        _, wt_pow_training, _, _ = trainer.farm_eval(yaw=yaw_list[i],
                                                     wd=wd,
                                                     ws=ws,
                                                     ti=ti,
                                                     shear=shear)
        wt_pow_training_list.append(wt_pow_training)

    # Parameter tuning - Run a single optimisation for each training condition

    # Initialise dataset
    optimal_parameter_dataset = {}

    for i, yaw in enumerate(yaw_list):

        # Initialise trainee
        trainee = wf_model

        # Parameters to tune
        param_class_list = ['wake_velocity_parameters']
        param_name_list = ['we']
        param_bounds_list = [(0.0, 0.1)]

        # TURBO options
        TURBO_opt = {"n_init": 2,
                     "max_evals": 100,
                     "batch_size": 4,  # 1 = Serial
                     "verbose": True,
                     "use_ard": True,
                     "max_cholesky_size": 2000,
                     "n_training_steps": 50,
                     "min_cuda": 1024,
                     "device": "cpu",
                     "dtype": "float64"}

        # Initialise parameter tuning object
        tune_obj = Tuning(wf_model=trainee,
                          variables_class_list=param_class_list,
                          variables_names_list=param_name_list,
                          variables_bounds_list=param_bounds_list,
                          obj_func_name='RMSE',
                          opt_method='TURBO_1',
                          opt_options=TURBO_opt)

        # Specify higher-fidelity tuning condition
        tune_obj.tuning_conditions(yaw_angles_list=[yaw],
                                   wind_directions_list=[wd],
                                   wind_speeds_list=[ws],
                                   turbulence_intensities_list=[ti],
                                   wind_shear_list=[shear])

        # Specify higher-fidelity turbine power measurements
        tune_obj.tuning_data(data_power_list=[wt_pow_training_list[i]])

        # Tune parameters, extract tuned dictionary, reinitialise wf_model object
        trainee, trainee_dict_opt = tune_obj.tune_parameters()

        # Extract tuned k parameter
        k_tuned = floris_extract_parameter(trainee_dict_opt,
                                           param_class_list[0],
                                           param_name_list[0])

        # Add yaw combination and optimal parameter to dataset
        optimal_parameter_dataset[tuple(yaw)] = k_tuned

    # %% Looping wso optimisation

    # Extract GP input
    yaw_data = []
    param_data = []
    for key in optimal_parameter_dataset.keys():
        yaw_data.append([key[turbine-1]])
        param_data.append([optimal_parameter_dataset[key]])

    # Construct Gaussian Process (GP)
    GP_obj = GPWrap(parameter_class=parameter_class,
                    parameter_name=parameter_name,
                    dimensions=1)
    GP_model = GP_obj.GP_so(yaw_data, param_data, num_restarts=50, noise=0.05)

    # Tuning object initialisation
    tuning_dyn_obj = TuningDyn_Looping_Turbine(param_class=parameter_class,
                                               param_name=parameter_name,
                                               tuning_turbine=[turbine],
                                               GP_model=GP_model,
                                               wf_pow_noyaw=wf_pow_noyaw)

    # SLSQP options
    SLSQP_options = {'maxiter': 100,
                     'disp': True,
                     'iprint': 2,
                     'ftol': 1e-16,
                     'eps': 0.01}

    # Initialise wake steering object
    wso_obj_tuning = WSOpt(wf_model=wf_model,
                           inflow=inflow,
                           variables=variables,
                           var_bounds=var_bounds,
                           var_initial=var_initial,
                           opt_method="SLSQP",
                           opt_options=SLSQP_options,
                           obj_function="Farm Power",
                           tuning_dynamic=True
                           )

    # Assign dynamic tuning to wake steering optimisation
    wso_obj_tuning.tuning_dyn_initialize([tuning_dyn_obj])

    # Optimise and print yaw angles for current turbine loop
    _, opt_yaw_angles_all_looping = wso_obj_tuning.optimize_yaw()
    print('Optimal farm yaw angles for current turbine looping:')
    print(opt_yaw_angles_all_looping)

    # Update initial conditions for next turbine loop
    yaw_initial = opt_yaw_angles_all_looping

print('-------------------------------------------------')
print('Optimal farm yaw angles before turbine looping:')
print(opt_yaw_angles_all)
print('Optimal farm yaw angles after turbine looping:')
print(opt_yaw_angles_all_looping)
