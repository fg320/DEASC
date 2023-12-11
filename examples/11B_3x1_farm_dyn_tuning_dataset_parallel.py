import numpy as np

import multiprocessing as mp

from deasc import WfModel
from deasc import Tuning

from deasc.utils import yaw_permutations_0last
from deasc.utils_floris import (
    floris_extract_object_dict,
    floris_extract_parameter,
    floris_param_change_object_dict,
    floris_param_change_object
)

"""
This example shows how to create, in parallel, an optimal parameter dataset for Jensen
wake expansion parameter k tuned to GCH model power predictions for a 3x1 wind farm of
NREL 5 MW turbines. The training conditions are defined as the yaw permutations of the
two most upstream turbines. For each condition, parameter k is tuned on that single
condition and added to the optimal parameter dataset.
"""

# %% Parameter tuning function - Run a single optimisation for each trainign condition


def function(i, yaw, inflow, wt_pow_training_list):

    # Extract inflow
    wd, ws, ti, shear = inflow

    # Initialise trainee and set farm layout
    path = "./inputs/"
    input_file_trainee = "jensen.yaml"
    trainee = WfModel(input_file_trainee, path)
    trainee.set_aligned_layout(3, 1, 7, 5)

    # Set kd deflection parameter
    trainee_dict = floris_extract_object_dict(trainee)
    trainee_dict = floris_param_change_object_dict(trainee_dict,
                                                   'wake_deflection_parameters',
                                                   'kd',
                                                   0.3)
    trainee = floris_param_change_object(trainee, trainee_dict)

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

    # Return single dictionary for the parameter tuning optimisation
    result_dict = {tuple(yaw): k_tuned}
    return result_dict

# %% Main code


def main():

    # Print available cpu cores
    print("Number of CPUs: ", mp.cpu_count())

    # %% Higher fidelity dataset

    # Initialise trainer and set farm layout
    path = "./inputs/"
    input_file_trainer = "gch.yaml"
    trainer = WfModel(input_file_trainer, path)
    trainer.set_aligned_layout(3, 1, 7, 5)

    # Define training set
    yaw_list = yaw_permutations_0last(dimensions=2,
                                      yaw_per_sweep=7,
                                      yaw_bounds=(-25, 25),
                                      n_0last=1)
    ws = 8.0
    wd = 270
    ti = 0.05
    shear = 0

    # Produce high-fidelity power measurement for each training condition
    wt_pow_training_list = []
    for i in range(len(yaw_list)):
        _, wt_pow_training, _, _ = trainer.farm_eval(yaw=yaw_list[i],
                                                     wd=wd,
                                                     ws=ws,
                                                     ti=ti,
                                                     shear=shear)
        wt_pow_training_list.append(wt_pow_training)

    # %% Run a single optimisation for each training condition

    # Initialise dataset
    optimal_parameter_dataset = {}

    # Run single optimisations with multiple processes
    inflow = wd, ws, ti, shear
    inputs = []
    for i, yaw in enumerate(yaw_list):
        inputs.append((i, yaw, inflow, wt_pow_training_list))
    with mp.Pool() as pool:
        results_dict = pool.starmap(function, inputs)
    for result_dict in results_dict:
        optimal_parameter_dataset.update(result_dict)

    # Save optimal parameter dataset
    results_path = "./optimal_parameter_datasets/"
    np.save(results_path+'%s_3x1_2dim' % ('we'), optimal_parameter_dataset)


# Required for multiprocessing
if __name__ == "__main__":
    main()
