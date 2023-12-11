# Copyright 2023 Filippo Gori

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import copy
from scipy.optimize import minimize
from turbo import Turbo1
from .utils_floris import (
    floris_reinitialise_atmosphere,
    floris_calculate_turbine_power,
    floris_extract_object_dict,
    floris_extract_models_dict,
    floris_print_params,
    floris_extract_parameter,
    floris_param_change_object_dict,
    floris_param_change_object
)
from .utils import (
    norm,
    unnorm
)


class Tuning:
    """
    Parameter tuning class for a low-fidelity model, where one or more
    parameters are tuned to higher fidelity power measurements. In particular,
    the RMSE is minimised for single turbine power measurements for a single or
    the sum of multiple atmospheric conditions. The wind farm layout is assumed fixed.
    """

    def __init__(self,
                 wf_model,
                 variables_class_list,
                 variables_names_list,
                 variables_bounds_list,
                 obj_func_name='RMSE',
                 opt_method='SLSQP',
                 opt_options=None
                 ):
        """
        Args
        ----
        wf_model : WfModel object (low-fidelity model)
          single WfModel object to tune
        variables_class_list: list of strings
          list of classes of parameters to tune, one per parameter
        variables_names_list : list of strings
          list of parameter names to tune
        variables_bounds_list : list of tuples
          list of parameter bounds, upper and lower limits for each parameter
        obj_func_name: string
          objective function. Default set to "RMSE"
        opt_method: string
          optimization method. Dafault set to "SLSQP" ("TURBO_1" also available)
        opt_options: dict
          optimizer options. Default set to None
        """
        self.obj_func_dict = {'RMSE': self._tuning_rmse_function}
        self.opt_method_list = ["SLSQP", "TURBO_1"]
        self.opt_options_dict = {"SLSQP": {'maxiter': 100,
                                           'disp': True,
                                           'iprint': 2,
                                           'ftol': 1e-12,
                                           'eps': 0.1},
                                 "TURBO_1": {"n_init": 2*len(variables_names_list),
                                             "max_evals": 100,
                                             "batch_size": 1,  # 1 = Serial
                                             "verbose": True,
                                             "use_ard": True,
                                             "max_cholesky_size": 2000,
                                             "n_training_steps": 50,
                                             "min_cuda": 1024,
                                             "device": "cpu",
                                             "dtype": "float64"}}
        self.tuning_optimizer_dict = {'SLSQP': self._tuning_optimizer_scipy,
                                      'TURBO_1': self._tuning_optimizer_turbo_1}

        self.wf_model = wf_model
        self.variables_class_list = variables_class_list
        self.variables_names_list = variables_names_list
        self.variables_bounds_list = variables_bounds_list

        self.obj_func_name = obj_func_name
        self.obj_func = self.obj_func_dict[self.obj_func_name]
        self.opt_method = opt_method
        if opt_options == None:
            self.opt_options = self.opt_options_dict[self.opt_method]
        else:
            self.opt_options = opt_options
        self._tuning_optimizer = self.tuning_optimizer_dict[self.opt_method]

        self.tuning_data_received = False
        self.tuning_conditions_received = False

        print("\nInitialised parameter tuning")
        print("%i parameters to tune" % (len(self.variables_names_list)))
        print("%s optimization method" % (self.opt_method))

    def tuning_data(self, data_power_list):
        """
        Provide training higher-fidelity data for parameter tuning.
        Limited to power of each turbine for each condition ('RMSE')

        Args
        ----
        data_power_list : list of lists
           For each condition:
              list of turbines power output ('RMSE')
        """
        self.tuning_data_power_list = data_power_list
        self.tuning_data_received = True
        pass

    def tuning_conditions(self,
                          yaw_angles_list,
                          wind_directions_list,
                          wind_speeds_list,
                          turbulence_intensities_list,
                          wind_shear_list):
        """
        Define the wind farm conditions (yaw and atmospheric)
        of the higher-fidelity data.

        Args
        ----
        yaw_angles_list : list of lists
            For each condition, list of turbines yaw_angles
        wind_directions_list: list
            For each condtion, wind direction
        wind_speeds_list: list
            For each condtion, wind speed
        turbulence_intensities_list: list
            For each condtion, wind direction
        wind_shear_list: list
            For each condtion, wind shear
        """
        self.yaw_angles_list = yaw_angles_list
        self.wind_directions_list = wind_directions_list
        self.wind_speeds_list = wind_speeds_list
        self.turbulence_intensities_list = turbulence_intensities_list
        self.wind_shear_list = wind_shear_list
        self.tuning_conditions_received = True
        pass

    def tune_parameters(self):
        """
        Tune specified parameters of a WfModel object.
        Requires higher-fidelity tuning data and the related conditions to be
        previously specified (refer to Tuning methods: tuning_data and tuning_conditions).

        Returns
        -------
        wf_model_tuned: WfModel object
            WfModel object with parameters tuned
        wf_model_dict_opt: dictionary
            tuned WfModel object dictionary
        """
        # Double check tuning data and conditions have been specified
        if self.tuning_data_received is False:
            err_msg = "Tuning data not specified. Use tuning_data method."
            raise Exception(err_msg)
        if self.tuning_conditions_received is False:
            err_msg = "Tuning conditions not specified. Use tuning_conditions method."
            raise Exception(err_msg)

        # Extract original wf_model object dictionary and print its parameters
        self.wf_model_dict_original = floris_extract_object_dict(self.wf_model)
        self.models_dict = floris_extract_models_dict(self.wf_model_dict_original)
        floris_print_params(self.wf_model_dict_original,
                            self.models_dict,
                            "Original model parameters")

        # Extract initial variable values and normalise them
        self.variables_init = self._wf_model_dict_to_variables(self.wf_model_dict_original,
                                                               self.variables_class_list,
                                                               self.variables_names_list)
        self.variables_init_norm = self._norm_variables(self.variables_init,
                                                        self.variables_bounds_list)

        # Normalize variable bounds
        tmp = self.variables_bounds_list
        (self.variables_bounds_list_norm,
         self.variables_low_bound_list_norm,
         self.variables_upp_bound_list_norm) = self._norm_variables_bounds_lists(tmp)

        # Minimisation of error | Extract optimal variables
        self._tuning_optimizer()
        self.opt_variables = self._unnorm_variables(self.opt_variables_norm,
                                                    self.variables_bounds_list)

        # Apply tuned parameters (opt_variables) to wf_model and print them
        self.wf_model_dict_opt = self._vars_to_wf_model_dict(self.wf_model_dict_original,
                                                             self.variables_class_list,
                                                             self.variables_names_list,
                                                             self.opt_variables)
        self.wf_model = floris_param_change_object(self.wf_model, self.wf_model_dict_opt)
        floris_print_params(self.wf_model_dict_opt,
                            self.models_dict,
                            "Optimal model parameters")

        return self.wf_model, self.wf_model_dict_opt

    # %% Private methods

    def _wf_model_dict_to_variables(self, wf_model_dict, class_list, names_list):
        variables = []
        for i in range(len(names_list)):
            variable = floris_extract_parameter(wf_model_dict,
                                                class_list[i],
                                                names_list[i])
            variables.append(variable)
        return variables

    def _norm_variables(self, variables, variables_bounds_list):
        variables_norm = ([norm(variables[i],
                                variables_bounds_list[i][0],
                                variables_bounds_list[i][1])
                           for i in range(len(variables))])
        return variables_norm

    def _norm_variables_bounds_lists(self, variables_bounds_list):
        variables_bounds_list_norm = []
        variables_low_bound_list_norm = []
        variables_upp_bound_list_norm = []
        for i, variable_bounds in enumerate(variables_bounds_list):
            lower_bound_norm = norm(variable_bounds[0],
                                    variable_bounds[0],
                                    variable_bounds[1])
            upper_bound_norm = norm(variable_bounds[1],
                                    variable_bounds[0],
                                    variable_bounds[1])
            bound_norm_tuple = (lower_bound_norm, upper_bound_norm)
            variables_bounds_list_norm.append(bound_norm_tuple)
            variables_low_bound_list_norm.append(lower_bound_norm)
            variables_upp_bound_list_norm.append(upper_bound_norm)
        return (variables_bounds_list_norm,
                np.array(variables_low_bound_list_norm),
                np.array(variables_upp_bound_list_norm))

    def _unnorm_variables(self, variables_norm, variables_bounds_list):
        variables = ([unnorm(variables_norm[i],
                             variables_bounds_list[i][0],
                             variables_bounds_list[i][1])
                      for i in range(len(variables_norm))])
        return variables

    def _vars_to_wf_model_dict(self,
                               wf_model_dict_original,
                               variables_class_list,
                               variables_names_list,
                               variables):
        wf_model_dict_new = copy.deepcopy(wf_model_dict_original)
        for i in range(len(variables)):
            wf_model_dict_new = floris_param_change_object_dict(wf_model_dict_new,
                                                                variables_class_list[i],
                                                                variables_names_list[i],
                                                                variables[i])
        return wf_model_dict_new

    def _tuning_optimizer_scipy(self):
        self.opt_results = minimize(self.obj_func,
                                    self.variables_init_norm,
                                    method=self.opt_method,
                                    bounds=self.variables_bounds_list_norm,
                                    options=self.opt_options)
        self.opt_variables_norm = self.opt_results.x

    def _tuning_optimizer_turbo_1(self):
        turbo_1 = Turbo1(f=self.obj_func,
                         lb=self.variables_low_bound_list_norm,
                         ub=self.variables_upp_bound_list_norm,
                         **self.opt_options,
                         )
        turbo_1.optimize()
        X = turbo_1.X     # Evaluated points
        fX = turbo_1.fX   # Observed values
        index_best = np.argmin(fX)
        f_best, x_best = fX[index_best], X[index_best, :]
        self.opt_variables_norm = x_best

    def _tuning_rmse_function(self, variables_norm):

        # Unnorm variables, create new wf_model dictionary
        variables = self._unnorm_variables(variables_norm, self.variables_bounds_list)
        wf_model_dict_new = self._vars_to_wf_model_dict(self.wf_model_dict_original,
                                                        self.variables_class_list,
                                                        self.variables_names_list,
                                                        variables)

        # Create new wf_model object and reinitialize (atmospheric conditions set later)
        self.wf_model = floris_param_change_object(self.wf_model, wf_model_dict_new)

        rmse = 0
        for i in range(len(self.tuning_data_power_list)):

            # Calculate wind turbine power outputs with model to tune
            floris_reinitialise_atmosphere(self.wf_model,
                                           ws=self.wind_speeds_list[i],
                                           wd=self.wind_directions_list[i],
                                           ti=self.turbulence_intensities_list[i],
                                           shear=self.wind_shear_list[i])
            yaw_angles = np.array([float(item) for item in self.yaw_angles_list[i]])
            power_turbines = floris_calculate_turbine_power(self.wf_model, yaw_angles)

            # Calculate root mean squared error single condition
            error = 0
            for j in range(len(power_turbines)):
                error += (self.tuning_data_power_list[i][j]-power_turbines[j])**2
            rmse_single = error/len(power_turbines)

            # Calculate sum of root mean squared errors
            rmse += rmse_single

        return rmse
