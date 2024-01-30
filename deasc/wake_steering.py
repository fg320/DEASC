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
import random
import copy
from scipy.optimize import minimize, LinearConstraint
from turbo import Turbo1
from turbo import TurboM
from .utils_floris import (
    floris_reinitialise_atmosphere,
    floris_calculate_farm_power,
    floris_extract_object_dict
)
from .utils import (
    norm,
    unnorm
)


class WSOpt:
    """
    Class to perform wake steering optimization with a WfModel object, given an a-priori
    specified wind farm layout and specified atmopheric conditions. Optimization can have
    all/some turbines as variables, or rows for wind farms with equal columns, or turbine
    groups. Optimizers available are the local SLSQP, where linear constraints can be 
    added, and the global optimizer TuRBO.
    """

    def __init__(self,
                 wf_model,
                 inflow,
                 variables,
                 var_bounds,
                 var_initial,
                 opt_method="SLSQP",
                 opt_options=None,
                 obj_function="Farm Power",
                 constraints=(None, None, None),
                 by_row=(False, None, None),
                 grouping=False,
                 tuning_dynamic=False
                 ):
        """
        Args
        ----
        wf_model: (WfModel)
            WfModel to perform wake steering optimization.
        inflow: (list) Inflow conditions for wake steering optimization.
            yaw_initial: (list) wind farm yaw angles (deg).
                         (string) 'random' for random intial wind farm yaw angles.
            wd: (float) input wind directions (deg).
            ws: (float) input wind speeds (m/s).
            ti: (float) input turbulence intensity.
            shear: (float) shear exponent.
        variables: (list)
            List of turbines (or rows, or groups) to optimize. Naming convention
            starts from 1. If groups, a list of sublists is required. In each 
            group list, specify which turbines are in the group.
        var_bounds: (tuple)
            low_bound: (float) variable (yaw angle) lower bound.
            upp_bound: (float) variable (yaw angle) upper bound.
        var_initial:
            SLSQP: (list) list of initial variable values for each variable.
                   (string) 'random' for random initial variable values.
            TURBO_1: (list of lists) list of n_init variable values lists
                     (see TURBO_1 options).
                     (string) 'LHS' latin hypercube sampling.
            TURBO_M: (string) 'LHS' latin hypercube sampling.
        opt_method: (string, optional) optimization method.
            'SLSQP', 'TURBO_1 and 'TURBO_M' available.
            Default set to 'SLSQP'.
        opt_options: (dict , optional) optimization method options dictionary.
            Default set to None.
        opt_function: (string , optional) objective function. 'Farm Power' available
            Default set to 'Farm Power'.
        constraints: (tuple) Linear constraints definition. Limited to SLSQP.
           A: (matrix) linear constraint matrix.
              Default set to None.
           low_bound_constr: (float) lower non-normalized contraint bound.
              Default set to None.
           upp_bnd_constr: (float) upper non-normalized contraint bound.
              Default set to None.
        by_row : (tuple, optional) Optimization by row, requires all farm columns to have
            the same amount of rows.
            by_row_bool: (bool) True if optimization variables are wind farm rows,
                False if wind farm turbines. Default set to False.
            rows: (int) wind farm rows. Default set to None.
            cols: (int) wind farm columns. Default set to None.
        grouping: (bool) True if optimization variables are groups of turbines,
            False if wind farm turbines or wind farm rows. Default set to False.
        tuning_dynamic : (bool, optional)
            If True, include dynamic parameter tuning. See tuning_dynamic_initialize
            method. Default to False.
        """
        # Opt Methods - Opt Options - Optimizers - Opt Functions
        self.opt_method_list = ["SLSQP", "TURBO_1", "TURBO_M"]
        self.opt_options_dict = {"SLSQP": {'maxiter': 100,
                                           'disp': True,
                                           'iprint': 2,
                                           'ftol': 1e-6,
                                           'eps': 0.01},
                                 "TURBO_1": {"n_init": len(variables)*2,
                                             "max_evals": 500,
                                             "batch_size": 1,  # 1 = Serial
                                             "verbose": True,
                                             "use_ard": True,
                                             "max_cholesky_size": 2000,
                                             "n_training_steps": 50,
                                             "min_cuda": 1024,
                                             "device": "cpu",
                                             "dtype": "float64"},
                                 "TURBO_M": {"n_init": len(variables)*2,
                                             "max_evals": 500,
                                             "n_trust_regions": 2,
                                             "batch_size": 1,  # 1 = Serial
                                             "verbose": True,
                                             "use_ard": True,
                                             "max_cholesky_size": 2000,
                                             "n_training_steps": 50,
                                             "min_cuda": 1024,
                                             "device": "cpu",
                                             "dtype": "float64"}}
        self.optimizer_dict = {'SLSQP': self._optimizer_scipy,
                               'TURBO_1': self._optimizer_turbo_1,
                               'TURBO_M': self._optimizer_turbo_m}
        self.obj_function_dict = {'Farm Power': self._obj_function_power}

        # Optimization methods and optimizer
        self.opt_method = opt_method
        self._opt_method_settler()
        self.optimizer = self.optimizer_dict[self.opt_method]

        # Optimizer options
        self.opt_options = opt_options
        self._opt_options_settler()

        # Optimization function
        self.obj_function_name = obj_function
        self._obj_function_settler()

        # Wind farm conditions
        self.wf_model = wf_model
        self.wf_model_dict_original = floris_extract_object_dict(self.wf_model)
        self.yaw_initial, self.wd, self.ws, self.ti, self.shear = inflow
        if not isinstance(self.yaw_initial, (list, np.ndarray)):
            if self.yaw_initial == 'random':
                self.yaw_initial = self._random_yaw_generator(self.wf_model.n_turbs,
                                                              var_bounds)
        self._yaw_initial_input_handler()
        self.yaw_initial = np.array([float(item) for item in self.yaw_initial])

        # Optimization per wind farm row
        self.by_row_bool = by_row[0]
        if self.by_row_bool:
            self.rows = by_row[1]
            self.cols = by_row[2]
            self._by_row_input_handler()

        # Optimization per groups of wind turbines
        self.grouping_bool = grouping
        if self.grouping_bool:
            self.turbine_groups = variables
            self._grouping_input_handler()

        # Variable bounds
        self.var_bounds = var_bounds
        self.low_bound, self.upp_bound = self.var_bounds
        self.low_bound_norm = norm(
            self.low_bound, self.low_bound, self.upp_bound)
        self.upp_bound_norm = norm(
            self.upp_bound, self.low_bound, self.upp_bound)
        self.var_bounds_norm = (self.low_bound_norm, self.upp_bound_norm)
        tmp = [self.var_bounds_norm for i in range(len(variables))]
        self.var_bounds_norm_list = tmp
        tmp = np.array([self.low_bound_norm for i in range(len(variables))])
        self.low_bound_norm_list = tmp
        tmp = np.array([self.upp_bound_norm for i in range(len(variables))])
        self.upp_bound_norm_list = tmp

        # Constraints
        self.A = constraints[0]
        self.low_bound_constr = constraints[1]
        self.upp_bound_constr = constraints[2]
        if self.A is not None:
            self._constraints_input_handler()
            self.low_bound_constr_norm = norm(self.low_bound_constr,
                                              self.low_bound,
                                              self.upp_bound)
            self.upp_bound_constr_norm = norm(self.upp_bound_constr,
                                              self.low_bound,
                                              self.upp_bound)

        # Yaw variables
        if self.grouping_bool:
            self.variables = [x[0] for x in variables]
        else:
            self.variables = variables
        self.var_initial = var_initial
        self._variables_input_handler()
        if not isinstance(self.var_initial, (list, np.ndarray)):
            if self.opt_method == 'SLSQP' and self.var_initial == 'random':
                self.var_initial = self._random_yaw_generator(len(self.variables),
                                                              self.var_bounds)
        self._var_initial_input_handler()
        self.var_initial_norm = self._var_initial_norm()

        # Dynamic tuning
        self.tuning_dyn_bool = tuning_dynamic
        self._tuning_dyn_bool_check()
        self.tuning_dyn_initialization = False

        self.opt_run = False

    def tuning_dyn_initialize(self, tuning_dyn_obj_list):
        """
        Assign list of tuning dynamic objects TuningDyn to the WSOpt object.

        Args
        ----
        tuning_dyn_object: (list of TuningDyn objects)
        """
        self.tuning_dyn_obj_list = tuning_dyn_obj_list
        self._tuning_dyn_init_input_handler()
        for tuning_dyn_obj in self.tuning_dyn_obj_list:
            tuning_dyn_obj.wso_compatibility_check(self)
        self.tuning_dyn_initialization = True

    def optimize_yaw(self):
        """
        Optimize the yaw angle for the given WSOpt object.

        Returns
        -------
        opt_yaw_angles_vars: (ndarray) optimal yaw angles for the optimization variables.
        opt_yaw_angles_all: (ndarray) optimal yaw angles for all.wind farm turbines.
        """
        # Tuning dynamic initialization check
        self._tuning_dyn_initialization_check()

        # Print optimization info
        self._print_info()

        # Wind farm power - no yaw
        self.wf_pow_noyaw = self._get_farm_power_noyaw()

        # Optimize
        self._iter_details_setup()
        self.opt_yaw_angles_vars, self.opt_yaw_angles_all = self.optimizer()
        self.opt_run = True

        return (self.opt_yaw_angles_vars, self.opt_yaw_angles_all)

    def get_optimization_details(self):
        """
        Return optimization details: optimizer iterations details and objective function
        evaluations details. The two are identical for TURBO optimizers as an objective
        function evaluation corresponds to an optimizer iteration, different for SLSQP as
        additional objective function evaluations are required to approximate gradients.

        Returns
        -------
        iter_details: (tuple) optimizer iterations details.
            iter_yaw_angles: (list) list of yaw angles per optimizer iteration.
            iter_obj_func: (list) list of objective function per optimizer iteration.
            iter_farm_power: (list) list of farm power values per optimizer iteration.
        eval_details: (tuple) objective fucntion evaluations details.
            eval_yaw_angles: (list) list of yaw angles per evaluation.
            eval_obj_func: (list) list of objective function per evaluation.
            eval_farm_power: (list) list of farm power values per evaluation.
        """
        iter_details = (self.iter_yaw_angles,
                        self.iter_obj_func,
                        self.iter_farm_power)
        eval_details = (self.eval_yaw_angles,
                        self.eval_obj_func,
                        self.eval_farm_power)
        return (iter_details, eval_details)

    # %% Private methods

    def _opt_method_settler(self):
        if self.opt_method not in self.opt_method_list:
            err_msg = "Optimization method not recognized"
            raise Exception(err_msg)

    def _opt_options_settler(self):
        if self.opt_options is None:
            self.opt_options = self.opt_options_dict[self.opt_method]

    def _obj_function_settler(self):
        if self.obj_function_name in list(self.obj_function_dict.keys()):
            self.obj_function = self.obj_function_dict[self.obj_function_name]
        else:
            err_msg = "Optimization function not recognized"
            raise Exception(err_msg)

    def _random_yaw_generator(self, yaw_number, yaw_bounds):
        yaw_angles = []
        for i in range(yaw_number):
            x = random.choice(range(yaw_bounds[0], yaw_bounds[1]+1))
            yaw_angles.append(x)
        return yaw_angles

    def _yaw_initial_input_handler(self):
        if len(self.yaw_initial) != self.wf_model.n_turbs:
            err_msg = "Initial yaw angles do not match turbine number"
            raise Exception(err_msg)

    def _by_row_input_handler(self):
        if self.rows*self.cols != self.wf_model.n_turbs:
            err_msg = "Farm rows and columns provided do not match turbine number"
            raise Exception(err_msg)

    def _grouping_input_handler(self):
        if self.grouping_bool and self.by_row_bool:
            err_msg = "by_row and grouping cannot be used together"
            raise Exception(err_msg)
        if not isinstance(self.turbine_groups[0], (list, np.ndarray)):
            err_msg = "In variable list, turbine groups need to be sublists"
            raise Exception(err_msg)

    def _constraints_input_handler(self):
        if self.opt_method != 'SLSQP':
            err_msg = "Linear constraints (on top of bounds) limited to SLSQP optimizer"
            raise Exception(err_msg)

    def _variables_input_handler(self):
        if self.by_row_bool:
            variables_check = self.variables
            for row in variables_check:
                if row > self.rows:
                    err_msg = "Row/s specified not in farm"
                    raise Exception(err_msg)
                if len(variables_check) > self.rows:
                    err_msg = "Too many rows specified"
                    raise Exception(err_msg)
        else:
            if self.grouping_bool:
                variables_check = [
                    x for sublist in self.turbine_groups for x in sublist]
            else:
                variables_check = self.variables
            repeated = set()
            for turb in variables_check:
                if turb > self.wf_model.n_turbs:
                    err_msg = "Turbine/s specified not in the farm"
                    raise Exception(err_msg)
                if turb in repeated:
                    err_msg = "Repeated turbine specified."
                    raise Exception(err_msg)
            if len(variables_check) > self.wf_model.n_turbs:
                err_msg = "Too many turbines specified"
                raise Exception(err_msg)
        if 0 in variables_check:
            err_msg = "Turbine/row counting convention starts from 1"
            raise Exception(err_msg)

    def _var_initial_input_handler(self):
        if self.opt_method == 'TURBO_1':
            if not isinstance(self.var_initial, (list, np.ndarray)):
                if self.var_initial == 'LHS':
                    pass
                elif self.var_initial == 'random':
                    err_msg = "Random initial variables limited to SLSQP optimizer"
                    raise Exception(err_msg)
            else:
                if len(self.var_initial) != self.opt_options["n_init"]:
                    err_msg = "n_init initial variable lists are needed (see TURBO options)"
                    raise Exception(err_msg)
                elif len(self.var_initial[0]) != len(self.variables):
                    err_msg = "var_initial sublists length not equal number of variables"
                    raise Exception(err_msg)
        elif self.opt_method == 'TURBO_M':
            if self.var_initial != 'LHS':
                err_msg = "TURBO_M optimizer requires LHS as initial sampling"
        elif self.opt_method == 'SLSQP':
            if not isinstance(self.var_initial, (list, np.ndarray)):
                if self.var_initial == 'LHS':
                    err_msg = "Latin Hypercube Sampling limited to TURBO optimizers"
                    raise Exception(err_msg)
            elif len(self.variables) != len(self.var_initial):
                err_msg = "var_initial length needs to equal number of variables"
                raise Exception(err_msg)

    def _var_initial_norm(self):
        if self.opt_method == "SLSQP":
            self.var_initial = np.array([float(item)
                                        for item in self.var_initial])
            var_initial_norm = norm(
                self.var_initial, self.low_bound, self.upp_bound)
        elif self.var_initial == 'LHS':
            var_initial_norm = None
        else:
            self.var_initial = np.array([np.array(x)
                                        for x in self.var_initial])
            var_initial_norm = []
            for x_list in self.var_initial:
                x_list_norm = []
                for x in x_list:
                    x_norm = norm(x, self.low_bound, self.upp_bound)
                    x_list_norm.append(x_norm)
                var_initial_norm.append(np.array(x_list_norm))
        return np.array(var_initial_norm)

    def _get_farm_power_noyaw(self):
        if (self.tuning_dyn_initialization and
                hasattr(self.tuning_dyn_obj_list[0], 'wf_pow_noyaw')):
            wf_pow_noyaw = self.tuning_dyn_obj_list[0].wf_pow_noyaw
        else:
            self.yaw_zero = np.full(
                shape=self.wf_model.n_turbs, fill_value=0.0)
            self.wf_model = floris_reinitialise_atmosphere(self.wf_model,
                                                           self.ws,
                                                           self.wd,
                                                           self.ti,
                                                           self.shear)
            # Tune parameters
            if self.tuning_dyn_initialization:
                for tuning_dyn_obj in self.tuning_dyn_obj_list:
                    self.wf_model = tuning_dyn_obj.tune_parameter(
                        self, self.yaw_zero)

            wf_pow_noyaw = floris_calculate_farm_power(
                self.wf_model, self.yaw_zero)
        return wf_pow_noyaw

    def _print_info(self):
        print("=====================================================")
        print("Optimizing wake redirection control...")
        print("Optimization method: %s" % (self.opt_method))
        print("Optimization function: %s \n" % (self.obj_function_name))
        if self.by_row_bool:
            print("Rows being optimized: ")
            print(self.variables)
        elif self.grouping_bool:
            print("First turbine of groups being optimized: ")
            print(self.variables)
        else:
            print("Turbines being optimized: ")
            print(self.variables)
        print("Number of variables to optimize = ", len(self.variables))
        print("=====================================================")

    def _iter_details_setup(self):
        # Details for each obj function evaluation
        self.eval_yaw_angles = []  # deg
        self.eval_obj_func = []
        self.eval_farm_power = []  # MW

        # Details for each optimizer iteration
        self.iter_yaw_angles = []  # deg
        self.iter_obj_func = []
        self.iter_farm_power = []  # MW

    def _variables_to_farm_yaw(self, yaw_initial, var_values):
        yaw_angles = copy.deepcopy(yaw_initial)
        if self.by_row_bool:
            for i, row_idx in enumerate(self.variables):
                idx_1 = row_idx*self.cols
                idx_0 = idx_1-self.cols
                yaw_angles[idx_0:idx_1] = var_values[i]
        elif self.grouping_bool:
            for i, turbine_idx_list in enumerate(self.turbine_groups):
                for j, turb_idx in enumerate(turbine_idx_list):
                    yaw_angles[turb_idx-1] = var_values[i]
        else:
            for i, turb_idx in enumerate(self.variables):
                yaw_angles[turb_idx-1] = var_values[i]
        return yaw_angles.tolist()

    # %% Optimizers

    def _optimizer_scipy(self):
        # Call back function for iter details
        def callback_func(xk):
            self.iter_yaw_angles.append(self.eval_yaw_angles[-1])
            self.iter_obj_func.append(self.eval_obj_func[-1])
            self.iter_farm_power.append(self.eval_farm_power[-1])
        # Linearly constrained case
        if self.A is not None:
            self.C = LinearConstraint(self.A,
                                      self.low_bound_constr_norm,
                                      self.upp_bound_constr_norm)
            self.residual_plant = minimize(self.obj_function,
                                           self.var_initial_norm,
                                           callback=callback_func,
                                           method=self.opt_method,
                                           bounds=self.var_bounds_norm_list,
                                           constraints=(self.C,),
                                           options=self.opt_options)
        # Unconstrained case
        else:
            self.residual_plant = minimize(self.obj_function,
                                           self.var_initial_norm,
                                           callback=callback_func,
                                           method=self.opt_method,
                                           bounds=self.var_bounds_norm_list,
                                           options=self.opt_options)
        # Extract optimal yaw angles for variables
        opt_yaw_angles_vars = unnorm(self.residual_plant.x,
                                     self.low_bound,
                                     self.upp_bound)
        # Extract optimal yaw angles for the entire farm
        opt_yaw_angles_all = self._variables_to_farm_yaw(self.yaw_initial,
                                                         opt_yaw_angles_vars)

        # Use best index because if total iterations reached, optimum not last evaluation
        eval_yaw_angles_lists = [x.tolist() for x in self.eval_yaw_angles]
        index_best = eval_yaw_angles_lists.index(opt_yaw_angles_all)
        opt_yaw_angles_all = np.array(opt_yaw_angles_all)
        self.obj_func_opt = self.eval_obj_func[index_best]
        self.farm_power_opt = self.eval_farm_power[index_best]

        # Add initial and last points to iteration details
        self.iter_yaw_angles.insert(0, self.eval_yaw_angles[0])
        self.iter_obj_func.insert(0, self.eval_obj_func[0])
        self.iter_farm_power.insert(0, self.eval_farm_power[0])
        self.iter_yaw_angles.append(self.eval_yaw_angles[-1])
        self.iter_obj_func.append(self.eval_obj_func[-1])
        self.iter_farm_power.append(self.eval_farm_power[-1])

        return (opt_yaw_angles_vars, opt_yaw_angles_all)

    def _optimizer_turbo_1(self):

        # TURBO initial sampling
        if not isinstance(self.var_initial, (list, np.ndarray)):
            if self.var_initial == 'LHS':
                X_init_provided = False
                X_init_same_norm = None
        else:
            X_init_provided = True
            X_init_same_norm = self.var_initial_norm

        # TURBO optimization
        turbo_1 = Turbo1(f=self.obj_function,
                         lb=self.low_bound_norm_list,
                         ub=self.upp_bound_norm_list,
                         **self.opt_options,
                         X_init_provided=X_init_provided,
                         X_init_same=X_init_same_norm,
                         )
        turbo_1.optimize()
        X = turbo_1.X     # Evaluated points
        fX = turbo_1.fX   # Observed values
        index_best = np.argmin(fX)
        f_best, x_best = fX[index_best], X[index_best, :]

        # Extract optimal yaw angles for variables and the entire farm
        opt_yaw_angles_vars = unnorm(x_best,
                                     self.low_bound,
                                     self.upp_bound)
        opt_yaw_angles_all = self._variables_to_farm_yaw(self.yaw_initial,
                                                         opt_yaw_angles_vars)

        # Update iteration details (same as evaluation details)
        self.iter_yaw_angles = self.eval_yaw_angles
        self.iter_obj_func = self.eval_obj_func
        self.iter_farm_power = self.eval_farm_power

        # Use best index because last iteration might not be the optimal one
        self.obj_func_opt = f_best[0]
        self.farm_power_opt = self.iter_farm_power[index_best]

        return (opt_yaw_angles_vars, opt_yaw_angles_all)

    def _optimizer_turbo_m(self):

       # TURBO optimization
        turbo_m = TurboM(f=self.obj_function,
                         lb=self.low_bound_norm_list,
                         ub=self.upp_bound_norm_list,
                         **self.opt_options,
                         )
        turbo_m.optimize()
        X = turbo_m.X     # Evaluated points
        fX = turbo_m.fX   # Observed values
        index_best = np.argmin(fX)
        f_best, x_best = fX[index_best], X[index_best, :]

        # Extract optimal yaw angles for variables and the entire farm
        opt_yaw_angles_vars = unnorm(x_best,
                                     self.low_bound,
                                     self.upp_bound)
        opt_yaw_angles_all = self._variables_to_farm_yaw(self.yaw_initial,
                                                         opt_yaw_angles_vars)

        # Update iteration details (same as evaluation details)
        self.iter_yaw_angles = self.eval_yaw_angles
        self.iter_obj_func = self.eval_obj_func
        self.iter_farm_power = self.eval_farm_power

        # Use best index because last iteration might not be the optimal one
        self.cost_func_opt = f_best[0]
        self.farm_power_opt = self.iter_farm_power[index_best]

        return (opt_yaw_angles_vars, opt_yaw_angles_all)

    # %% Objective functions

    def _obj_function_power(self, var_norm):

        # Extract farm yaw angles
        var_unnorm = unnorm(var_norm, self.low_bound, self.upp_bound)
        yaw_angles = self._variables_to_farm_yaw(self.yaw_initial, var_unnorm)
        yaw_angles = np.array([float(item) for item in yaw_angles])

        # Tune parameters dynamically
        if self.tuning_dyn_initialization:
            for tuning_dyn_obj in self.tuning_dyn_obj_list:
                self.wf_model = tuning_dyn_obj.tune_parameter(self, yaw_angles)

        # Calculate negative of the farm power normalized by power for zero yaw
        self.wf_model = floris_reinitialise_atmosphere(self.wf_model,
                                                       self.ws,
                                                       self.wd,
                                                       self.ti,
                                                       self.shear)
        wf_pow = floris_calculate_farm_power(self.wf_model, yaw_angles)
        obj_function = (-1 * wf_pow / self.wf_pow_noyaw)

        # Update evalauation details
        self.eval_yaw_angles.append(yaw_angles)
        self.eval_obj_func.append(obj_function)
        self.eval_farm_power.append(wf_pow)

        return obj_function

    # %% Tuning Dynamic methods

    def _tuning_dyn_bool_check(self):
        if self.tuning_dyn_bool and self.by_row_bool:
            err_msg = "Dynamic tuning not available for optimization by row."
            raise Exception(err_msg)

    def _tuning_dyn_init_input_handler(self):
        if isinstance(self.tuning_dyn_obj_list, (list, np.ndarray)) is False:
            err_msg = "TuningDyn objects need to be in a list, even if only one."
            raise Exception(err_msg)
        # Check grouping boolean is the same between tuning objects and wso one
        for tuning_dyn_obj in self.tuning_dyn_obj_list:
            if (tuning_dyn_obj.grouping_bool ^ self.grouping_bool):
                err_msg = "TuningDyn objects and WSOpt have different grouping booleans."
                raise Exception(err_msg)
        # Check dynamic grouping tuning objects have the same tuning groups
        if self.grouping_bool:
            tuning_groups_first = self.tuning_dyn_obj_list[0].tuning_groups
            same_groups = all(obj.tuning_groups == tuning_groups_first
                              for obj in self.tuning_dyn_obj_list)
            if same_groups is False:
                err_msg = "TuningDyn objects have different groupings."
                raise Exception(err_msg)

    def _tuning_dyn_initialization_check(self):
        if self.tuning_dyn_bool and self.tuning_dyn_initialization is False:
            err_msg = "Tuning dynamic not initialized. See tuning_dyn_initialize method."
            raise Exception(err_msg)
