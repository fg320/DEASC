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
from abc import ABC, abstractmethod, abstractproperty
from deasc.utils_floris import (
    floris_extract_object_dict,
    floris_extract_parameter,
    floris_param_change_object_dict,
    floris_param_change_object
)

# Starting index for turbine counting convention
STARTING_INDEX = 1


class TuningDyn(ABC):
    """High-level abstract class for dynamic parameter tuning of a single parameter."""

    def __init__(self, param_class, param_name):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        """
        # Parameter info
        self.param_class = param_class
        self.param_name = param_name

    @abstractproperty
    def tuning_turbines(self):
        """Abstract property for tuning turbines."""
        pass

    @abstractmethod
    def wso_compatibility_check(self, wso_obj):
        """Abstract method for compatibility between TuningDyn and WSOpt objects."""
        pass

    @abstractmethod
    def tune_parameter(self, wso_obj, yaw_angles):
        """Abstract method for tuning a parameter in a WSOpt objects."""
        pass


class TuningDyn_SharedMethods(ABC):
    """Abstract class for methods shared by some or all TuningDyn child classes."""

    def __init__(self):
        pass

    def _GP_dimension_check(self, tuning_dimensions, GP_model):
        if tuning_dimensions == 0:
            pass
        elif GP_model.input_dim != tuning_dimensions:
            err_msg = (
                "Number of GP model dimensions does not match tuning variables.")
            raise Exception(err_msg)

    def _GP_dimension_check_1GP(self, tuning_dimensions, GP_model):
        if tuning_dimensions == 0:
            pass
        elif GP_model.input_dim < tuning_dimensions:
            err_msg = (
                "Number of GP model dimensions is lower than column tuning variables.")
            raise Exception(err_msg)

    def _tuning_turbines_check(self, wso_obj, turbines):
        repeated = set()
        for turbine in turbines:
            if turbine > wso_obj.wf_model.n_turbs:
                err_msg = "Turbine specified not in the farm."
                raise Exception(err_msg)
            if turbine < STARTING_INDEX:
                err_msg = "Turbine/row counting convention starts from 1."
                raise Exception(err_msg)
            if turbine in repeated:
                err_msg = "Repeated turbine specified."
                raise Exception(err_msg)
            else:
                repeated.add(turbine)
        if len(turbines) > wso_obj.wf_model.n_turbs:
            err_msg = "Too many turbines specified."
            raise Exception(err_msg)

    def _tuning_groups_check(self, wso_obj):
        # Check that wso and tuning groups match when there is a common turbine
        for i, tuning_group in enumerate(self.tuning_groups):
            common_elements_found = False
            for wso_group in wso_obj.turbine_groups:
                if any(element in tuning_group for element in wso_group):
                    common_elements_found = True
                    condition1 = len(tuning_group) == len(wso_group)
                    condition2 = all(x == y for x, y in zip(
                        tuning_group, wso_group))
                    if (condition1 and condition2) is False:
                        err_msg = ("Tuning group %i does not match between "
                                   % (i+1) + "wso and tuning.")
                        raise Exception(err_msg)
            # Check that for the tuning groups not in wso, the turbines have
            # the same yaw angle
            if not common_elements_found:
                indices = [i for i in tuning_group]
                if len(set(wso_obj.yaw_initial[i] for i in indices)) != 1:
                    err_msg = ("Tuning groups not in wso do not have equal " +
                               "yaw angles.")
                    raise Exception(err_msg)

    def _turbines_cols_check(self, wso_obj):
        turbines = [x for sublist in self.turbines_cols for x in sublist]
        if len(turbines) != wso_obj.wf_model.n_turbs:
            err_msg = ("Mismatch in number of turbines in columns and total number"
                       + " in the farm.")
            raise Exception(err_msg)
        self._tuning_turbines_check(wso_obj, turbines)

    def _tuning_turbines_cols_dict_check(self):
        for key in self.tuning_variables_cols_dict.keys():
            col_len = int(key[0])
            tuning_turbines_loc = self.tuning_variables_cols_dict[key]
            if len(tuning_turbines_loc) > col_len:
                err_msg = "Too many turbine specified in tuning turbines dictionary."
                raise Exception(err_msg)
            for turbine_loc in tuning_turbines_loc:
                if turbine_loc > col_len:
                    err_msg = "Turbine specified outside of column."
                    raise Exception(err_msg)
                if turbine_loc < STARTING_INDEX:
                    err_msg = "Turbine/row counting convention starts from 1."
                    raise Exception(err_msg)

    def _tuning_groups_cols_dict_check(self):
        for key in self.tuning_variables_cols_dict.keys():
            col_len = int(key[0])
            tuning_groups_loc = self.tuning_variables_cols_dict[key]
            if len(tuning_groups_loc) > col_len:
                err_msg = "Too many groups specified in tuning turbines dictionary."
                raise Exception(err_msg)
            for group_loc in tuning_groups_loc:
                if len(group_loc) > col_len:
                    err_msg = "Too many turbines specified in tuning groups dictionary."
                    raise Exception(err_msg)
                for turbine_loc in group_loc:
                    if turbine_loc > col_len:
                        err_msg = "Turbine specified outside of column."
                        raise Exception(err_msg)
                    if turbine_loc < STARTING_INDEX:
                        err_msg = "Turbine/row counting convention starts from 1."
                        raise Exception(err_msg)

    def _get_tuning_turbines_cols(self):
        tuning_turbines_cols = []
        for turbines in self.turbines_cols:
            tuning_turbines = []
            key = "%ix1" % (len(turbines))
            tuning_turbines_loc = self.tuning_variables_cols_dict[key]
            for turbine_loc in tuning_turbines_loc:
                turbine_idx = turbine_loc-STARTING_INDEX
                tuning_turbines.append(turbines[turbine_idx])
            tuning_turbines_cols.append(tuning_turbines)
        return tuning_turbines_cols

    def _get_tuning_groups_cols(self):
        tuning_groups_cols = []
        for turbines in self.turbines_cols:
            tuning_groups = []
            key = "%ix1" % (len(turbines))
            tuning_groups_loc = self.tuning_variables_cols_dict[key]
            for group_loc in tuning_groups_loc:
                tuning_turbines = []
                for turbine_loc in group_loc:
                    turbine_idx = turbine_loc-STARTING_INDEX
                    tuning_turbines.append(turbines[turbine_idx])
                tuning_groups.append(tuning_turbines)
            tuning_groups_cols.append(tuning_groups)
        return tuning_groups_cols

    def _get_GP_model_cols(self):
        GP_model_cols = []
        for col_len in self.turbines_cols_len:
            key = "%ix1" % (col_len)
            GP_model = self.GP_model_cols_dict[key]
            GP_model_cols.append(GP_model)
        return GP_model_cols

    def _get_GP_corr_cols(self):
        GP_corr_cols = []
        for col_len in self.turbines_cols_len:
            key = "%ix1" % (col_len)
            GP_corr = self.GP_corr_cols_dict[key]
            GP_corr_cols.append(GP_corr)
        return GP_corr_cols

    def _get_GP_input_turbines(self, tuning_turbines, yaw_angles):
        GP_input = np.array([])
        for turbine in tuning_turbines:
            turbine_idx = turbine-STARTING_INDEX
            GP_input = np.append(GP_input, yaw_angles[turbine_idx])
        return GP_input

    def _get_GP_input_groups(self, tuning_groups, yaw_angles):
        GP_input = np.array([])
        for group in tuning_groups:
            turbine_idx = group[0]-STARTING_INDEX
            GP_input = np.append(GP_input, yaw_angles[turbine_idx])
        return GP_input

    def _fill_opt_param_list(self,
                             opt_param_list,
                             turbines,
                             optimal_parameter):
        for turbine in turbines:
            idx = turbine-STARTING_INDEX
            opt_param_list[idx] = optimal_parameter
        return opt_param_list


class TuningDyn_Turbines(TuningDyn, TuningDyn_SharedMethods):
    """Class for dynamic parameter tuning of single turbines within a wind farm."""

    def __init__(self, param_class, param_name, tuning_turbines, GP_model):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        tuning_turbines: (list) list of turbines included in the tuning.
        GP_model: (GPy object) GP model with len(tuning_turbines) input dimensions.
        """
        super().__init__(param_class, param_name)
        # Tuning info
        self.tuning_variables = tuning_turbines
        self.tuning_dimensions = len(self.tuning_variables)
        self.GP_model = GP_model
        self._GP_dimension_check(self.tuning_dimensions, self.GP_model)
        self.grouping_bool = False

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return self.tuning_variables

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)

        # Create and apply tuned WfModel dictionary
        GP_input = self._get_GP_input_turbines(
            self.tuning_turbines, yaw_angles)
        mu, var, = self.GP_model.predict_noiseless(np.array([GP_input]))
        optimal_parameter = mu[0][0]
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              optimal_parameter)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Grouping(TuningDyn, TuningDyn_SharedMethods):
    """Class for dynamic parameter tuning with grouping of turbines within a wind farm."""

    def __init__(self, param_class, param_name, tuning_groups, GP_model):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        tuning_groups: (list of lists) list of turbine groups included in the tuning. In
            each list, specify the turbines in the group.
        GP_model: (GPy object) GP model with len(tuning_groups) input dimensions.
        """
        super().__init__(param_class, param_name)
        # Tuning info
        self.tuning_variables = tuning_groups
        self.tuning_dimensions = len(self.tuning_variables)
        self.GP_model = GP_model
        # GP dimension check
        self._GP_dimension_check(self.tuning_dimensions, self.GP_model)
        # Grouping info
        self.tuning_groups = tuning_groups
        self.grouping_bool = True

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_variables for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._tuning_groups_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)

        # Create and apply tuned WfModel dictionary
        GP_input = self._get_GP_input_groups(self.tuning_groups, yaw_angles)
        mu, var, = self.GP_model.predict_noiseless(np.array([GP_input]))
        optimal_parameter = mu[0][0]
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              optimal_parameter)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Turbines_CI(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning using column-independence (CI) of turbines
    within a wind farm.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_turbines_cols_dict,
                 GP_model_cols_dict):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_turbines_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding list values with the turbines
            to tune (turbine naming convention relative to the single column, e.g. [1,2]).
            All column lenghts to be included, even if tuning turbines is an empty [] list.
        GP_model_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding values are the corresponding
            GPy models for the column length. All column lenghts to be included. For the
            "1x1" key, None is acceptable as no tuning is performed.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_turbines_cols_dict
        self._tuning_turbines_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_turbines_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        self.GP_model_cols_dict = GP_model_cols_dict
        self.GP_model_cols = self._get_GP_model_cols()
        # GP dimension check
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check(self.tuning_dimensions_cols[i],
                                     self.GP_model_cols[i])
        self.grouping_bool = False

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_variables_cols for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no turbines to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_turbines(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model_cols[i]
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Grouping_CI(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning with grouping using column-independence (CI)
    of turbines within a wind farm.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_groups_cols_dict,
                 GP_model_cols_dict):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_groups_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "5x1") and corresponding list of lists values with the
            groups of turbines to tune. For each group list, include sublists of turbines
            in the group (turbine naming convention relative to the single column,
            e.g. [[1,2],[3,4]]). All column lenghts to be included, where for a
            "1x1" an empt list [] is required.
        GP_model_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding values are the corresponding
            GPy models for the column length. All column lenghts to be included. For the
            "1x1" key, None is acceptable as no tuning is performed.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_groups_cols_dict
        self._tuning_groups_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_groups_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        self.GP_model_cols_dict = GP_model_cols_dict
        self.GP_model_cols = self._get_GP_model_cols()
        # GP dimension check
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check(self.tuning_dimensions_cols[i],
                                     self.GP_model_cols[i])
        # Grouping info
        self.tuning_groups_cols = self.tuning_variables_cols
        self.tuning_groups = [x for y in self.tuning_groups_cols for x in y]
        self.grouping_bool = True

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_groups for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._tuning_groups_check(wso_obj)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no group to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_groups(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model_cols[i]
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Looping_Turbine(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning with the looping approach of turbines within
    a wind farm.
    """

    def __init__(self, param_class, param_name, tuning_turbine, GP_model, wf_pow_noyaw):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        tuning_turbines: (list) list of single turbine included in the tuning.
        GP_model: (GPy object) GP model with a single input dimension.
        wf_pow_noyaw: (float) value of the wind farm power without any yaw applied,
            usually extracted from the previous grouping optimisation to refine.
        """
        super().__init__(param_class, param_name)
        # Tuning info
        self.tuning_variables = tuning_turbine
        self.tuning_dimensions = len(self.tuning_variables)
        self.GP_model = GP_model
        self._GP_dimension_check(self.tuning_dimensions, self.GP_model)
        # Looping info
        self.wf_pow_noyaw = wf_pow_noyaw
        self.tuning_bool = True
        self.grouping_bool = False

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return self.tuning_variables

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._looping_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)

        # Create and apply tuned WfModel dictionary
        GP_input = self._get_GP_input_turbines(
            self.tuning_turbines, yaw_angles)
        mu, var, = self.GP_model.predict_noiseless(np.array([GP_input]))
        optimal_parameter = mu[0][0]
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              optimal_parameter)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned

    def _looping_check(self, wso_obj):
        if len(self.tuning_variables) != 1:
            err_msg = "While looping, only a single turbine can be tuned."
            raise Exception(err_msg)
        if len(wso_obj.variables) != 1:
            err_msg = "While looping, only a single turbine can be optimised."
            raise Exception(err_msg)


class TuningDyn_Turbines_CI_1GP(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning using column-independence (CI) of turbines
    within a wind farm. A single GP tuned on a single column is used for all 
    column lenghts. No correctors are added. Dimensions in excess between column
    yaw variables are set to 0.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_turbines_cols_dict,
                 GP_model):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_turbines_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding list values with the turbines
            to tune (turbine naming convention relative to the single column, e.g. [1,2]).
            All column lenghts to be included, even if tuning turbines is an empty [] list.
        GP_model: (GPy object) single GP model for all farm columns.
            All column lenghts interpolate from this GP model and require an equal
            or lower number of turbines to tune.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_turbines_cols_dict
        self._tuning_turbines_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_turbines_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        self.GP_model = GP_model
        # GP dimension check (greater or equal than tuning dimensions per column)
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check_1GP(self.tuning_dimensions_cols[i],
                                         self.GP_model)
        self.grouping_bool = False

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_variables_cols for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no turbines to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_turbines(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model
                # Add missing dimensions to GP input if required. Set to zero.
                dim_missing = (GP_model.input_dim -
                               self.tuning_dimensions_cols[i])
                for _ in range(dim_missing):
                    GP_input = np.append(GP_input, float(0))
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Turbines_CI_1GP_corr(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning using column-independence (CI) of turbines
    within a wind farm. A single GP tuned on a single column is used for all 
    column lenghts. Correctors are added for each column length. Dimensions in 
    excess between column yaw variables are set to 0.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_turbines_cols_dict,
                 GP_model,
                 GP_corr_cols_dict):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_turbines_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding list values with the turbines
            to tune (turbine naming convention relative to the single column, e.g. [1,2]).
            All column lenghts to be included, even if tuning turbines is an empty [] list.
        GP_model: (GPy object) single GP model for all farm columns.
            All column lenghts interpolate from this GP model and require an equal
            or lower number of turbines to tune.
        GP_corr_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding values are the corresponding
            GPy models correctors for the column length. All column lenghts to be included.
            For the "1x1" key, None is acceptable as no tuning is performed.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_turbines_cols_dict
        self._tuning_turbines_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_turbines_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        # GP model
        self.GP_model = GP_model
        # GP dimension check (greater or equal than tuning dimensions per column)
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check_1GP(self.tuning_dimensions_cols[i],
                                         self.GP_model)
        # GP correctors
        self.GP_corr_cols_dict = GP_corr_cols_dict
        self.GP_corr_cols = self._get_GP_corr_cols()
        # GP corrections dimension check
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check(self.tuning_dimensions_cols[i],
                                     self.GP_corr_cols[i])
        self.grouping_bool = False

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_variables_cols for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no turbines to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_turbines(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model
                GP_corr = self.GP_corr_cols[i]
                # Corrector prediction
                mu_corr, var_corr = GP_corr.predict_noiseless(
                    np.array([GP_input]))
                correction = mu_corr[0][0]
                # Add missing dimensions to GP input if required. Set to zero.
                dim_missing = (GP_model.input_dim -
                               self.tuning_dimensions_cols[i])
                for _ in range(dim_missing):
                    GP_input = np.append(GP_input, float(0))
                # GP k_opt prediction
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]+correction
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Grouping_CI_1GP(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning with grouping using column-independence (CI)
    of turbines within a wind farm. A single GP tuned on a single column is used for
    all column lenghts. No correctors are added. Dimensions in excess between column
    group yaw variables are set to 0.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_groups_cols_dict,
                 GP_model):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_groups_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "5x1") and corresponding list of lists values with the
            groups of turbines to tune. For each group list, include sublists of turbines
            in the group (turbine naming convention relative to the single column,
            e.g. [[1,2],[3,4]]). All column lenghts to be included, where for a
            "1x1" an empt list [] is required.
        GP_model: (GPy object) single GP model for all farm columns.
            All column lenghts interpolate from this GP model and require an equal
            or lower number of groups to tune.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_groups_cols_dict
        self._tuning_groups_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_groups_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        self.GP_model = GP_model
        # GP dimension check (greater or equal than tuning dimensions per column)
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check_1GP(self.tuning_dimensions_cols[i],
                                         self.GP_model)
        # Grouping info
        self.tuning_groups_cols = self.tuning_variables_cols
        self.tuning_groups = [x for y in self.tuning_groups_cols for x in y]
        self.grouping_bool = True

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_groups for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._tuning_groups_check(wso_obj)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no group to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_groups(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model
                # Add missing dimensions to GP input if required. Set to zero.
                dim_missing = (GP_model.input_dim -
                               self.tuning_dimensions_cols[i])
                for _ in range(dim_missing):
                    GP_input = np.append(GP_input, float(0))
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned


class TuningDyn_Grouping_CI_1GP_corr(TuningDyn, TuningDyn_SharedMethods):
    """
    Class for dynamic parameter tuning with grouping using column-independence (CI)
    of turbines within a wind farm. A single GP tuned on a single column is used for
    all column lenghts. Correctors are added for each column length. Dimensions in 
    excess between column group yaw variables are set to 0.
    """

    def __init__(self,
                 param_class,
                 param_name,
                 turbines_cols,
                 tuning_groups_cols_dict,
                 GP_model,
                 GP_corr_cols_dict):
        """
        Args
        ----
        param_class: (string) tuning parameter class.
        param_name: (string) tuning parameter name.
        turbine_cols: (list of lists) list of lists, each containing the turbines for
            each column in the effective wind farm layout. List of list even for a single
            turbine column.
        tuning_groups_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "5x1") and corresponding list of lists values with the
            groups of turbines to tune. For each group list, include sublists of turbines
            in the group (turbine naming convention relative to the single column,
            e.g. [[1,2],[3,4]]). All column lenghts to be included, where for a
            "1x1" an empt list [] is required.
        GP_model: (GPy object) single GP model for all farm columns.
            All column lenghts interpolate from this GP model and require an equal
            or lower number of groups to tune.
        GP_corr_cols_dict: (dict) dictionary with string keys corresponding to
            column lenghts (e.g., "2x1") and corresponding values are the corresponding
            GPy models correctors for the column length. All column lenghts to be included.
            For the "1x1" key, None is acceptable as no tuning is performed.
        """
        super().__init__(param_class, param_name)
        # Farm columns info
        self.turbines_cols = turbines_cols
        self.turbines_cols_len = [len(col) for col in self.turbines_cols]
        # Tuning info
        self.tuning_variables_cols_dict = tuning_groups_cols_dict
        self._tuning_groups_cols_dict_check()
        self.tuning_variables_cols = self._get_tuning_groups_cols()
        self.tuning_dimensions_cols = [len(item)
                                       for item in self.tuning_variables_cols]
        # GP model
        self.GP_model = GP_model
        # GP dimension check (greater or equal than tuning dimensions per column)
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check_1GP(self.tuning_dimensions_cols[i],
                                         self.GP_model)
        # GP correctors
        self.GP_corr_cols_dict = GP_corr_cols_dict
        self.GP_corr_cols = self._get_GP_corr_cols()
        # GP corrections dimension check
        for i in range(len(self.turbines_cols)):
            self._GP_dimension_check(self.tuning_dimensions_cols[i],
                                     self.GP_corr_cols[i])
        # Grouping info
        self.tuning_groups_cols = self.tuning_variables_cols
        self.tuning_groups = [x for y in self.tuning_groups_cols for x in y]
        self.grouping_bool = True

    @property
    def tuning_turbines(self):
        """List of the tuning turbines in the wind farm."""
        return [x for sublist in self.tuning_groups for x in sublist]

    def wso_compatibility_check(self, wso_obj):
        """
        Check compatibility with a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object to which dynamic parameter tuning is added.
        """
        self._tuning_turbines_check(wso_obj, self.tuning_turbines)
        self._tuning_groups_check(wso_obj)
        self._turbines_cols_check(wso_obj)

    def tune_parameter(self, wso_obj, yaw_angles):
        """
        Perform parameter tuning in a WSOpt object.

        Args
        ----
        wso_obj: (WSOpt) WSOpt object.
        yaw_angles: (np.ndarray) yaw angles of all turbines in the wind farm.

        Returns
        -------
        wf-model_tuned: (WfModel) tuned WfModel to use in the current iteration of the
        wake steering optimisation.
        """
        # Extract WSOpt WfModel dictionary and default parameter
        wf_model_dict = floris_extract_object_dict(wso_obj.wf_model)
        default_parameter = floris_extract_parameter(wso_obj.wf_model_dict_original,
                                                     self.param_class,
                                                     self.param_name)

        # Create tuned parameter list
        opt_param_list = [0]*wso_obj.wf_model.n_turbs
        for i, tuning_variables in enumerate(self.tuning_variables_cols):
            # If no group to tune in the column, assign default non-tuned value
            if len(tuning_variables) == 0:
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           default_parameter)
            # Tune parameter for the each column
            else:
                GP_input = self._get_GP_input_groups(
                    tuning_variables, yaw_angles)
                GP_model = self.GP_model
                GP_corr = self.GP_corr_cols[i]
                # Corrector prediction
                mu_corr, var_corr = GP_corr.predict_noiseless(
                    np.array([GP_input]))
                correction = mu_corr[0][0]
                # Add missing dimensions to GP input if required. Set to zero.
                dim_missing = (GP_model.input_dim -
                               self.tuning_dimensions_cols[i])
                for _ in range(dim_missing):
                    GP_input = np.append(GP_input, float(0))
                mu, var, = GP_model.predict_noiseless(np.array([GP_input]))
                optimal_parameter = mu[0][0]+correction
                opt_param_list = self._fill_opt_param_list(opt_param_list,
                                                           self.turbines_cols[i],
                                                           optimal_parameter)
        # Apply tuned parameter list
        wf_model_dict_tuned = floris_param_change_object_dict(wf_model_dict,
                                                              self.param_class,
                                                              self.param_name,
                                                              opt_param_list)
        wf_model_tuned = floris_param_change_object(wso_obj.wf_model,
                                                    wf_model_dict_tuned)
        return wf_model_tuned
