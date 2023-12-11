# Copyright 2023 Filippo Gori

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import warnings
import numpy as np

from .utils_floris import (
    floris_input_handler,
    floris_properties,
    floris_current_yaw,
    floris_reinitialise_layout,
    floris_farm_eval
)


class WfModel:
    """
    Class for wind farm modelling (Interface setup but not limited to FLORIS
    framework).
    """

    def __init__(self, input_file, path):
        """
        Initialise wind farm object by pointing towards an input file.
        (FLORIS interface object).

        Args
        ----
        input file:(FLORIS .json input file).
        """
        # Read and initialize input file
        self.input_file = input_file
        self.interface = floris_input_handler(self.input_file, path)

        # Assign wind farm model proporties
        self.D, self.H_hub, self.n_turbs = floris_properties(self)

    def set_aligned_layout(self, n_row, n_col, spac_x, spac_y, coordinates=False):
        """
        Modify farm layout in aligned wind turbines with constant spacing,
        differing only from rows to columns. Flow field is also reinitialized.

        Args
        ----
        n_row: (float) number of turbine rows
        n_col: (float) number of turbine columns
        spac_x: (float) WT diam normalized turbines distance in x direction
        spac_y: (float) WT diam normalized turbines distance in y direction
        coordinates: (bool, opt) False if no coordinates wanted.
            Default set to False.

        Returns
        -------
        if coordinates is False:
            None
        if coordinates is True:
            x-coordinates: (numpy array) turbines x-coordinates
            y-coordinates: (numpy array) turbines y-coordinates
        """
        # Input type check
        if not all(isinstance(i, int) for i in [n_row, n_col]) or \
                not all(isinstance(j, (int, float)) for j in [spac_x, spac_y]):
            err_msg = "Incorrect input value types"
            raise ValueError(err_msg)

        # Calculate new coordinate farm layout
        layout_x = []
        layout_y = []
        for i in range(int(n_row)):
            for j in range(int(n_col)):
                layout_x.append(i * spac_x * self.D)
                layout_y.append(j * spac_y * self.D)

        # Reinitialize wind farm object
        floris_reinitialise_layout(self, layout_x, layout_y)

        if coordinates:
            return (np.array(layout_x), np.array(layout_y))
        else:
            return None

    def set_HR_layout(self, coordinates=False):
        """
        Set Horns Rev wind farm layout to wind farm object and
        returns turbines' x and y coordinates if coordinates=True.

        Args
        ----
        coordinates: (bool, opt) False if no coordinates wanted.
              Default set to False.

        Returns
        -------
        if coordinates is False:
            None
        if coordinates is True:
            x-coordinates: (numpy array) turbines x-coordinates
            y-coordinates: (numpy array) turbines y-coordinates
        """
        # Vestas V80 2 MW diameter check
        if self.D != 80:
            warning = "Rotor diameter not from the Vestas V80 2 MW turbine"
            warnings.warn(warning, UserWarning)

        n_rows = 10
        n_cols = 8
        spac_x = 7
        spac_y = 7
        angle = 6
        layout_x = []
        layout_y = []
        for i in range(int(n_rows)):
            for j in range(int(n_cols)):
                layout_x.append((i * spac_x * self.D) -
                                (np.sin(np.radians(angle)) * j * spac_y * self.D))
                layout_y.append(j * spac_y * self.D * np.cos(np.radians(angle)))

        # Reinitialize wind farm object
        floris_reinitialise_layout(self, layout_x, layout_y)

        if coordinates:
            return (np.array(layout_x), np.array(layout_y))
        else:
            return None

    def farm_eval(self, yaw=None, ws=None, wd=None, ti=None, shear=None):
        """
        Calculate farm flow field for given wind farm layout and input conditions.
        Return main outputs, such as yaw angles, turbines power, farm power, etc.

        Args
        ----
        yaw: (list, optional) turbines yaw angles (deg). Default to None.
        ws: (float, optional) input wind speeds (m/s). Default to None.
        wd: (float, optional) input wind directions (deg). Default to None.
        ti: (float, optional) input turbulence intensity. Default to None.
        shear: (float, optional) shear exponent. Default to None.

        Returns
        -------
        wf_pow: (float) WF power (MWatts).
        wt_pow: (np.array) WTs power (MWatts).
        wt_ti: (list) WTs turbulence intensity.
        wt_yaw: (np.array) WTs yaw angles (deg).
        """
        # Main wind farm calculation
        wf_pow, wt_pow, wt_ti, wt_yaw, _ = floris_farm_eval(self,
                                                            yaw,
                                                            ws,
                                                            wd,
                                                            ti,
                                                            shear)

        return (wf_pow, wt_pow, wt_ti, wt_yaw)

    def pow_yaw_sweep_1var(self, layout, var_info):
        """
        Return wind farm power for a single yaw variable, either a
        single turbine or a single row of turbines. Sweep by row not possible
        for not aligned "custom" layouts.

        Args
        ----
        layout: (tuple)
            row: (integer) number of farm rows
            cols: (integer) number of farm columns
            or string "custom"
        var_info: (tuple)
            var_type: (string) "T" for turbine,
                               "R" for row (not for custom layouts)
            var: (integer) turbine or row number
            var_value: (list of floats) variable values

        Returns
        -------
        obj_out: tuple
            obj: (list) objective values
            obj_func: (string) objective function
        var_info: (tuple) see input
        model: (string) model name
        """
        # Extract inputs and check inputs
        var_type, var, var_value = var_info
        if layout != "custom":
            rows, cols = layout
        if var_type == 'R' and layout == "custom":
            err_msg = "Row not allowed for custom layouts"
            raise ValueError(err_msg)
        if var_type == 'R' and var > rows:
            err_msg = "Row specified not in farm"
            raise ValueError(err_msg)
        if var_type == 'T' and var > self.n_turbs:
            err_msg = "Turbine specified not in farm"
            raise ValueError(err_msg)

        # Calculations
        yaw_angles = np.array(floris_current_yaw(self))
        wf_pow = []

        for yaw_change in var_value:
            if layout != "custom":
                rows, cols = layout
            if var_type == 'T':
                yaw_angles[(var-1)] = yaw_change
            elif var_type == 'R':
                idx_1 = var*cols
                idx_0 = idx_1-cols
                yaw_angles[idx_0:idx_1] = yaw_change
            else:
                err_msg = "var_type either 'T' or 'R'"
                raise ValueError(err_msg)

            wf_pow_single, _, _, _ = self.farm_eval(yaw=yaw_angles)
            wf_pow.append(wf_pow_single)

        obj_out = (wf_pow, 'Farm Power')
        var_info = (var_type, var, var_value)
        print("Function exploration complete")

        return obj_out, var_info
