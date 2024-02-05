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
import matplotlib.pyplot as plt
from deasc.utils_floris import (
    floris_get_hor_plane_hub,
    floris_farm_eval
)
from deasc.visualisation_floris import (floris_visualize_cut_plane)


# %% WfModel plotting

def obj_yaw_sweep_1var_plot(function):
    """
    Plot any objective function over a yaw range for a farm
    turbine, group, or row. See WfModel.pow_yaw_sweep_1var for function input.

    Function input requirements / Returns
    -------------------------------------
    obj_out: (tuple)
        obj: (list) objective values.
        obj_func: (string) objective function.
    var_info: (tuple)
        var_type: (string) "T" for turbine, "G" for group, "R" for row.
        var: 
            (integer) turbine or row number
            or
            (list of integers) list of turbines in the group
            (turbine counting convention: starting from 1,
             row-all columns-next row, etc.)
        var_value: (list of floats) variable values.
    """
    def decorated(*args, **kwargs):
        plt.figure(figsize=(3, 2.5), tight_layout=True)
        (obj, obj_func), (var_type, var, var_value) = function(*args, **kwargs)
        plt.plot(var_value, obj, '-', color='tab:blue')
        if var_type == 'G':
            title_group = " ".join(["T%i" % item for item in var])
            plt.title(r"%s for %s (%s) yaw"
                      % (obj_func, var_type, title_group), fontsize=9)
            plt.xlabel(r"$\gamma_{%s}$ [deg]" % (var_type), fontsize=8)
        else:
            plt.title(r"%s for %s%i yaw" %
                      (obj_func, var_type, var), fontsize=9)
            plt.xlabel(r"$\gamma_{%s%i}$ [deg]" % (var_type, var), fontsize=8)
        if obj_func == 'Farm Power':
            plt.ylabel(r'$P_{WF}$ [MW]', fontsize=8)
        else:
            plt.ylabel(r'%s' % (obj_func), fontsize=8)
        plt.tick_params(labelsize=8)
        return (obj, obj_func), (var_type, var, var_value)
    return decorated

# %% WSOpt plotting


def wso_optimal_yaw_angles(wso_obj, radius=1.5, ax=None):
    """
    Plot the the optimal yaw angles for the wake steering optimisation.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    radius: (float) radius of circle around each turbine.
    ax: (:py:class:`matplotlib.pyplot.axes`, optional) Figure axes. Defaults to None.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots()

    # Get farm layout
    wf_model = wso_obj.wf_model
    x_coordinates = wf_model.interface.get_turbine_layout()[0]
    y_coordinates = wf_model.interface.get_turbine_layout()[1]

    # Scatter plot with optimal yaw angles
    lb = wso_obj.low_bound
    ub = wso_obj.upp_bound
    colors = plt.cm.coolwarm(np.linspace(
        0, 1, int(len(np.arange(lb, ub, 1))*1.5)))
    ax.scatter(x_coordinates/wf_model.D, y_coordinates/wf_model.D, s=0)
    for coord_idx in range(len(x_coordinates)):
        # Coloured patch
        yaw_single = wso_obj.opt_yaw_angles_all[coord_idx]
        color = colors[int(yaw_single)+int(len(np.arange(lb, ub, 1))*1.5/2)]
        circ = plt.Circle((x_coordinates[coord_idx]/wf_model.D,
                           y_coordinates[coord_idx]/wf_model.D),
                          radius=radius, color=color, fill=True)
        ax.add_patch(circ)
        # Yaw angle as text
        string = f"{(round(yaw_single)):d}"
        ax.text(x_coordinates[coord_idx]/wf_model.D,
                (y_coordinates[coord_idx]/wf_model.D -
                 max(y_coordinates/wf_model.D)*0.015),
                string, fontsize=8, ha='center', color='k')

    ax.set_title("Optimal yaw angles", fontsize=10)
    ax.set_xlabel("$xD^{-1}$", fontsize=10)
    ax.set_ylabel("$yD^{-1}$", fontsize=10)
    ax.set_aspect("equal")

    return ax


def wso_optimal_flow_field(wso_obj, ax=None):
    """
    Plot the streamwise velocity flow field at hub height for the optimal yaw angles at
    the inflow conditions specified in the optimisation.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    ax: (:py:class:`matplotlib.pyplot.axes`, optional) Figure axes. Defaults to None.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots()

    # Tune parameters
    if wso_obj.tuning_dyn_initialization:
        for tuning_dyn_obj in wso_obj.tuning_dyn_obj_list:
            wso_obj.wf_model = tuning_dyn_obj.tune_parameter(wso_obj,
                                                             wso_obj.opt_yaw_angles_all)

    # Get hub height streamwise velocity field
    _ = floris_farm_eval(wso_obj.wf_model,
                         wso_obj.opt_yaw_angles_all,
                         wso_obj.ws,
                         wso_obj.wd,
                         wso_obj.ti,
                         wso_obj.shear)
    hor_plane = floris_get_hor_plane_hub(
        wso_obj.wf_model, wso_obj.opt_yaw_angles_all)

    # Plot streamwise velocity field
    floris_visualize_cut_plane(hor_plane,
                               ax=ax,
                               vel_component='u',
                               cmap="coolwarm",
                               levels=None,
                               color_bar=True,
                               title='Optimized Yaw')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return ax


def wso_plot_details_iterations(wso_obj, ax=None):
    """
    Plot the optimizer iteration details with the progressive values of the
    objective function.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    ax: (:py:class:`matplotlib.pyplot.axes`, optional) Figure axes. Defaults to None.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot details
    _wso_plot_details(wso_obj, 'iterations', ax)

    return ax


def wso_plot_details_evaluations(wso_obj, ax=None):
    """
    Plot the wind farm evaluations details with the progressive values of the
    objective function.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    ax: (:py:class:`matplotlib.pyplot.axes`, optional) Figure axes. Defaults to None.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot details
    _wso_plot_details(wso_obj, 'evaluations', ax)

    return ax


def wso_explore_optimum_power_1var(wso_obj, var_type, variable, yaw_bounds, yaw_number):
    """
    Plot the power function for the yaw sweep of a single turbine, single 
    group of turbines, or row of turbines, having the wake steering optimal 
    yaw angles as initial condition. Groups and row sweeps require wake steering
    with grouping and by row, respectively, where groups and rows are wake 
    steering variables. When dynamic tuning is in place, not possible to sweep
    by row and to sweep by turbine when wso grouping or by_row.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    var_type: (string) "T" for turbine (only for turbine wso when tuning in place)
                       "G" for group (only for wso grouping=True)
                       "R" for row (only for wso by_row[0]=True)
    variable: (integer) turbine or row to sweep yaw angle.
              or
              (list of integers) list of turbines in the group to sweep yaw angle.
    yaw_bounds: (tuple) yaw bounds for yaw sweep.
    yaw_number: (integer): number of yaw angles withing the specified yaw bounds.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)
    # Check function input
    _wso_explore_optimum_input_handler(wso_obj, var_type, variable)
    # Assign layout
    layout = (wso_obj.rows, wso_obj.cols) if var_type == 'R' else "custom"
    # Run optimal yaw angles solution
    _ = floris_farm_eval(wso_obj.wf_model,
                         wso_obj.opt_yaw_angles_all,
                         wso_obj.ws,
                         wso_obj.wd,
                         wso_obj.ti,
                         wso_obj.shear)
    # Get yaw sweep plot
    yaw_sweep = np.linspace(yaw_bounds[0], yaw_bounds[1], yaw_number)
    decorated = obj_yaw_sweep_1var_plot(wso_obj.wf_model.pow_yaw_sweep_1var)
    decorated(layout, (var_type, variable, yaw_sweep), wso_obj=wso_obj)
    # Add optimum
    if var_type == 'T':
        idx_opt = variable - 1
    elif var_type == 'G':
        idx_opt = variable[0] - 1
    elif var_type == 'R':
        idx_opt = variable*wso_obj.cols - 1
    plt.plot(wso_obj.opt_yaw_angles_all[idx_opt],
             wso_obj.farm_power_opt,
             'or',
             label='Optimum')
    plt.legend(loc='best', fontsize=6, markerscale=0.6)

# %% Private methods


def _wso_plot_details(wso_obj, plotting, ax):
    # Optimiser iterations or farm evaluations details
    if plotting == 'iterations':
        y = wso_obj.iter_obj_func
        text = 'Optimiser iterations'
    elif plotting == 'evaluations':
        y = wso_obj.eval_obj_func
        text = 'Wind farm evaluations'
    x = range(1, len(y)+1)

    # Plotting
    ax.plot(x, -np.array(y), '-o')
    ax.axhline(1, 0, 1, color='tab:red',
               linestyle='--', label='no wake steering')
    ax.set_xlabel("%s" % (text))
    ax.set_ylabel("Objective function")
    ax.set_title("%s details" % (text))
    ax.legend(loc='best')


def _wso_explore_optimum_input_handler(wso_obj, var_type, variable):
    if var_type == 'T':
        # Do not allow turbine sweep for wso grouping when tuning is in place.
        if (wso_obj.grouping_bool and wso_obj.tuning_dyn_initialization):
            err_msg = "Cannot sweep turbines when wso with grouping has dynamic" +\
                " tuning in place."
            raise ValueError(err_msg)
        else:
            pass
    else:
        # Row sweep requires wso by row and vice-versa (unless turbine sweep)
        if ((var_type == 'R') ^ wso_obj.by_row_bool):
            err_msg = "Cannot sweep row if wake steering arg by_row[0]=False " +\
                "and vice versa (unless a turbine sweep)."
            raise ValueError(err_msg)
        # Group sweep requires wso with grouping and vice-versa (unless turbine sweep)
        if ((var_type == 'G') ^ wso_obj.grouping_bool):
            err_msg = "Cannot sweep group if wake steering arg grouping=False " +\
                "and vice versa (unless a turbine sweep without tuning)."
            raise ValueError(err_msg)
        # Check that swept row and group are part of wso variables
        if (var_type == 'R' and variable not in wso_obj.variables):
            err_msg = "Cannot sweep row not included in wso variables."
            raise ValueError(err_msg)
        if (var_type == 'G' and variable not in wso_obj.turbine_groups):
            err_msg = "Cannot sweep group not included in wso variables."
            raise ValueError(err_msg)
