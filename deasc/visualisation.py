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
    Plot any objective function over a yaw range for either a farm
    row or single turbine. See WfModel.pow_yaw_sweep_1var for function input.

    Function input requirements / Returns
    -------------------------------------
    obj_out: (tuple)
        obj: (list) objective values.
        obj_func: (string) objective function.
    var_info: (tuple)
        var_type: (string) "T" for turbine, "R" for row.
        var: (integer) turbine or row number
        (turbine counting convention: starting from 1,
         row-all columns-next row, etc.).
        var_value: (list of floats) variable values.
    """
    def decorated(*args, **kwargs):
        plt.figure(figsize=(3, 2.5), tight_layout=True)
        (obj, obj_func), (var_type, var, var_value) = function(*args, **kwargs)
        plt.plot(var_value, obj, '-', color='tab:blue')
        plt.title(r"%s for %s%i yaw" % (obj_func, var_type, var), fontsize=9)
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
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(np.arange(lb, ub, 1))+(ub+1)))
    ax.scatter(x_coordinates/wf_model.D, y_coordinates/wf_model.D, s=0)
    for coord_idx in range(len(x_coordinates)):
        # Coloured patch
        yaw_single = wso_obj.opt_yaw_angles_all[coord_idx]
        color = colors[int(yaw_single)+40]
        circ = plt.Circle((x_coordinates[coord_idx]/wf_model.D,
                           y_coordinates[coord_idx]/wf_model.D),
                          radius=radius, color=color, fill=True)
        ax.add_patch(circ)
        # Yaw angle as text
        string = f"{(round(yaw_single)):d}"
        ax.text(x_coordinates[coord_idx]/wf_model.D,
                y_coordinates[coord_idx]/wf_model.D,
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

    # Get hub height streamwise velocity field
    _ = floris_farm_eval(wso_obj.wf_model,
                         wso_obj.opt_yaw_angles_all,
                         wso_obj.ws,
                         wso_obj.wd,
                         wso_obj.ti,
                         wso_obj.shear)
    hor_plane = floris_get_hor_plane_hub(wso_obj.wf_model, wso_obj.opt_yaw_angles_all)

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


def wso_explore_optimum_power_1var(wso_obj, turbine, yaw_bounds, yaw_number):
    """
    Plot the power function for the yaw sweep of a single turbine within the farm,
    having the wake steering optimal yaw angles as initial condition.

    Args
    ----
    wso_obj: (WSOpt) WSOpt object with method optimize_yaw run.
    turbine: (integer) turbine to sweep yaw angle.
    yaw_bounds: (tuple) yaw bounds for yaw sweep.
    yaw_number: (integer): number of yaw angles withing the specified yaw bounds.
    """
    # Check if optimisation is run
    if wso_obj.opt_run is False:
        err_msg = "Wake steering optimisation not run. See optimize_yaw method"
        raise Exception(err_msg)
    # Check function input
    _wso_explore_optimum_input_handler(wso_obj, turbine)
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
    decorated("custom", ("T", turbine, yaw_sweep))
    # Add optimum
    plt.plot(wso_obj.opt_yaw_angles_all[turbine-1],
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
    ax.axhline(1, 0, 1, color='tab:red', linestyle='--', label='no wake steering')
    ax.set_xlabel("%s" % (text))
    ax.set_ylabel("Objective function")
    ax.set_title("%s details" % (text))
    ax.legend(loc='best')


def _wso_explore_optimum_input_handler(wso_obj, turbine):
    # Check input
    if isinstance(turbine, (int, float)) is False:
        err_msg = "Only a single turbine can be specified"
        raise Exception(err_msg)
    if turbine > wso_obj.wf_model.n_turbs:
        err_msg = "Turbine specified not in the farm"
        raise Exception(err_msg)
    if turbine == 0:
        err_msg = "Turbine counting convention starts from 1"
        raise Exception(err_msg)
