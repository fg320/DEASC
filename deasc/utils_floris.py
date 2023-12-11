# Copyright 2023 Filippo Gori

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Set of functions to interface with FLORIS V3.4 framework #

import time
import numpy as np
import copy
from floris.tools import FlorisInterface as FI


def floris_input_handler(input_file, path):
    """Convert input file into a FLORIS interface object."""
    # No input file
    if input_file == None:
        err_msg = "Input file required"
        raise Exception(err_msg)

    # Multiple input files
    elif isinstance(input_file, list) is True:
        err_msg = "Required a single input file, multiple are provided"
        raise Exception(err_msg)

    # Initialize single floris object
    else:
        fi = FI(path+input_file)
        print("Successfull single file import!")

    return fi


def floris_properties(wf_model):
    """Extract wind farm model information from FLORIS object."""
    fi = wf_model.interface
    D = (fi.floris.farm.rotor_diameters).flatten()[0]  # Flatten over wd and ws
    H_hub = (fi.floris.farm.hub_heights).flatten()[0]  # Flatten over wd and ws
    n_turbs = len(wf_model.interface.get_turbine_layout()[0])

    return D, H_hub, n_turbs


def floris_reinitialise_layout(wf_model, layout_x, layout_y):
    """
    Modify wind farm layout based on the coordinates provided and also
    reinitialises the flow field. If the number of turbines is unchanged,
    yaw angles are stored in the farm object. Limited to a FLORIS interface
    object.
    """
    # Extract FLORIS interface object
    fi = wf_model.interface

    # As floris reinitializes, it sets all yaw angles to 0 deg.
    yaw_temp = fi.floris.farm.yaw_angles[0][0]
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
    # Update number of turbines
    wf_model.n_turbs = len(wf_model.interface.get_turbine_layout()[0])
    # Keep old yaw angles only if the number of turbines is the same
    if len(yaw_temp) == len(layout_x):
        fi.calculate_wake(yaw_angles=np.array([[yaw_temp]]))


def floris_reinitialise_atmosphere(wf_model, ws, wd, ti, shear):
    """Modify atmopheric conditions and return FLORIS object, one ws and wd."""
    wf_model.interface.reinitialize(wind_speeds=[ws],
                                    wind_directions=[wd],
                                    turbulence_intensity=ti,
                                    wind_shear=shear)
    return wf_model


def floris_calculate_turbine_power(wf_model, yaw_angles):
    """Calculate and returns power of turbines for given yaw angles, one ws and wd."""
    wf_model.interface.calculate_wake(yaw_angles=np.array([[yaw_angles]]))
    power_turbines = (np.array(wf_model.interface.get_turbine_powers())*10**(-6))[0][0]
    return power_turbines


def floris_calculate_farm_power(wf_model, yaw_angles):
    """Calculate and returns wind farm power for given yaw angles, one ws and wd."""
    wf_model.interface.calculate_wake(yaw_angles=np.array([[yaw_angles]]))
    power_farm = (np.array(wf_model.interface.get_farm_power())*10**(-6))[0][0]
    return power_farm


def floris_farm_eval(wf_model, yaw, ws, wd, ti, shear):
    """
    Calculate wind farm power and wind turbine powers given an
    atmopheric and yaw condition. Farm layout is unchanged and information such
    as yaw angles preserved even if not explicitly specified.
    """
    # Extract FLORIS interface object
    fi = wf_model.interface

    # Start floris farm computational time
    start = time.time()

    # Get yaw angles before reinitializing - set to 0 when reinitializing flow
    yaw = fi.floris.farm.yaw_angles[0][0] if yaw is None else yaw

    # Get wd and ws as None is not an option in reinitialize flow
    ws = fi.floris.flow_field.wind_speeds[0] if ws is None else ws
    wd = fi.floris.flow_field.wind_directions[0] if wd is None else wd

    # Error if yaw angles don't match turbine number
    if len(yaw) != len(fi.floris.farm.yaw_angles[0][0]):
        err_msg = "Yaw prescribed not matching turbine number"
        raise Exception(err_msg)

    # Reinitialize flow field and set previous yaw angles
    fi.reinitialize(wind_speeds=[ws],
                    wind_directions=[wd],
                    turbulence_intensity=ti,
                    wind_shear=shear)
    yaw = np.array([float(item) for item in yaw])
    fi.calculate_wake(yaw_angles=np.array([[yaw]]))

    # Calculate wf power, wt powers, wt turbulence intensities, wt yaw angles
    wf_pow = (fi.get_farm_power()*10**(-6))[0][0]
    wt_pow = (np.array(fi.get_turbine_powers())*10**(-6))[0][0]
    wt_ti = (fi.get_turbine_TIs())[0][0]
    wt_yaw = np.array(fi.floris.farm.yaw_angles[0][0])

    # Report CPU time
    cpu_time = time.time()-start
    return (wf_pow, wt_pow, wt_ti, wt_yaw, cpu_time)


def floris_current_yaw(wf_model):
    """Extract and returns the current wind farm yaw angles."""
    return wf_model.interface.floris.farm.yaw_angles[0][0]


def floris_get_hor_plane_hub(wf_model, yaw):
    """Extract horizontal plane of streamwise velocity deficit at hub height."""
    hor_plane = wf_model.interface.calculate_horizontal_plane(height=wf_model.H_hub,
                                                              yaw_angles=(np.array([[yaw]])))
    return hor_plane

# %% Parameter FLORIS functions


def floris_extract_object_dict(wf_model):
    """Extract and return the current FLORIS object dictionary."""
    return wf_model.interface.floris.as_dict()


def floris_extract_models_dict(wf_model_dict):
    """Extract and return the current FLORIS models dictionary."""
    models_dict = {'wake_velocity_parameters':
                   wf_model_dict['wake']['model_strings']['velocity_model'],
                   'wake_deflection_parameters':
                   wf_model_dict['wake']['model_strings']['deflection_model'],
                   'wake_turbulence_parameters':
                   wf_model_dict['wake']['model_strings']['turbulence_model']}
    return models_dict


def floris_print_params(wf_model_dict, models_dict, title):
    """Print the current FLORIS parameters."""
    print("=====================================================")
    print(title)
    print("Wake_velocity_parameters")
    print(wf_model_dict['wake']['wake_velocity_parameters']
          [models_dict['wake_velocity_parameters']])
    print("Wake_deflection_parameters")
    print(wf_model_dict['wake']['wake_deflection_parameters']
          [models_dict['wake_deflection_parameters']])
    print("Wake_turbulence_parameters")
    print(wf_model_dict['wake']['wake_turbulence_parameters']
          [models_dict['wake_turbulence_parameters']])
    print("=====================================================")


def floris_extract_parameter(wf_model_dict, param_class, param_name):
    """Extract and return the current parameter value of a FLORIS object parameter."""
    models_dict = floris_extract_models_dict(wf_model_dict)
    return wf_model_dict['wake'][param_class][models_dict[param_class]][param_name]


def floris_param_change_object_dict(wf_model_dict, param_class, param_name, param_value):
    """
    Change FLORIS object with a new model parameter, return new FLORIS object dictionary.
    FLORIS object is not reinitialised (see function floris_parameter_change_object).
    """
    wf_model_dict_new = copy.deepcopy(wf_model_dict)
    models_dict = floris_extract_models_dict(wf_model_dict_new)
    (wf_model_dict_new['wake'][param_class]
     [models_dict[param_class]][param_name]) = param_value
    return wf_model_dict_new


def floris_param_change_object(wf_model, wf_model_dict_new):
    """Change FLORIS object with new object dictionary. Also reinitialise farm layout."""
    x_reinit, y_reinit = wf_model.interface.get_turbine_layout()
    wf_model.interface = FI(wf_model_dict_new)
    wf_model.interface.reinitialize(layout_x=x_reinit, layout_y=y_reinit)
    return wf_model
