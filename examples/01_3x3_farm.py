import numpy as np
import matplotlib.pyplot as plt

from deasc import WfModel
from deasc.utils_floris import floris_get_hor_plane_hub
from deasc.visualisation_floris import (
    floris_visualize_layout,
    floris_visualize_cut_plane
)

"""
This example initialises a 3x3 wind farm of NREL 5 MW turbines, calculates the
power of the farm given an atmopsheric condition, and plots the farm layout and
resulting flow field.
"""

# Input file definition
path = "./inputs/"
input_file = "jensen.yaml"

# Initialise wind farm model
wf_model = WfModel(input_file, path)

# Change wind farm layout
n_row = 3
n_col = 3
spac_x = 7  # Rotor diameters
spac_y = 5  # Rotor diameters
wf_model.set_aligned_layout(n_row, n_col, spac_x, spac_y)

# Specifiy atmopheric and operating conditions
ws = 8  # m/s
wd = 270  # deg
ti = 0.05
shear = 0.0
yaw = np.full(shape=(n_row*n_col), fill_value=25.0)

# Evaluate wind farm and turbines power
wf_pow, wt_pow, wt_ti, wt_yaw = wf_model.farm_eval(yaw,
                                                   ws,
                                                   wd,
                                                   ti,
                                                   shear)
print("Wind farm power %.3f MW" % (wf_pow))

# Visualisation layout
fig, ax = plt.subplots(constrained_layout=True)
floris_visualize_layout(wf_model, ax=ax, radius=1)

# Visualisation flow field
fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
u_max, u_min = (8.0, 1.0)
hor_plane = floris_get_hor_plane_hub(wf_model, yaw)
floris_visualize_cut_plane(hor_plane,
                           ax=ax,
                           vel_component='u',
                           cmap="coolwarm",
                           levels=None,
                           color_bar=False)
