import numpy as np
import matplotlib.pyplot as plt

from deasc import WfModel
from deasc.visualisation import obj_yaw_sweep_1var_plot

"""
This example initialises a 3x3 wind farm of NREL 5 MW turbines with an
atmopsheric condition, and calculates the change in power for a yaw sweep of
the first, most upstream row of turbines
"""

# Input file definition
path = "./inputs/"
input_file = "gch.yaml"

# Initialise wind farm model
wf_model = WfModel(input_file, path)

# Change wind farm layout
n_row = 3
n_col = 3
spac_x = 7
spac_y = 5
wf_model.set_aligned_layout(n_row, n_col, spac_x, spac_y)

# Evaluate wind farm for given condition
yaw = np.full(shape=(n_row*n_col), fill_value=0.0)
_, _, _, _ = wf_model.farm_eval(yaw,
                                ws=8,
                                wd=270,
                                ti=0.05,
                                shear=0.0)

# Yaw sweep of the most upstream turbine row, Row 1, from -25 deg to 25 deg yaw
layout = (n_row, n_col)
var_info = ("R", 1, np.linspace(-25, 25, 51))
decorated = obj_yaw_sweep_1var_plot(wf_model.pow_yaw_sweep_1var)
obj_out, var_info = decorated(layout, var_info)

plt.show()
