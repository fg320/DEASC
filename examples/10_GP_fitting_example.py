import numpy as np
import matplotlib.pyplot as plt

from deasc import GPWrap
from deasc.utils import yaw_permutations

"""
This example shows an example of constructing a 3-dimensional Gaussian Process
given a database on a sample function.
"""

# Yaw permuations for a 3-turbine column.
yaw_data = yaw_permutations(dimensions=3,
                            yaw_per_sweep=7,
                            yaw_bounds=[-25, 25])

# Sample function to determine the illustrative optimal parameter


def multimodal_function_3d(x):
    y = np.sin(0.1 * x[0]) + np.sin(0.3 * x[1]) + np.sin(0.5 * x[2])
    return y


# Create tuning database
param_data = []
for x in yaw_data:
    y = multimodal_function_3d(x)
    param_data.append([y])

# Create GPWrap Object
GP_obj = GPWrap(parameter_class='example',
                parameter_name='k',
                dimensions=3)

# Create and plot GP model
GP_model = GP_obj.GP_so(yaw_data, param_data, num_restarts=50, noise=0.05)
GP_obj.GP_so_plot(parameter_range_plot=[-3, 3], yaw_range_plot=[-25, 25])
