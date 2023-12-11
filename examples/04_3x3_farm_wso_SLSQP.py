import numpy as np

from deasc import WfModel
from deasc import WSOpt

"""
This example shows wake steering optimisation on a 3x3 wind farm of NREL 5 MW turbines.
The initial conditions are 0 deg for all wind turbines. The optimisation variables are
all turbines, except the last, most downstream row. The optimiser is SLSQP with the
default settings.
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

# Specify atmopheric conditions
ws = 8.0
wd = 270
ti = 0.05
shear = 0.0

# Wake steering optimisation inputs
yaw_initial = np.full(shape=(n_row*n_col), fill_value=0)
inflow = (yaw_initial, wd, ws, ti, shear)
variables = [1, 2, 3, 4, 5, 6]
var_bounds = (-25, 25)
var_initial = np.full(shape=(len(variables)), fill_value=0)

# Initialise optimisation object
wso_obj = WSOpt(wf_model=wf_model,
                inflow=inflow,
                variables=variables,
                var_bounds=var_bounds,
                var_initial=var_initial,
                opt_method="SLSQP",
                opt_options=None,
                obj_function="Farm Power",
                constraints=(None, None, None),
                by_row=(False, None, None),
                tuning_dynamic=False
                )

# Optimise, extract and print optimal yaw angles
opt_yaw_angles_vars, opt_yaw_angles_all = wso_obj.optimize_yaw()
print('Optimal farm yaw angles:')
print(opt_yaw_angles_all)

# Get optimisation details and print number of iterations and evaluations
iter_details, eval_details = wso_obj.get_optimization_details()
print('Number of optimiser iterations: %i' % (len(iter_details[0])))
print('Number of objective function evaluations: %i' % (len(eval_details[0])))
