import numpy as np

from deasc import WfModel
from deasc import WSOpt

"""
This example shows constrained wake steering optimisation on a 3x3 wind farm of NREL
5 MW turbines for random initial conditions.The optimisation variables are all turbines.
The first constrained applied is resctricting the range to the positive domain. Second,
the set of linear constraints applied enforces a row-monotonic decrease in optimal yaw
settings with downstream distance. The optimiser is SLSQP with the default settings.
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
variables = [1, 2, 3, 4, 5, 6, 7, 8, 9]
var_bounds = (0, 25)
var_initial = 'random'

# Constraints
A = np.zeros((len(variables)-n_col, len(variables)), dtype=float)
for i in range(len(variables)-n_col):
    A[i, i] = 1
    A[i, i+n_col] = -1
low_bound_constr = 0
upp_bound_constr = np.inf
constraint = (A, low_bound_constr, upp_bound_constr)

# Initialise optimisation object
wso_obj = WSOpt(wf_model=wf_model,
                inflow=inflow,
                variables=variables,
                var_bounds=var_bounds,
                var_initial=var_initial,
                opt_method="SLSQP",
                opt_options=None,
                obj_function="Farm Power",
                constraints=constraint,
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
