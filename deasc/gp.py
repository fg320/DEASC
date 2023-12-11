# Copyright 2023 Filippo Gori

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import GPy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools


class GPWrap:
    """
    Wrapper class to create, modify and visualise Gaussian Processes for dynamic parameter
    tuning. Currently limited to a single output GP.
    """

    def __init__(self, parameter_class, parameter_name, dimensions):
        self.parameter_class = parameter_class
        self.parameter_name = parameter_name
        self.dimensions = dimensions
        """
        Args
        ----
        parameter_class: string
            Parameter class of the optimal parameter to fit.
        parameter_name: string
            Name of the optimal parameter to fit.
        dimensions: integer
            Dimensions/inputs/variables of the GP.
        """

    def GP_so(self, yaw_data, param_data, num_restarts=50, noise=0.05):
        """
        Construct and returns a single-output (SO) GP for the given input dataset
        (optimal parameter for a given yaw configuration).

        Args
        ----
        yaw_data: list of lists
            list of input yaw configurations for which parameter has been tuned
        param_data: list of lists
            for each yaw configuration in yaw_data, list containing the optimal parameter
        num_restarts: int
            number of random starts of the GP hyperparameter tuning optimization
        noise: float
            noise in output prediction. Default is 0.05

        Returns
        -------
        m: GPy single-output Gaussian Process model
        """
        # Sample check on argument dimension
        if len(yaw_data[0]) != self.dimensions:
            err_msg = ("Yaw input and GP dimensions do not match")
            raise Exception(err_msg)
        if len(param_data[0]) != 1:
            err_msg = ("Single-output GPs only")
            raise Exception(err_msg)

        # Data structure arguments
        yaw_data_GP = np.array(yaw_data)
        param_data_GP = np.array(param_data)

        # GP model
        kernel = GPy.kern.RBF(input_dim=self.dimensions, variance=1., lengthscale=1.)
        self.m = GPy.models.GPRegression(yaw_data_GP,
                                         param_data_GP,
                                         kernel,
                                         noise_var=noise)

        # Hyperparameter tuning
        self.m.optimize(optimizer=None,  # Default lbfgsb
                        start=None,
                        messages=False,
                        max_iters=1000)
        self.m.optimize_restarts(num_restarts=num_restarts)
        return self.m

    def GP_so_plot(self, parameter_range_plot, yaw_range_plot):
        """
        Plot a single-output (SO) GP model. 1D and 2D plots are generated for each
        variable combination.

        Args
        ----
        parameter_range: tuple
            range of the optimal parameter to plot
        parameter_range: tuple
            range of the yaw variables to plot
        """
        # Plotting library choice and defaults values
        GPy.plotting.change_plotting_library('matplotlib')
        GPy.plotting.matplot_dep.defaults.data_2d = {'s': 0,
                                                     'edgecolors': 'none',
                                                     'linewidth': 0.0,
                                                     'cmap': cm.get_cmap('hot'),
                                                     'alpha': 0.5}

        # 1D Plots
        if self.dimensions == 1:
            figure = GPy.plotting.plotting_library().figure(1, 1, figsize=(5, 2.5))
            title = 'GP %s' % (self.parameter_name)
            xlabel = '$\gamma_{1}$ [deg]'
            ylabel = '$%s_{opt}$' % (self.parameter_name)
            fig = self.m.plot(figure=figure,
                              col=1,
                              row=1,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              ylim=list(parameter_range_plot),
                              legend=False,
                              plot_data=True)
        else:
            n_cuts = 3
            slices = np.linspace(yaw_range_plot[0], yaw_range_plot[1], n_cuts)
            figsize = (5*n_cuts, 2.5*self.dimensions)
            figure = GPy.plotting.plotting_library().figure(self.dimensions,
                                                            n_cuts,
                                                            figsize=figsize)

            for dim_idx in range(self.dimensions):
                for i, slice_single in zip(range(n_cuts), slices):
                    title = "GP %s - $\gamma_{others}$" \
                            "%.1f $^{\circ}$" % (self.parameter_name, slice_single)
                    xlabel = '$\gamma_{%i}$ [deg]' % (dim_idx+1)
                    ylabel = '$%s_{opt}$' % (self.parameter_name)
                    inputs = []
                    for j in range(self.dimensions):
                        if j == dim_idx:
                            pass
                        else:
                            inputs.append((j, slice_single))
                    fig = self.m.plot(figure=figure,
                                      col=(i+1),
                                      row=(dim_idx+1),
                                      fixed_inputs=inputs,
                                      title=title,
                                      xlabel=xlabel,
                                      ylabel=ylabel,
                                      ylim=list(parameter_range_plot),
                                      legend=False,
                                      plot_data=False)

        # 2D Plots
        # Countours are fine ##
        # Data points (training) plotted are off ##
        # double checked with GP and training database ##
        if self.dimensions == 1:
            pass
        elif self.dimensions == 2:
            figure = GPy.plotting.plotting_library().figure(1, 1, figsize=(3, 2.5))

            title = 'GP %s' % (self.parameter_name)
            xlabel = '$\gamma_{1}$ [deg]'
            ylabel = '$\gamma_{2}$ [deg]'

            fig = self.m.plot(figure=figure,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              legend=False,
                              plot_data=True)

            ax = plt.gca()
            mappable = ax.collections[0]
            cbar = plt.colorbar(mappable)
            # cbar.set_label('$%s_{opt}$'%(self.parameter_name))
        else:
            n_cuts = 3
            slices = np.linspace(yaw_range_plot[0], yaw_range_plot[1], n_cuts)
            plot_rows = self.dimensions-1
            plot_cols = self.dimensions-1
            combinations = list(itertools.combinations(
                list(range(0, self.dimensions)), 2))

            figsize = (3*plot_cols*len(slices), 2.5*plot_rows)
            figure = GPy.plotting.plotting_library().figure(plot_rows,
                                                            plot_cols*len(slices),
                                                            figsize=figsize)
            for i, slice_single in zip(range(n_cuts), slices):
                for comb_idx, comb in enumerate(combinations):
                    title = 'GP %s - $\gamma_{others}$' \
                            '%.1f $^{\circ}$' % (self.parameter_name, slice_single)
                    xlabel = '$\gamma_{%i}$ [deg]' % (comb[0]+1)
                    ylabel = '$\gamma_{%i}$ [deg]' % (comb[1]+1)
                    inputs = []
                    for j in range(self.dimensions):
                        if j in comb:
                            pass
                        else:
                            inputs.append((j, slice_single))

                    fig = self.m.plot(figure=figure,
                                      col=(comb[0]+1+plot_cols*i),
                                      row=(comb[1]),
                                      fixed_inputs=inputs,
                                      title=title,
                                      xlabel=xlabel,
                                      ylabel=ylabel,
                                      legend=False,
                                      plot_data=True)

                    ax = plt.gca()
                    mappable = ax.collections[0]
                    cbar = plt.colorbar(mappable)
                    # cbar.set_label('$%s_{opt}$'%(self.parameter_name))
