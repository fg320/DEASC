# Copyright 2023 Filippo Gori

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Set of plotting functions for FLORIS V3.4 framework #

import matplotlib.pyplot as plt


def floris_visualize_layout(wf_model, ax=None, title="", radius=1.5):
    """FLORIS V3.4 plotting function for layout."""
    # Get farm layout
    x_coordinates = wf_model.interface.get_turbine_layout()[0]
    y_coordinates = wf_model.interface.get_turbine_layout()[1]

    if not ax:
        fig, ax = plt.subplots()

    # Scatter plot with turbine naming convention
    ax.scatter(x_coordinates/wf_model.D, y_coordinates/wf_model.D, s=0)
    for coord_idx in range(len(x_coordinates)):
        circ = plt.Circle((x_coordinates[coord_idx]/wf_model.D,
                           y_coordinates[coord_idx]/wf_model.D),
                          radius=radius, color='lightgray', fill=True)
        ax.add_patch(circ)
        string = f"{(coord_idx+1):d}"
        ax.text(x_coordinates[coord_idx]/wf_model.D,
                y_coordinates[coord_idx]/wf_model.D,
                string, fontsize=8, ha='center', color='k')

    ax.set_title(title, fontsize=10, loc='left')
    ax.set_xlabel("$xD^{-1}$", fontsize=10)
    ax.set_ylabel("$yD^{-1}$", fontsize=10)


def floris_visualize_cut_plane(
    cut_plane,
    ax=None,
    vel_component='u',
    min_speed=None,
    max_speed=None,
    cmap="coolwarm",
    levels=None,
    clevels=None,
    color_bar=False,
    title="",
    **kwargs
):
    """FLORIS V3.4 modified plotting function for flow field."""
    if not ax:
        fig, ax = plt.subplots()

    if vel_component == 'u':
        # vel_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1],
        #                                          cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.u.min()
        if max_speed is None:
            max_speed = cut_plane.df.u.max()
    elif vel_component == 'v':
        # vel_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1],
        #                                          cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.v.min()
        if max_speed is None:
            max_speed = cut_plane.df.v.max()
    elif vel_component == 'w':
        # vel_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1],
        #                                          cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.w.min()
        if max_speed is None:
            max_speed = cut_plane.df.w.max()

    # Allow separate number of levels for tricontourf and for line_contour
    if clevels is None:
        clevels = levels

    # Plot the cut-through
    im = ax.tricontourf(
        cut_plane.df.x1,
        cut_plane.df.x2,
        cut_plane.df.u,
        vmin=min_speed,
        vmax=max_speed,
        levels=clevels,
        cmap=cmap,
        extend="both",
    )

    if cut_plane.normal_vector == "x":
        ax.invert_xaxis()

    if color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('m/s')

    # Set the title
    ax.set_title(title)

    # Make equal axis
    ax.set_aspect("equal")

    return im
