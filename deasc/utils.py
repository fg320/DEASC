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
import itertools


def norm(val, x1, x2):
    return (val - x1) / (x2 - x1)


def unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1


def yaw_permutations(dimensions, yaw_per_sweep, yaw_bounds):
    """
    Generate all possible yaw permutations for the given dimensions.

    Args
    ----
    dimensions: int
        number of dimensions (turbines/groups/etc.).
    yaw_per_sweep: int
        number of equally-spaced yaw angles in the given range for each dimension.
    yaw-bounds: tuple
        upper and lower limit of the yaw sweep shared for all dimensions.

    Returns
    -------
    unique_combinations: list of lists
        list of all possible yaw combinations.
    """
    yaw_list_single = np.linspace(yaw_bounds[0], yaw_bounds[1], yaw_per_sweep).tolist()
    yaw_list = [yaw_list_single for _ in range(dimensions)]
    unique_combinations = list(itertools.product(*yaw_list))
    return unique_combinations


def yaw_permutations_0last(dimensions, yaw_per_sweep, yaw_bounds, n_0last):
    """
    Generate all possible yaw permutations for the given dimensions and adds "n"
    additional dimensions at the end which are held at zero.

    Args
    ----
    dimensions: int
        number of dimensions (turbines/groups/etc.), without counting additional ones
        held at zero.
    yaw_per_sweep: int
        number of equally-spaced yaw angles in the given range for each dimension.
    yaw-bounds: tuple
        upper and lower limit of the yaw sweep shared for all dimensions.
    n_0last: int
        additional dimensions to keep fixed at 0. These are added to the end.

    Returns
    -------
    unique_combinations: list of lists
        list of all possible yaw combinations with some last dimensions held at 0.
    """
    yaw_list_single = np.linspace(yaw_bounds[0], yaw_bounds[1], yaw_per_sweep).tolist()
    yaw_list = [yaw_list_single for _ in range(dimensions)]
    unique_combinations_ = list(itertools.product(*yaw_list))
    yaw_last = [0]*n_0last
    unique_combinations = [list(item)+yaw_last for item in unique_combinations_]
    return unique_combinations


def yaw_permutations_grouping(groups, yaw_per_sweep, yaw_bounds, dims_per_groups):
    """
    Generate all possible yaw permutations for the groups of dimensions, where
    dimensions in the same groups are forced equal.

    Args
    ----
    groups: int
        number of groups.
    yaw_per_sweep: int
        number of equally-spaced yaw angles in the given range for each dimension.
    yaw-bounds: tuple
        upper and lower limit of the yaw sweep shared for all dimensions.
    groups: list of integers
        number of equal dimensions per group.

    Returns
    -------
    unique_combinations: list of lists
        list of all possible yaw combinations.
    """
    yaw_list_single = np.linspace(yaw_bounds[0], yaw_bounds[1], yaw_per_sweep).tolist()
    yaw_list = [yaw_list_single for _ in range(groups)]
    unique_combinations_ = list(itertools.product(*yaw_list))
    unique_combinations = []
    for combination_ in unique_combinations_:
        combination = []
        for i, group in enumerate(combination_):
            combination += [group]*dims_per_groups[i]
        unique_combinations.append(combination)
    return unique_combinations


def yaw_permutations_grouping_0last(groups,
                                    yaw_per_sweep,
                                    yaw_bounds,
                                    dims_per_groups,
                                    n_0last):
    """
    Generate all possible yaw permutations for the groups of dimensions, where
    dimensions in the same groups are forced equal, and adds "n" additional dimensions
    at the end which are held at zero.

    Args
    ----
    groups: int
        number of groups.
    yaw_per_sweep: int
        number of equally-spaced yaw angles in the given range for each dimension.
    yaw-bounds: tuple
        upper and lower limit of the yaw sweep shared for all dimensions.
    groups: list of integers
        number of equal dimensions per group.
    n_0last: int
        additional dimensions to keep fixed at 0. These are added to the end.

    Returns
    -------
    unique_combinations: list of lists
        list of all possible yaw combinations.
    """
    yaw_list_single = np.linspace(yaw_bounds[0], yaw_bounds[1], yaw_per_sweep).tolist()
    yaw_list = [yaw_list_single for _ in range(groups)]
    unique_combinations_ = list(itertools.product(*yaw_list))
    yaw_last = [0]*n_0last
    unique_combinations = []
    for combination_ in unique_combinations_:
        combination = []
        for i, group in enumerate(combination_):
            combination += [group]*dims_per_groups[i]
        combination += yaw_last
        unique_combinations.append(combination)
    return unique_combinations
