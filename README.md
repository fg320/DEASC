# DEASC
A data-integrated framework for wake steering control of wind farms.

## Overview
Data-integrated wake steering control (DEASC) is a wind farm control-oriented software that performs wake steering optimisation of wind farms while integrating higher-fidelity measurements through wake model parameter tuning. Built around, but not limited to, the wind farm simulator ```FLORIS```, DEASC allows to perform wake steering optimisation of wind farms with various optimisation algorithms, tune wake model parameters, and implement the yaw-dependent Gaussian process-based parameter tuning approach developed by F. Gori, S. Laizet, and A. Wynn in **Wind farm power maximisation via wake steering: a Gaussian process‐based yaw‐dependent parameter tuning approach**. For details on the capabilities of the software, please refer to the extensive ```examples``` folder. 

## Installation
DEASC can be installed by downloading the git repository from GitHub with ```git``` and using ```pip``` to install it locally. It is strongly reccomended using a Python vritual environment manager such as ```conda``` to maintain a clean environment. The installation steps are described in the below shell or terminal commands:

First, the custom ```DEASC_FLORIS``` (wind farm simulator) and ```DEASC_TuRBO``` (Bayesian optimisation algorithm) packages need to be installed as shown:
```
# Download the source codes from the `main` or 'master' branch
git clone -b main https://github.com/fg320/DEASC_FLORIS.git
git clone -b main https://github.com/fg320/DEASC_TuRBO.git

# If using conda, be sure to activate your environment prior to installing
# conda activate <env name>

# Install DEASC_FLORIS and DEASC_TuRBO
pip install -e DEASC_FLORIS
pip install -e DEASC_TuRBO
```

Second, DEASC can be installed with:
```
# Download the source code from the `main` branch
git clone -b main https://github.com/fg320/DEASC.git

# If using conda, be sure to activate your environment prior to installing
# conda activate <env name>

# Install DEASC_FLORIS and DEASC_TuRBO
pip install -e DEASC
```

## Contributing
Contributions are warmly welcomed! Whether it's bug fixes, feature enhancements, or new ideas, your contributions help improve this project.

## Citation
The project can be cited using the following DOI:
```Missing DOI badge```

## LICENCE
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Sources
This project is based on the following publications:
```
@Article{gori_2023,
AUTHOR = {Gori, F. and Laizet, S. and Wynn, A.},
TITLE = {Wind farm power maximisation via wake steering: a Gaussian process‐based yaw‐dependent parameter tuning approach},
JOURNAL = {Wind Energy (Under review)},
YEAR = {2023 (Submitted)}
}
```
```
@Article{wes-8-1425-2023,
AUTHOR = {Gori, F. and Laizet, S. and Wynn, A.},
TITLE = {Sensitivity analysis of wake steering optimisation for wind farm power maximisation},
JOURNAL = {Wind Energy Science},
VOLUME = {8},
YEAR = {2023},
NUMBER = {9},
PAGES = {1425--1451},
URL = {https://wes.copernicus.org/articles/8/1425/2023/},
DOI = {10.5194/wes-8-1425-2023}
}
```
