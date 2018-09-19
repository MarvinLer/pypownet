# PyPowNet
PyPowNet stands for Python Power Network, which is a simulator for power (electrical) networks.

The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of temporal injections (productions and consumptions) for discretized timesteps. Loadflow computations relies on Matpower and can be run under the AC or DC models. The simulator is able to simulate cascading failures, where successively overflowed lines are switched off and a loadflow is computed on the subsequent grid.

The simulator comes with an Reinforcement Learning-focused environment, which implements states (observations), actions (reduced to node-splitting and line status switches) as well as a reward signal. Finally, a renderer is available, such that the observations of the network can be plotted in real-time (synchronized with the game time).

*   [1 Installation](#installation)
    *   [1.1 Using Docker](#using-docker)
    *   [1.2 Without using Docker](#without-using-docker)
        *   [1.2.1 Requirements](#requirements)
        *   [1.2.2 Instructions](#instructions)
*   [2 Basic usage](#basic-usage)
    *   [2.1 Without using Docker](#without-using-docker-1)
    *   [2.2 Using Docker](#using-docker-1)
*   [3 Main features of pypownet](#main-features)
*   [4 Generate the documentation](#generate-the-documentation)
*   [5 License information](#license-information)

# Installation
## Using Docker
Retrieve the Docker image:
```
sudo docker pull marvinler/pypownet:2.0.4
```

## Without using Docker
### Requirements:
*   Python >= 3.6
*   Octave >= 4.0.6
*   Matpower >= 6.0
*   PyPowNet >= 2.0.3

### Instructions:
#### Step 1: Install Octave

To install Octave >= 4.0.0 on Ubuntu >= 14.04:
```
sudo add-apt-repository ppa:octave/stable
sudo apt-get update
sudo apt-get install octave
```
If Octave is already installed on your machine, ensure that its version from `octave --version` is higher than 4.0.0.

#### Step 2: Install Python3.6
```
sudo apt-get update
sudo apt-get install python3.6
```
If you have any trouble with this step, please refer to [the official webpage of Python](https://www.python.org/downloads/release/python-366/).

#### (Step 3 if not already done: Get pypownet)
In a parent folder, clone the current sources:
```
mkdir parent_folder && cd parent_folder
git clone https://github.com/MarvinLer/pypownet
```
This should create a folder pypownet with the current sources.

#### Step 4: Get Matpower
The latest sources of matpower need to be installed for computing loadflows. This can be done using the command that should be run within the parent folder of this file:
```
git clone https://github.com/MATPOWER/matpower.git
```

In any case, you can update the path of matpower download folder within the ```matpower_path.config``` file (prefer absolute path).

#### Step 5: Run the installation script of PyPowNet
Finally, run the following pip command to install the current simulator (including the Python libraries dependencies):
```
python3.6 setup.py install
```
After this, this simulator is available under the name pypownet (e.g. ```import pypownet```).


# Basic usage
## Without using Docker
Experiments can be conducted using the CLI.
### Using CLI arguments
Some experiements parameters are available via the CLI; you can use `python -m pypownet.main --help` for further information about these runners arguments. Example running 4 simultaneous experiments for 100 iterations each with verbose:
```
python -m pypownet.main --parameters parameters/default14 --niter 1000 --verbose --render
```
With the parameters of default14, 1000 timesteps takes approximately 100 seconds (depending on your machine).
## Using Docker
You can use the command line of the image with shared display (for running the renderer):
```
sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/pypownet sh
```
This will open a terminal of the image. The usage is then identical to without docker, by doing the steps within this terminal.

# Main features
pypownet is a power grid simulator, that emulates a power grid that is subject to pre-computed injections, planned maintenance as well as random external hazards. Here is a list of pypownet main features:
* emulates a grid of any size and electrical properties in a game discretized in timesteps of any (fixed) size
* computes and apply cascading failure process: at each timestep, overflowed lines with certain conditions are switched off, with a consequent loadflow computation to retrieve the new grid steady-state, and reiterating the process
* has an RL-focused interface, where players or controlers can play actions (node-splitting or line status switches) on the current grid, based on a partial observation of the grid (high dimension), with a customable reward signal (and game over options)
* has a renderer that enables the user to see the grid evolving in real-time, as well as the actions of the controler currently playing and further grid state details (works only for pypownet official grid cases)
* has a runner that enables to use pypownet fully by simply coding an agent (with a method act(observation))
* possess some baselines models (including treesearches) illustrating how to use the furnished environment
* can be launched with CLI with the possibility of managing certain parameters (such as renderer toggling or the agent to be played)
* functions on both DC and AC mode
* has a set of parameters that can be customized (including AC or DC mode, or hard-overflow coefficient), associated with sets of injections, planned maintenance and random hazards of the various chronics
* handles node-splitting (at the moment only max 2 nodes per substation) and lines switches off for topology management

# Generate the documentation
A copy of the documentation can be assess within the file [doc/build](doc/build/index.html).
If you want to compute the latest updated documentation, you will need Sphinx, a Documentation building tool, and a nice-looking custom [Sphinx theme similar to the one of readthedocs.io](https://sphinx-rtd-theme.readthedocs.io/en/latest/):
```
pip install sphinx sphinx_rtd_theme
```
Then:
```
cd doc
sphinx-build -b html ./source ./build
```
The html will be available within the folder [doc/build](doc/build/index.html).

# License information

Copyright 2017-2018 RTE and INRIA (France)

    RTE: http://www.rte-france.com
    INRIA: https://www.inria.fr/

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.
