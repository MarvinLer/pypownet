# PyPowNet
PyPowNet stands for Python Power Network, which is a simulator for power (electrical) networks.
The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of temporal injections (productions and consumptions) for discretized timesteps. Loadflow computations relies on Matpower and can be run under the AC or DC models. The simulator is able to simulate cascading failures, where successively overflowed lines are switched off and a loadflow is computed on the subsequent grid.
The simulator comes with an Reinforcement Learning-focused environement, which implements states (observations), actions (reduced to node-splitting and line status switches) as well as a reward signal.
Finally, a renderer is available, such that the observations of the network can be plotted in real-time (synchronized with the game time).

# Installation
## Using Docker
Retrieve the Docker image:
```
sudo docker pull marvinler/l2rpn
```

## Without using Docker
### Requirements:
- Python >= 3.6
- Octave >= 4.0.6

### Instructions:
#### Step 1: Install Octave

First, Octave >= 4.0.6 needs to be installed. To install Octave on Ubuntu >= 14.04:
```
sudo add-apt-repository ppa:octave/stable
sudo apt-get update
sudo apt-get install octave
```

#### Step 2: Install Python3.6
```
sudo apt-get update
sudo apt-get install python3.6
```
If you have any trouble with this step, please refer to [the official webpage of Python](https://www.python.org/downloads/release/python-366/).

#### Step 3: Download matpower
A matpower .zip file can be downloaded on the official matpower repository (select version 6.0): http://www.pserc.cornell.edu/matpower/

The archive file should be uncompressed such that the resulting matpower6.0 folder should be in the parent folder of this project folder with a resulting architecture similar to:
```
parent_folder
   gym-powernetwork/
      README.md (the current file)
      setup.py
      matpower_path.config
      gym_powernetwork/
         __init__.py
         envs/
   matpower6.0/
```

#### Step 4: Update the matpower path
The matpower path is already set by default if your folder architecture is the same as above.

Otherwise, you need to specify the path to the matpower6.0 folder within the quotes of the matpower_path.config file (relative or absolute) according to the following template:
```
matpower_path = 'path_to_matpower6.0_folder'
```

#### Step 5: Run pip on the required libraries
Finally, run the following pip command to install the current simulator (including the Python libraries dependencies):
```
python setup.py install
```
After this, this simulator is available under the name pypownet.


# Basic usage
## Using Docker
Launch with (display is shared for running the renderer):
```
sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/l2rpn:1.0.4
```
By default, this will run the main program which can be used to launch experiments using the simulator with CLI (the default agent is a do-nothing agent, which produces no action at each step).

## Without using Docker
Experiments can be conducted using the CLI.
### Simple usage
```
python -m pypownet.main
```
### Controling some experiments parameters
Some experiements parameters are available via the CLI; please use `python -m pypownet.main --help` for further information on these arguments. Example running 4 simultaneous experiments for 100 iterations each with verbose:
```
python -m pypownet.main --batch 4 --niter 100 --verbose
```