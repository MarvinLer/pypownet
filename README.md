# PyPowNet
PyPowNet stands for Python Power Network, which is a simulator for power (electrical) networks.

The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of temporal injections (productions and consumptions) for discretized timesteps. Loadflow computations relies on Matpower and can be run under the AC or DC models. The simulator is able to simulate cascading failures, where successively overflowed lines are switched off and a loadflow is computed on the subsequent grid.

The simulator comes with an Reinforcement Learning-focused environement, which implements states (observations), actions (reduced to node-splitting and line status switches) as well as a reward signal. Finally, a renderer is available, such that the observations of the network can be plotted in real-time (synchronized with the game time).

# Installation
## Using Docker
Retrieve the Docker image:
```
sudo docker pull marvinler/pypownet
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

#### (Step 3 if not already done: Get these sources)
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

In any case, you can update the path of matpower download folder within the ```matpower_path.config``` file.

#### Step 5: Run the installation script of PyPowNet
Finally, run the following pip command to install the current simulator (including the Python libraries dependencies):
```
python3.6 setup.py install
```
After this, this simulator is available under the name pypownet (e.g. ```import pypownet```).


# Basic usage
## Without using Docker
Experiments can be conducted using the CLI.
### Simple usage for launching experiments
You can use the command line to make an agent play a game. The simplest usage will launch the agent within the __Agent__ class of the file pypownet/agent.py on 100 timesteps for the grid with 118 substations, and with only one simultaneous game:
```
python -m pypownet.main
```
### Using CLI arguments
Some experiements parameters are available via the CLI; you can use `python -m pypownet.main --help` for further information about these runners arguments. Example running 4 simultaneous experiments for 100 iterations each with verbose:
```
python -m pypownet.main --batch 4 --niter 100 --verbose
```
## Using Docker
You can use the command line of the image with shared display (for running the renderer):
```
sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/pypownet sh
```
This will open a terminal of the image. The usage is then identical to without docker, by doing the steps within this terminal.