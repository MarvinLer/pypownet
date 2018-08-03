# L2RPNenv
This repo contains an environment for Learning to Run a Power Network (L2RPN).

<a name="installation"></a>
## Installation
### Using Docker

Launch with:
```
sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/l2rpn:1.0.4
```

### Requirements:
- Python >= 3.4
- octave >= 4.0.0

### Instructions:
#### Install Octave

First, Octave >= 4.0.0 needs to be installed. To install Octave on Ubuntu >= 14.04:
```
sudo add-apt-repository ppa:octave/stable
sudo apt-get update
sudo apt-get install octave
```

#### Download matpower
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

#### Update the matpower path
The matpower path is already set by default if your folder architecture is the same as above.

Otherwise, you need to specify the path to the matpower6.0 folder within the quotes of the matpower_path.config file (relative or absolute) according to the following template:
```
matpower_path = 'path_to_matpower6.0_folder'
```

#### Run the Python installation script
Finally, run the installation script to install all the Python dependencies:
```
python setup.py install
```

To run a usage example of the environment:
```
python -m src.usage_example
```
