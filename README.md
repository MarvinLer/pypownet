# L2RPNenv
This repo contains an environment for Learning to Run a Power Network (L2RPN).

# Installation
## Using Docker

Launch with:
```
sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/l2rpn:1.0.4
```

## Without using Docker
### Requirements:
- Python >= 3.6
- Octave >= 4.0.6

### Instructions:
#### Step 1: Install Python3.6
```
sudo apt-get update
sudo apt-get install python3.6
```
If you have any trouble with this step, please refer to [the official webpage of Python](https://www.python.org/downloads/release/python-366/).

#### Step 1: Install Octave

First, Octave >= 4.0.6 needs to be installed. To install Octave on Ubuntu >= 14.04:
```
sudo add-apt-repository ppa:octave/stable
sudo apt-get update
sudo apt-get install octave
```

#### Step 2: Download matpower
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

#### Step 3: Update the matpower path
The matpower path is already set by default if your folder architecture is the same as above.

Otherwise, you need to specify the path to the matpower6.0 folder within the quotes of the matpower_path.config file (relative or absolute) according to the following template:
```
matpower_path = 'path_to_matpower6.0_folder'
```

#### Step 4: Run pip on the required libraries
Finally, run the following pip command to install all the Python dependencies:
```
pip install -r requirement.txt
```


# Basic usage
## Using Docker
TODO

## Without using Docker
To run a usage example of the environment, go to the project root then:
```
python -m src.usage_example
```
