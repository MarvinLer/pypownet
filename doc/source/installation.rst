************
Installation
************

Using Docker
============
Retrieve the Docker image::

    sudo docker pull marvinler/pypownet:2.1.1
This docker image contains all the necessary dependencies built on top of a Linux distribution. The sources of pypownet are available under the **pypownet** folder. See Usage Example for launching the image.

Without using Docker
====================
Requirements
------------
.. bibliographic fields:

:Python: >= 3.6
:Octave: >= 4.0
:Matpower: >= 6.0

Instructions
------------
Step 1: Install Octave
^^^^^^^^^^^^^^^^^^^^^^

To install Octave >= 4.0.0 on Ubuntu >= 14.04::

    sudo add-apt-repository ppa:octave/stable
    sudo apt-get update
    sudo apt-get install octave

If Octave is already installed on your machine, ensure that its version from ``octave --version`` is higher than 4.0.0.

Step 2: Install Python3.6
^^^^^^^^^^^^^^^^^^^^^^^^^
The standard procedure::

    sudo apt-get update
    sudo apt-get install python3.6
If you have any trouble with this step, please refer to `the official webpage of Python 3.6.6 <https://www.python.org/downloads/release/python-366/>`__.

Step 3: Get pypownet
^^^^^^^^^^^^^^^^^^^^
In a parent folder, clone the current sources::

    mkdir parent_folder && cd parent_folder
    git clone https://github.com/MarvinLer/pypownet.git
This should create a folder **pypownet** with the current sources.

Step 4: Get Matpower
^^^^^^^^^^^^^^^^^^^^
The latest sources of matpower need to be installed for computing loadflows. This can be done using the command that should be run within the parent folder of this file::

    git clone https://github.com/MATPOWER/matpower.git


.. Attention:: In any case, you need to ensure that the path specified in ``matpower_path.config`` leads to the matpower folder (prefer absolute path; path relative to the configuration file are tolerated if changed before running the next setup script of pypownet).

Step 5: Run the installation script of pypownet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, pypownet relies on some python packages (including e.g. numpy). Run the following command to install the current simulator (including the Python libraries dependencies)::

    python3.6 setup.py install
After this, this simulator is available under the name pypownet (e.g. ``import pypownet``).