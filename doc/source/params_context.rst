=====================
Parameters management
=====================

Some mechanisms and inputs of the game are parameterizable to ensure that the users of the package have some control with respect to the simulations to be run. New simulations environments can be created with any virtual grid (provinding an initial grid case) and an associated set of grid timestep entries to be injected.

.. Hint:: The simulator has primarly been designed for RL research; consequently, the overall parameters environment organization is influenced for RL integration.

Parameters are organized into a generic folder structure:

.. image:: archi_params_env.png
    :align: center

An environment parameters is made of the following elements:

    - one reference grid (defines static parameters of the simulated power grid)
    - sets of chronics (defines the temporal data driving the inputs of the grid environment)
    - a configuration file (contains parameters for some mechanisms of the simulator)
    - (*optional*) a reward signal python file (implements the formula to compute a reward based on a observation and an action)

Before rewieving the details about each of these elements, pypownet comes with a helper that creates a template parameters environment. The script can be launched using::

    python -m parameters.build_new_parameters_environment

The terminal will ask you several questions relative to the latter elements, and then build the overal architecture of the environment. After the execution of this script, you will need to:

    - (*mandatory*) fill the data for the chronics
    - (*optional*) modify the default values of the auto-generated configuration file
    - (*optional*) modify the reward signal of the file reward_signal.py.

