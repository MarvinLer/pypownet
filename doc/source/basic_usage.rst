***********
Basic usage
***********

.. _docker_launch:

Command-line usage
******************

.. WARNING:: Currently, the sources of pypownet are mandatory to run the command line argument (if using the Docker image, they are already within the image).

If you installed pypownet using the Docker image, you will first need to launch bash within the image using::

    sudo docker run -it --privileged --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/pypownet sh



In any case, pypownet comes with a command-line interface that allows to run simulations with a specific agent and control parameters. The basic usage will run a do-nothing policy with the default parameters::

    python -m pypownet.main

This basic usage will run a do-nothing agent on the **default14/** parameters (emulates a grid with 14 substations, 5 productions, 11 consumptions and 20 lines), it takes ~100 seconds to run 1000 timesteps (old i5).

.. Hint:: You can use ``python -m pypownet.main --help`` for information about the CLI tool or check `the CLI usage section <cli_usage.rst>`__


As package usage
****************

The installation process should ensure that pypownet in installed along with your corresponding python3.6.
As a consequence, pypownet should be importable in your projects just like any package installed using pip: ``import pypownet``.
Only the modules **pypownet.environment**, **pypownet.agent** and either **pypownet.runner.Runner** or **pypownet.main** should be used, as other packages are not supposed to be used out of the induced context.

Usually, what you would do is first create an instance of **pypownet.environment.RunEnv** with an environment parameters + crate an instance of **pypownet.agent.Agent** (or any subclass), then get an instance of ``pypownet.runner.Runner`` with appropriate parameters (including previous ``RunEnv`` and ``Agent``), and run its ``loop`` method to run the simulator.
The latter functions output the total reward at the end of the experiment.
