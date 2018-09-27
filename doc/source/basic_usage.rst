.. WARNING:: Currently, the sources of pypownet are mandatory to run the command line argument (if using the Docker image, they are already within the image).

***********
Basic usage
***********

If you installed pypownet using the Docker image, you will first need to launch bash within the image using::

    sudo docker run -it --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" marvinler/pypownet sh



In any case, pypownet comes with a command-line interface that allows to run simulations with a specific agent and control parameters. The basic usage will run a do-nothing policy with the default parameters::

    python -m pypownet.main

This basic usage will run a do-nothing agent on the **default14/** parameters (emulates a grid with 14 substations, 5 productions, 11 consumptions and 20 lines), it takes ~100 seconds to run 1000 timesteps (old i5).

.. Hint:: You can use ``python -m pypownet.main --help`` for information about the CLI tool or check `the CLI usage section <cli_usage.rst>`__
