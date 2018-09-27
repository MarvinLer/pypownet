============
Introduction
============


pypownet stands for Python Power Network, which is a simulator for power (electrical) networks.

The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of temporal injections (productions and consumptions) for discretized timesteps. Loadflow computations relies on Matpower and can be run under the AC or DC models. The simulator is able to simulate cascading failures, where successively overflowed lines are switched off and a loadflow is computed on the subsequent grid.

The simulator comes with an Reinforcement Learning-focused environment, which implements states (observations), actions (reduced to node-splitting and line status switches) as well as a reward signal.
Finally, a renderer is available, such that the observations of the network can be plotted in real-time (synchronized with the game time).

Background
**********


Main features
*************
pypownet is a power grid simulator, that emulates a power grid that is subject to pre-computed injections, planned maintenance as well as random external hazards. Here is a list of pypownet main features:

    - emulates a grid of any size and electrical properties in a game discretized in timesteps of any (fixed) size
    - computes and apply cascading failure process: at each timestep, overflowed lines with certain conditions are switched off, with a consequent loadflow computation to retrieve the new grid steady-state, and reiterating the process
    - has an RL-focused interface, where players or controlers can play actions (node-splitting or line status switches) on the current grid, based on a partial observation of the grid (high dimension), with a customable reward signal (and game over options)
    - has a renderer that enables the user to see the grid evolving in real-time, as well as the actions of the controler currently playing and further grid state details (works only for pypownet official grid cases)
    - has a runner that enables to use pypownet fully by simply coding an agent (with a method act(observation))
    - possess some baselines models (including treesearches) illustrating how to use the furnished environment
    - can be launched with CLI with the possibility of managing certain parameters (such as renderer toggling or the agent to be played)
    - functions on both DC and AC mode
    - has a set of parameters that can be customized (including AC or DC mode, or hard-overflow coefficient), associated with sets of injections, planned maintenance and random hazards of the various chronics
    - handles node-splitting (at the moment only max 2 nodes per substation) and lines switches off for topology management

