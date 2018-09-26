Introduction
************


pypownet stands for Python Power Network, which is a simulator for power (electrical) networks.

The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of temporal injections (productions and consumptions) for discretized timesteps. Loadflow computations relies on Matpower and can be run under the AC or DC models. The simulator is able to simulate cascading failures, where successively overflowed lines are switched off and a loadflow is computed on the subsequent grid.

The simulator comes with an Reinforcement Learning-focused environment, which implements states (observations), actions (reduced to node-splitting and line status switches) as well as a reward signal. Finally, a renderer is available, such that the observations of the network can be plotted in real-time (synchronized with the game time).