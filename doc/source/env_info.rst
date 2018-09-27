=============================
Using Environment information
=============================


Building actions
----------------
By construction, every agent has access to the current Environment of the simulator (see **__init__** method of **pypownet.agent.Agent** displayed above).
More formally, the ``environment`` member of the class **Agent** is an instance of **pypownet.environment.RunEnv** which is the interface to the simulator.
A **RunEnv** notably contains a step method, which is used for computing the next observation, the resulting reward (which might depend on the action), and whether there was a game over.
This method should not be explicitely used, since pypownet comes with a **pypownet.runnner.Runner** class (which is used when calling pypownet with CLI) and that automates the simulator looping.

**RunEnv** also contains the action space of the simulation.
Formally, an action space is an instance of **pypownet.environment.ActionSpace**, which implements several functions that are used to build actions, which are instances of **pypownet.game.Action**.

.. Important:: In pypownet, the agents are not supposed to construct actions by manipulating **pypownet.game.Action** but rather by using an **ActionSpace**, which is similar to an **Action** factory.

Given an instance *environment* of **RunEnv**, its action space can be assessed with ``environment.action_space``.


Reading observations
--------------------
