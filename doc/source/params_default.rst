Default environments
====================

pypownet comes with 4 sets of paremeters of environment which are named according to their top folder:

    - default14/
    - default30/
    - default118/
    - custom14/

Both environments suse the official IEEE case format associated with the int in their name (e.g. default14/ uses the official case14.m reference grid).
Each has one **level0/** level folder, and has 12 sets of chronics (one per month, starting by january).
The values of the parameters of the **configuration.yaml** file are printing at each start of pypownet.

The reward signal of the custom14/ is rather simple: if the timestep lead to a game over, then return -1, otherwise return 1. This reward signal is not very representative of the factors to optimize for real conditions grid conduct, but illustrates that the reward signal can be designed very simply.

For the *default* environments above, the reward signal has the same mechanism (in fact, they are the same except for hyperparameters which scale with the size of the grid).
More precisally, their reward signal is made of 5 subrewards (the output is then a list of 5 values; the input is still the new observation form the application of the action, as well as the action and a flag indicating game overs, see :ref:`reward_signal`):

:subreward proportional to the number of (topologically) isolated productions:
    Negatively proportional to the number of fully isolated productions.
    In real life settings, production plants are not controlled by dispatchers, so it is best to take actions such that no production is isolated, as it would perturbate the scheme of producers.
:subreward proportional to the number of (topologically) isolated loads:
    Negatively proportional to the number of fully isolated consumtpions (i.e. demands or loads).
    In real life settings, no consumer should be deprive of electricity which notably happens when the load are isolated (since no line reaches them).
    Typically it is more problematic to have an isolated load compared to an isolated production: if the geographic zone without electricity has hospitals, the life of many patients are at risk (even if some have personal generators).
    Consequently, the amplitude of the multiplicative factor to the sum of isolated loads is chosen twice the one for isolated productions (2 isolated productions *equivalent* to 1 isolated load in terms of reward).
:subreward of the cost of the action:
    Recall that all the values of actions are switches: this is roughly what happens in real life, where switches are activated to isolate lines and potentially wire them on the other node (or nodes).
    In practice, a team of line operators need to go the the particular switch to activate, which takes some time and thus cost money to the company.
    Consequently, each switch will account for some negative reward, which scale to the number of activated switches.
    For these reward signals, the cost of a node switch activation halves the one of a line status switch activation, because we would like agents to perform more node-splitting actions (the *a priori* here is that generally, switching the line status of a line will put it OFFLINE, which will reduce the overall capacity of the grid making it more sensible to failures or global outages).
:subreward of the distance to the reference grid:
    The *distance* to the reference grid is precisally the number of nodes of every element in the current grid that differs from the nodes of every element of the initiale grid (note that this in independent from the lines status).
    As such, the distance to the reference grid can be viewed as the minimal number of node switches to activate to convert the current grid topology to the reference one.
    We introduce this distance scale by a negative factor to *force* agents to keep the grid topology around the reference one.
    This is motivated by the fact that human dispatchers are *used to* work with a grid close to a reference grid topology (i.e. the *normal* topology); the dispatchers know the macro patterns of the reference grid, so ensure that their action renders the grid topology close to their confidence zone.
    Since the models would eventually be used to *assist* dispatchers in real life, they should ensure that the grid topology does not drift.
    An important consequence of the design of this reward to be kept in mind is that the reward will sum at each step, such that there might be some delay between the intentions of models (e.g. return to reference grid) and the improvement of the associated reward.
:subreward proportional to the sum of squared lines capacity usage:
    Finally, we use the lines capacity usage to make the models avoid overflows.
    One approach could have been for this subreward to be equal to the number of overflows, but the counting function is not smooth (not even continuous), which is usually not wanted for learning models.
    Instead, we use the square of the nominal lines capacity usage: for each line, its ampere flow is divided by its thermal limit, and the result is squared.
    Since an overflowed line is one with a capacity usage >= 1, the lines currently overflows will be amplified (>=1 squared), while the non-overflowed lines will be deamplified (<1 squared).
    Another positive point of this subreward is that it discriminizes well two grid situations with the same number of overflows, since it takes into account the capacity usage information of all lines.
    For instance, one grid with 1 overflow and all other lines at 99% capacity usage should have a worse reward than one with 1 overflow and 10% usage for other lines.

