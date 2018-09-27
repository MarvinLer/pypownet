*****************
Example of agents
*****************

This page contains a collection of *simple* (non-learning) policies, with some of them leveraging the information contained in observations and the environment.
These agents are all available using the command line agent parameters.

Do-nothing agent
================

This agent returns a do-nothing action at each timestep, i.e. an action that has no consequence on the grid.
The implementation of this agent is actually the default Agent class of pypownet:

.. code-block:: python
   :linenos:
   :caption: agent.py

    import pypownet.environment
    import pypownet.agent


    class CustomAgent(pypownet.agent.Agent):
        """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
        class.
        """
        def __init__(self, environment):
            assert isinstance(environment, pypownet.environment.RunEnv)
            super().__init__(environment)

        def act(self, observation):
            """Produces an action given an observation of the environment. Takes as argument an observation of the current state, and returns the chosen action."""
            action = self.environment.action_space.get_do_nothing_action()

            # Alternative equivalent code:
            # action_space = self.environment.action_space  # Retrieve action builder, an instance of pypownet.environment.ActionSpace
            # action_asarray = np.zeros(action_space.action_length)
            # action = action_space.array_to_action(action_asarray)

            return action

Random switch agent
===================
This agent will simply return an action where only one value is 1, which is equivalent of saying that only 1 switch is activated (whether it is a node-splitting switch or a line status switch).

This method performs poorly, but is here to illustrate the gateway between (numpy) arrays and actions.

.. code-block:: python
   :linenos:

    import pypownet.environment
    import pypownet.agent
    import numpy as np


    class RandomSwitch(pypownet.agent.Agent):
        def __init__(self, environment):
            super().__init__(environment)

        def act(self, observation):
            # Sanity check: an observation is a structured object defined in the environment file.
            assert isinstance(observation, pypownet.environment.Observation)
            action_space = self.environment.action_space  # Retrieve action builder, an instance of pypownet.environment.ActionSpace

            # Build array of action length with all 0 except one random 1
            action_asarray = np.zeros(action_space.action_length)
            action_asarray(np.random.randint(action_length)) = 1  # Activate one random switch

            # Convert previous array into an instance of pypownet.game.Action
            action = action_space.array_to_action(action_asarray)
            return action

Random 1-line switch agent
==========================

The actions of this agent are completely restricted to line status switches with at most 1 switch.
At each timestep, the agent will pick a random line id, and switch its status: that is, its action vector is all 0 except one 1 value within the line status subaction.

At the beginning (so after each reset), all of the power lines of the grid should be switched ON, which implies that this model mainly switches OFF power lines, which tend to create isolated loads or productions or even global outage.
As a consequence, this model does not perform well (by switching line status of ON lines, the model only lower the capacity of the grid).
It illustrates how to leverage the inner structure of an action, which is roughly made of two subarrays, one for node-splitting related switches, and one for lines status related switches.

.. code-block:: python
   :linenos:
   :caption: agent.py

    import pypownet.environment
    import pypownet.agent
    import numpy as np


    class RandomLineSwitch(pypownet.agent.Agent):
        """ An example of a baseline controler that randomly switches the status of one random power line per timestep
        (if the random line is previously online, switch it off, otherwise switch it on).
        """
        def __init__(self, environment):
            super().__init__(environment)

        def act(self, observation):
            # Sanity check: an observation is a structured object defined in the environment file.
            assert isinstance(observation, pypownet.environment.Observation)
            action_space = self.environment.action_space

            # Create template of action with no switch activated (do-nothing action)
            action = action_space.get_do_nothing_action()
            # Select random line id
            random_line_id = np.random.randint(action_space.lines_status_subaction_length)

            # Given the template 0 action, and the line id, set new line status SWITCH to 1 (i.e. activate line status switch)
            action_space.set_lines_status_switch_from_id(action=action,
                                                         line_id=random_line_id,
                                                         new_switch_value=1)

            return action

Random 1-substation node-splitting switch agent
===============================================

To the image of the previous agent, the second part of an action is related to node-splitting switches: if a switch is activated, then the corresponding element will move from its current node within its substation to the other one (there are 2 nodes per substation).
This agent leverages some helpers in the action space that allows the model to fully exploit the topological structure of the grid.

Indeed, the model first retrieve the true IDs of the substations of the grid, which are contained within any observation given by the Environment in their sublist **substations_ids**.
It then picks one random substation ID, and uses the function **get_number_elements_of_substation** of action space to retrieve the number of elements in this substation (this function is a helper, which returns the total number of productions, loads, lines origins and lines extremities that are part of the associated substation).
The agent then construct a random binary configuration of size the previous number, and insert this new configuration into a 0 action using the function **set_switches_configuration_of_substation** of the action space, that *build* the new configuration on top of the action (i.e. it does not modify the other values of the input action).

This agent mainly illustrates how to leverage some of the grid system topological structure, with the use of the action space to construct meaningful actions.


.. code-block:: python
   :linenos:
   :caption: agent.py

    import pypownet.environment
    import pypownet.agent
    import numpy as np


    class RandomNodeSplitting(Agent):
        """ Implements a "random node-splitting" agent: at each timestep, this controler will select a random substation
        (id), then select a random switch configuration such that switched elements of the selected substations change the
        node within the substation on which they are directly wired.
        """
        def __init__(self, environment):
            super().__init__(environment)

        def act(self, observation):
            # Sanity check: an observation is a structured object defined in the environment file.
            assert isinstance(observation, pypownet.environment.Observation)
            action_space = self.environment.action_space

            # Create template of action with no switch activated (do-nothing action)
            action = action_space.get_do_nothing_action()

            # Select a random substation ID on which to perform node-splitting, and retrieve its total number of elements (i.e. size of its subaction list)
            substations_ids = observation.substations_ids  # Retrieve the true substations ID in the observation (they are fixed per environment)
            target_substation_id = np.random.choice(substations_ids)

            # Computes the number of elements of substation (= size of values-related subaction) and choses a new random
            # switch configuration (binary array)
            expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
            target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))

            # Incorporate this new subaction into the action (which is initially a 0 action)
            action_space.set_switches_configuration_of_substation(action=action,
                                                                  substation_id=target_substation_id,
                                                                  new_configuration=target_configuration)

        return action


Exhaustive 1-line switch search agent
=====================================

The actions of this agent are completely restricted to line status switches with at most 1 switch.
At each timestep, the agent will *simulate* every 1-line switch action, independently from one another, as well as the do-nothing action.
The *simulate* method will return one reward per action simulated.
After this, the model will return the action that maximizes the expected reward for the next timestep restricted to 1-line switches.

.. Note:: The *simulate* method is an approximation of the *step* method: the mode is automatically DC (should be AC in *step*), and the hazards are not computed for future timesteps becaseu they should not be predicted by the agent.

This agent should perform relatively well compared to the other ones above.
Even if the *simulate* is approximative, the model can easily discard actions that would lead to global outage (if outage in DC, then probably also in AC).
Also, this agent is way longer than the other ones, because it simulates multiple actions per timestep: one per line (+ 1 do-nothing) and the simulate method takes some time on its own.



.. code-block:: python
   :linenos:
   :caption: agent.py

    import pypownet.environment
    import pypownet.agent
    import numpy as np


    class SearchLineServiceStatus(Agent):
    """ Exhaustive tree search of depth 1 limited to no action + 1 line switch activation
    """
        def __init__(self, environment):
            super().__init__(environment)

        def act(self, observation):
            # Sanity check: an observation is a structured object defined in the environment file.
            assert isinstance(observation, pypownet.environment.Observation)
            action_space = self.environment.action_space

            number_of_lines = action_space.lines_status_subaction_length
            # Simulate the line status switch of every line, independently, and save rewards for each simulation (also store
            # the actions for best-picking strat)
            simulated_rewards = []
            simulated_actions = []
            for l in range(number_of_lines):
                print('    Simulating switch activation line %d' % l, end='')

                # Construct the action where only line status of line l is switched
                action = action_space.get_do_nothing_action()
                action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)

                # Call the simulate method with the action to be simulated
                simulated_reward = self.environment.simulate(action=action)

                # Store ROI values
                simulated_rewards.append(simulated_reward)
                simulated_actions.append(action)
                print('; expected reward %.5f' % simulated_reward)

            # Also simulate the do nothing action
            print('    Simulating switch activation line %d' % l, end='')
            donothing_action = self.environment.action_space.get_do_nothing_action()
            donothing_simulated_reward = self.environment.simulate(action=donothing_action)
            simulated_rewards.append(donothing_simulated_reward)
            simulated_actions.append(donothing_action)

            # Seek for the action that maximizes the reward
            best_simulated_reward = np.max(simulated_rewards)
            best_action = simulated_actions[simulated_rewards.index(best_simulated_reward)]

            print('  Best simulated action: disconnect line %d; expected reward: %.5f' % (
                simulated_rewards.index(best_simulated_reward), best_simulated_reward))

            return best_action
