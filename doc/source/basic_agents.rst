********************
Agents specificities
********************

pypownet has an Environment interface, which wraps the backend game instance, and allows a RL-focused mechanism that can be used by the agents (also called models or controlers) which act as grid conduct operators.

At each timestep, upon reception of the current observation of the game, an agent is supposed to produce an action, that will be applied onto the grid, producing the next timestep observation which will be given to the agent and so on.

In more details, an agent should be a python subclass of the pypownet class pypownet.agent.Agent, whose only mandatory method is **act**. This is the basic structure of an agent called CustomAgent:

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
            assert isinstance(observation, pypownet.environment.Observation)

            # Implement your policy here.
            action = self.environment.action_space.get_do_nothing_action()

            return action
