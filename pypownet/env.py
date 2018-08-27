from pypownet.game import IllegalActionException

__author__ = 'marvinler'
import numpy as np
import pypownet.game
import pypownet.grid


class RunEnv(object):
    class Observation(object):
        """
        The class State is a container for all the values representing the state of a given grid at a given time. It
        contains the following values:
        * The active and reactive power values of the loads
        * The active power values and the voltage setpoints of the productions
        * The values of the power through the lines: the active and reactive values at the origin/extremity of the
        lines as well as the lines capacity usage
        * The exhaustive topology of the grid, as a stacked vector of one-hot vectors
        """

        def __init__(self, active_loads, reactive_loads, voltage_loads, active_productions, reactive_productions,
                     voltage_productions, active_flows_origin, reactive_flows_origin, voltage_flows_origin,
                     active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity, ampere_flows,
                     thermal_limits, topology_vector, n_cut_loads):
            # Loads related state values
            self.active_loads = active_loads
            self.reactive_loads = reactive_loads
            self.voltage_loads = voltage_loads
            self.number_cut_loads = n_cut_loads

            # Productions related state values
            self.active_productions = active_productions
            self.reactive_productions = reactive_productions
            self.voltage_productions = voltage_productions

            # Origin flows related state values
            self.active_flows_origin = active_flows_origin
            self.reactive_flows_origin = reactive_flows_origin
            self.voltage_flows_origin = voltage_flows_origin
            # Extremity flows related state values
            self.active_flows_extremity = active_flows_extremity
            self.reactive_flows_extremity = reactive_flows_extremity
            self.voltage_flows_extremity = voltage_flows_extremity

            # Ampere flows and thermal limits
            self.ampere_flows = ampere_flows
            self.thermal_limits = thermal_limits

            # Topology vector
            self.topology = topology_vector

    class ObservationSpace(object):
        def __init__(self, grid_case):
            assert isinstance(grid_case, int), 'The argument grid_case should be an integer of the case number'
            if grid_case == 14:
                self.n_loads = 11
                self.n_lines = 20
                self.n_prods = 5
            elif grid_case == 30:
                self.n_loads = 20
                self.n_lines = 41
                self.n_prods = 6
            elif grid_case == 118:
                self.n_prods = 56
                self.n_loads = 99
                self.n_lines = 186
            self.n = self.n_prods + self.n_loads + 3 * self.n_lines

    class ActionSpace(object):
        def __init__(self, grid_case):
            assert isinstance(grid_case, int), 'The argument grid_case should be an integer of the case number'
            if grid_case == 118:
                self.n_prods = 56
                self.n_loads = 99
                self.n_lines = 186
            elif grid_case == 14:
                self.n_loads = 11
                self.n_lines = 20
                self.n_prods = 5
            self.n = self.n_prods + self.n_loads + 3 * self.n_lines

        def verify_action_shape(self, action):
            # None action is no action so valid
            if action is None:
                return

            if len(action) != self.n:
                raise IllegalActionException('Expected action of size %d, got %d' % (self.n, len(action)))
            if not set(action).issubset([0, 1]):
                raise IllegalActionException('Some values of the action are not 0 nor 1')

    def __init__(self, grid_case=118):
        # Instantiate game & action space
        """
        Instante the game Environment as well as the Action Space.

        :param grid_case: an integer indicating which grid to play with; currently available: 14, 118 for respectively
            case14 and case118.
        """
        self.game = pypownet.game.Game(grid_case=grid_case)
        self.action_space = self.ActionSpace(grid_case)
        self.observation_space = self.ObservationSpace(grid_case)

        # Configuration parameters
        self.simulate_cascading_failure = True
        self.apply_cascading_output = True

        # Reward hyperparameters
        self.multiplicative_factor_line_usage_reward = -1.  # Mult factor for line capacity usage subreward
        self.additive_factor_distance_initial_grid = -.05  # Additive factor for each differed node in the grid
        self.additive_factor_load_cut = -grid_case // 10.  # Additive factor for each isolated load
        self.connexity_exception_reward = -self.observation_space.n  # Reward when the grid is not connexe
                                                                     # (at least two islands)
        self.loadflow_exception_reward = -self.observation_space.n  # Reward in case of loadflow software error

        self.illegal_action_exception_reward = -grid_case  # Reward in case of bad action shape/form

        # Action cost reward hyperparameters
        self.cost_line_switch = .1  # 1 line switch off or switch on
        self.cost_node_switch = 0.  # Changing the node on which an element is directly wired

        self.last_rewards = []
        self.last_action = None

    def _get_obs(self):
        return self.game.get_observation()

    def _get_distance_reference_grid(self, observation):
        # Reference grid distance reward
        """ Computes the distance of the current observation with the reference grid (i.e. initial grid of the game).
        The distance is computed as the number of different nodes on which two identical elements are wired. For
        instance, if the production of first current substation is wired on the node 1, and the one of the first initial
        substation is wired on the node 0, then their is a distance of 1 (there are different) between the current and
        reference grid (for this production). The total distance is the sum of those values (0 or 1) for all the
        elements of the grid (productions, loads, origin of lines, extremity of lines).

        :return: the number of different nodes between the current topology and the initial one
        """
        initial_topology = self.game.get_initial_topology(as_array=True)
        current_topology = observation.topology

        n_lines = self.observation_space.n_lines
        n_differed_nodes = np.sum((initial_topology[:-n_lines] != current_topology[:-n_lines]))
        return n_differed_nodes

    def _get_action_cost(self, action):
        # Action cost reward: compute the number of line switches, node switches, and return the associated reward
        """ Compute the >=0 cost of an action. We define the cost of an action as the sum of the cost of node-splitting
        and the cost of lines status switches. In short, the function sums the number of 1 in the action vector, since
        they represent activation of switches. The two parameters self.cost_node_switch and self.cost_line_switch
        control resp the cost of 1 node switch activation and 1 line status switch activation.

        :param action: an instance of Action or a binary numpy array of length self.action_space.n
        :return: a >=0 float of the cost of the action
        """
        if action is None:
            return 0.

        n_nodes_switches = np.sum(action[:-self.action_space.n_lines])
        n_lines_switches = np.sum(action[-self.action_space.n_lines:])
        action_cost = self.cost_node_switch * n_nodes_switches + self.cost_line_switch * n_lines_switches
        return action_cost

    def _get_lines_capacity_usage(self, observation):
        ampere_flows = observation.ampere_flows
        thermal_limits = observation.thermal_limits
        lines_capacity_usage = np.divide(ampere_flows, thermal_limits)
        return lines_capacity_usage

    def get_reward(self, observation, action, do_sum=True):
        # Load cut reward: TODO
        load_cut_reward = self.additive_factor_load_cut * observation.number_cut_loads

        # Reference grid distance reward
        reference_grid_distance = self._get_distance_reference_grid(observation)
        reference_grid_distance_reward = self.additive_factor_distance_initial_grid * reference_grid_distance
        print('ref grid reward', reference_grid_distance_reward)

        # Action cost reward: compute the number of line switches, node switches, and return the associated reward
        action_cost_reward = -1. * self._get_action_cost(action)

        # The line usage subreward is the sum of the square of the lines capacity usage
        lines_capacity_usage = self._get_lines_capacity_usage(observation)
        line_usage_reward = self.multiplicative_factor_line_usage_reward * np.sum(np.square(lines_capacity_usage))

        self.last_rewards = [line_usage_reward, action_cost_reward, reference_grid_distance_reward, load_cut_reward]

        reward_aslist = [load_cut_reward, action_cost_reward, reference_grid_distance_reward, line_usage_reward]
        return sum(reward_aslist) if do_sum else reward_aslist

    def step(self, action, do_sum=True):
        """ Performs a game step given an action. """
        # First verify that the action is in expected condition (if it is not None); if not, end the game
        try:
            self.action_space.verify_action_shape(action)
        except IllegalActionException as e:
            return self.__game_over(reward=self.illegal_action_exception_reward, info=e)
        self.last_action = action  # Store action to plot indicators in renderer if used

        try:
            # Call the step function from the game: if no error raised, then no outage
            self.game.step(action, cascading_failure=self.simulate_cascading_failure,
                           apply_cascading_output=self.apply_cascading_output)
            observation = self.game.get_observation()
            reward_aslist = self.get_reward(observation, action, False)
            done = False
            info = None
        except pypownet.game.NoMoreScenarios as e:
            observation = None
            reward_aslist = [0., 0., 0., 0.]
            done = True
            info = e
        except pypownet.grid.DivergingLoadflowException as e:
            observation = e.last_observation
            reward_aslist = [0., -self._get_action_cost(action), self.loadflow_exception_reward, 0.]
            done = True
            info = e

        reward = sum(reward_aslist) if do_sum else reward_aslist
        return observation, reward, done, info

    def simulate(self, action=None, do_sum=True):
        """ Computes the reward of the simulation of action to the current grid. """
        # First verify that the action is in expected condition (if it is not None); if not, end the game
        try:
            self.action_space.verify_action_shape(action)
        except IllegalActionException as e:
            return self.illegal_action_exception_reward

        try:
            # Get the output simulated state (after action and loadflow computation) or errors if loadflow diverged
            simulated_observation = self.game.simulate(action, cascading_failure=self.simulate_cascading_failure,
                                                       apply_cascading_output=self.apply_cascading_output)
            reward_aslist = self.get_reward(simulated_observation, action, False)
        except pypownet.game.NoMoreScenarios:
            reward_aslist = [0., 0., 0., 0.]
        except (pypownet.grid.GridNotConnexeException, pypownet.grid.DivergingLoadflowException):
            reward_aslist = [0., -self._get_action_cost(action), self.loadflow_exception_reward, 0.]

        return sum(reward_aslist) if do_sum else reward_aslist

    def reset(self, restart=True):
        # Reset the grid overall topology
        self.game.reset(restart=restart)
        return self._get_obs()

    def render(self, mode='human', close=False, game_over=False):
        if mode == 'human':
            self.game._render(self.last_rewards, self.last_action, close, game_over=game_over)
        else:
            raise ValueError("Unsupported render mode: " + mode)

    def get_current_scenario_id(self):
        return self.game.get_current_scenario_id()


# TODO
ACTION_MEANING = {

}