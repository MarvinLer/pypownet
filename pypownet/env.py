from pypownet.game import IllegalActionException, LoadCutException

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

        def __init__(self, active_loads, reactive_loads, voltage_loads,
                     active_productions, reactive_productions, voltage_productions,
                     active_flows_origin, reactive_flows_origin, voltage_flows_origin,
                     active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity,
                     lines_capacity_usage, topology_vector):
            # Loads related state values
            self.active_loads = active_loads
            self.reactive_loads = reactive_loads
            self.voltage_loads = voltage_loads

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
            self.lines_capacity_usage = lines_capacity_usage

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

        def sample(self):
            action = np.zeros((self.n,))
            action[np.random.randint(self.n)] = 1
            return action

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
        self.multiplicative_factor_line_usage_reward = -1.  # Mult factor for line usage subreward
        self.multiplicative_factor_distance_reference_grid = -1.  # Mult factor to the sum of differed nodes
        self.illegal_action_exception_reward = -grid_case  # Reward in case of bad action shape/form
        self.loadflow_exception_reward = -grid_case  # Reward in case of loadflow software error
        self.connexity_exception_reward = -grid_case  # Reward when the grid is not connexe (at least two islands)
        self.load_cut_exception_reward = -grid_case  # Reward when one load is isolated

        # Action cost reward hyperparameters
        self.cost_line_switch = 0  # 1 line switch off or switch on
        self.cost_node_switch = 0  # Changing the node on which an element is directly wired

        self.last_rewards = []
        self.last_action = None

    def _get_obs(self):
        return self.game.get_observation()

    @staticmethod
    def __game_over(reward, info):
        """ Utility function that returns every timestep tuple with observation equals to None and done equals to True.
        """
        observation = None
        done = True
        return observation, reward, done, info

    def get_reward(self, observation, action):
        # Load cut reward: TODO
        load_cut_reward = 0

        # Reference grid distance reward
        initial_topology = self.game.get_initial_topology(as_array=True)
        current_topology = self._get_obs().topology
        n_differed_nodes = np.sum(np.where(
            initial_topology[:-self.observation_space.n_lines] != current_topology[:-self.observation_space.n_lines]))
        reference_grid_distance_reward = self.multiplicative_factor_distance_reference_grid * n_differed_nodes

        # Action cost reward: compute the number of line switches, node switches, and return the associated reward
        if action is None:
            action_cost_reward = 0
        else:
            n_nodes_switches = np.sum(action[:-self.action_space.n_lines])
            n_lines_switches = np.sum(action[-self.action_space.n_lines:])
            action_cost_reward = self.cost_node_switch * n_nodes_switches + self.cost_line_switch * n_lines_switches

        # Line usage subreward: compute the mean square of the per-line thermal usage
        lines_capacity_usage = observation.lines_capacity_usage
        line_usage_reward = self.multiplicative_factor_line_usage_reward * np.sum(np.square(lines_capacity_usage))

        self.last_rewards = [line_usage_reward, action_cost_reward, reference_grid_distance_reward, load_cut_reward]

        return load_cut_reward + action_cost_reward + reference_grid_distance_reward + line_usage_reward

    def step(self, action):
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
            reward = self.get_reward(observation, None)
            done = False
            info = None
        except pypownet.game.NoMoreScenarios as e:  # All input have been played
            observation = None
            reward = 0.
            done = True
            info = e
        except LoadCutException as e:
            return self.__game_over(reward=self.load_cut_exception_reward, info=e)
        except pypownet.grid.GridNotConnexeException as e:
            return self.__game_over(reward=self.connexity_exception_reward, info=e)
        except pypownet.grid.DivergingLoadflowException as e:
            return self.__game_over(reward=self.loadflow_exception_reward, info=e)

        return observation, reward, done, info

    def simulate(self, action=None):
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
            reward = self.get_reward(simulated_observation, None)
        except pypownet.game.NoMoreScenarios:
            reward = 0.
        except LoadCutException:
            reward = self.load_cut_exception_reward
        except pypownet.grid.GridNotConnexeException:
            reward = self.connexity_exception_reward
        except pypownet.grid.DivergingLoadflowException:
            reward = self.loadflow_exception_reward

        return reward

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

if __name__ == '__main__':
    env = RunEnv(grid_case=118)
    observation = env._get_obs()  # Initial observation

    action_size = env.action_space.n
    n_lines = env.action_space.n_lines
    topology_subaction = np.zeros((action_size - n_lines,))

    timestep_rewards = []
    n_timesteps = 30
    for l in range(n_timesteps):
        env.game.grid.filename = 'swoff_line%d.m' % l
        print(' Simulation with line %d switched off' % l)
        line_service_subaction = np.zeros((n_lines,))
        line_service_subaction[l] = 1  # Toggle line l

        action = np.concatenate((topology_subaction, line_service_subaction))
        simulated_reward = env.simulate(action)

        timestep_rewards.append(simulated_reward)
    print(' Simulation with no action')
    env.game.grid.filename = 'nothing.m'
    simulated_reward = env.simulate(None)
    timestep_rewards.append(simulated_reward)

    argmax_reward = np.argmax(timestep_rewards)
    print('rewards', timestep_rewards, 'argmax', argmax_reward)
    if argmax_reward == len(timestep_rewards) - 1:
        print('Action chosen: no action')
        action = None
    else:
        line_service_subaction = np.zeros((n_lines,))
        line_service_subaction[argmax_reward] = 1
        action = np.concatenate((topology_subaction, line_service_subaction))
