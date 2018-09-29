__author__ = 'marvinler'
import pypownet.environment
import pypownet.reward_signal
import numpy as np


class CustomRewardSignal(pypownet.reward_signal.RewardSignal):
    def __init__(self):
        super().__init__()

        constant = 14

        # Hyper-parameters for the subrewards
        # Mult factor for line capacity usage subreward
        self.multiplicative_factor_line_usage_reward = -1.
        # Multiplicative factor for total number of differed nodes in the grid and reference grid
        self.multiplicative_factor_distance_initial_grid = -.02
        # Multiplicative factor total number of isolated prods and loads in the grid
        self.multiplicative_factor_number_loads_cut = -constant / 5.
        self.multiplicative_factor_number_prods_cut = -constant / 10.

        # Reward when the grid is not connexe (at least two islands)
        self.connexity_exception_reward = -constant
        # Reward in case of loadflow software error (e.g. 0 line ON)
        self.loadflow_exception_reward = -constant

        # Multiplicative factor for the total number of illegal lines reconnections
        self.multiplicative_factor_number_illegal_lines_reconnection = -constant / 100.

        # Reward when the maximum number of isolated loads or prods are exceeded
        self.too_many_productions_cut = -constant
        self.too_many_consumptions_cut = -constant

        # Action cost reward hyperparameters
        self.multiplicative_factor_number_line_switches = -.2  # equivalent to - cost of line switch
        self.multiplicative_factor_number_node_switches = -.1  # equivalent to - cost of node switch

    def compute_reward(self, observation, action, flag):
        # First, check for flag raised during step, as they indicate errors from grid computations (usually game over)
        if flag is not None:
            if isinstance(flag, pypownet.environment.DivergingLoadflowException):
                reward_aslist = [0., 0., -self.__get_action_cost(action), self.loadflow_exception_reward, 0.]
            elif isinstance(flag, pypownet.environment.IllegalActionException):
                # If some broken lines are attempted to be switched on, put the switches to 0, and add penalty to
                # the reward consequent to the newly submitted action
                reward_aslist = self.compute_reward(observation, action, flag=None)
                n_illegal_reconnections = np.sum(flag.illegal_lines_reconnections)
                illegal_reconnections_subreward = self.multiplicative_factor_number_illegal_lines_reconnection * \
                                                  n_illegal_reconnections
                reward_aslist[2] += illegal_reconnections_subreward
            elif isinstance(flag, pypownet.environment.TooManyProductionsCut):
                reward_aslist = [0., self.too_many_productions_cut, 0., 0., 0.]
            elif isinstance(flag, pypownet.environment.TooManyConsumptionsCut):
                reward_aslist = [self.too_many_consumptions_cut, 0., 0., 0., 0.]
            else:  # Should not happen
                raise flag
        else:
            # Load cut reward
            number_cut_loads = sum(observation.are_loads_cut)
            load_cut_reward = self.multiplicative_factor_number_loads_cut * number_cut_loads

            # Prod cut reward
            number_cut_prods = sum(observation.are_productions_cut)
            prod_cut_reward = self.multiplicative_factor_number_prods_cut * number_cut_prods

            # Reference grid distance reward
            reference_grid_distance = self.__get_distance_reference_grid(observation)
            reference_grid_distance_reward = self.multiplicative_factor_distance_initial_grid * reference_grid_distance

            # Action cost reward: compute the number of line switches, node switches, and return the associated reward
            action_cost_reward = -self.__get_action_cost(action)

            # The line usage subreward is the sum of the square of the lines capacity usage
            lines_capacity_usage = self.__get_lines_capacity_usage(observation)
            line_usage_reward = self.multiplicative_factor_line_usage_reward * np.sum(np.square(lines_capacity_usage))

            # Format reward
            reward_aslist = [load_cut_reward, prod_cut_reward, action_cost_reward, reference_grid_distance_reward,
                             line_usage_reward]

        return reward_aslist

    def __get_action_cost(self, action):
        # Action cost reward: compute the number of line switches, node switches, and return the associated reward
        """ Compute the >=0 cost of an action. We define the cost of an action as the sum of the cost of node-splitting
        and the cost of lines status switches. In short, the function sums the number of 1 in the action vector, since
        they represent activation of switches. The two parameters self.cost_node_switch and self.cost_line_switch
        control resp the cost of 1 node switch activation and 1 line status switch activation.

        :param action: an instance of Action or a binary numpy array of length self.action_space.n
        :return: a >=0 float of the cost of the action
        """
        # Computes the number of activated switches of the action
        number_line_switches = np.sum(action.get_lines_status_subaction())

        number_prod_nodes_switches = np.sum(action.get_prods_switches_subaction())
        number_load_nodes_switches = np.sum(action.get_loads_switches_subaction())
        number_line_or_nodes_switches = np.sum(action.get_lines_or_switches_subaction())
        number_line_ex_nodes_switches = np.sum(action.get_lines_ex_switches_subaction())
        number_node_switches = number_prod_nodes_switches + number_load_nodes_switches + \
                               number_line_or_nodes_switches + number_line_ex_nodes_switches

        action_cost = self.multiplicative_factor_number_node_switches * number_node_switches + \
                      self.multiplicative_factor_number_line_switches * number_line_switches
        return action_cost

    @staticmethod
    def __get_lines_capacity_usage(observation):
        ampere_flows = observation.ampere_flows
        thermal_limits = observation.thermal_limits
        lines_capacity_usage = np.divide(ampere_flows, thermal_limits)
        return lines_capacity_usage

    @staticmethod
    def __get_distance_reference_grid(observation):
        # Reference grid distance reward
        """ Computes the distance of the current observation with the reference grid (i.e. initial grid of the game).
        The distance is computed as the number of different nodes on which two identical elements are wired. For
        instance, if the production of first current substation is wired on the node 1, and the one of the first initial
        substation is wired on the node 0, then their is a distance of 1 (there are different) between the current and
        reference grid (for this production). The total distance is the sum of those values (0 or 1) for all the
        elements of the grid (productions, loads, origin of lines, extremity of lines).

        :return: the number of different nodes between the current topology and the initial one
        """
        #initial_topology = np.asarray(self.game.get_initial_topology())
        initial_topology = np.concatenate((observation.initial_productions_nodes, observation.initial_loads_nodes,
                                           observation.initial_lines_or_nodes, observation.initial_lines_ex_nodes))
        current_topology = np.concatenate((observation.productions_nodes, observation.loads_nodes,
                                           observation.lines_or_nodes, observation.lines_ex_nodes))

        return np.sum((initial_topology != current_topology))  # Sum of nodes that are different
