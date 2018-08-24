__author__ = 'marvinler'
import datetime

import os
import copy
import numpy as np
import pypownet.grid
from pypownet.scenarios_chronic import ScenariosChronic
from pypownet import root_path, ARTIFICIAL_NODE_STARTING_STRING


class Game(object):
    def __init__(self, grid_case, seed=None):
        """
        Initializes an instance of the game. This class is sufficient to play the game both as human and as AI.
        """
        dc_loadflow = True  # True if DC approximation for loadflow computation; False for AC
        if seed:
            np.random.seed(seed)

        # Check that the grid case is one of the expected
        if not isinstance(grid_case, int):
            raise ValueError('grid_case parameter should be an integer instead of', type(grid_case))
        if grid_case == 14:
            reference_grid = os.path.join(root_path, 'input/reference_grid14.m')
            chronic_folder = os.path.join(root_path, 'input/chronics/14/')
            new_slack_bus = 1
        elif grid_case == 118:
            reference_grid = os.path.join(root_path, 'input/reference_grid118.m')
            chronic_folder = os.path.join(root_path, 'input/chronics/118/')
            new_slack_bus = 69
        else:
            raise ValueError('The game does not currently support a grid with %d substations' % grid_case)
        self.grid_case = grid_case

        # Date variables
        self.initial_date = datetime.datetime(2017, month=1, day=2, hour=0, minute=0, second=0)
        self.timestep_date = datetime.timedelta(hours=1)
        self.current_date = self.initial_date

        # Checks that input reference grid/chronic folder do exist
        if not os.path.exists(reference_grid):
            raise FileExistsError('The reference grid %s does not exist' % reference_grid)
        if not os.path.exists(chronic_folder):
            raise FileExistsError('The chronic folder %s does not exist' % chronic_folder)

        # Loads the scenarios chronic and retrieve reference grid file
        self.chronic_folder = os.path.abspath(chronic_folder)
        self.chronic = ScenariosChronic(source_folder=self.chronic_folder)
        self.reference_grid_file = os.path.abspath(reference_grid)
        print('Using chronics folder', self.chronic_folder, 'and reference grid', self.reference_grid_file)

        # Retrieve all the pertinent values of the chronic
        self.scenarios_ids = self.chronic.get_scenarios_ids()
        self.number_scenarios = self.chronic.get_number_scenarios()
        self.current_scenario_id = None

        # Loads the grid in a container for the EmulGrid object given the current scenario + current RL state container
        self.grid = pypownet.grid.Grid(src_filename=self.reference_grid_file,
                                       dc_loadflow=dc_loadflow,
                                       new_slack_bus=new_slack_bus,
                                       new_imaps=self.chronic.get_imaps())
        # Save the initial topology to potentially reset the game (=reset topology)
        self.initial_topology = copy.deepcopy(self.grid.get_topology())

        # Loads first scenario
        self.load_next_scenario(do_trigger_lf_computation=True, cascading_failure=False, apply_cascading_output=True)

        # Graphical interface container
        self.gui = None

        self.epoch = 1
        self.timestep = 1

    def load_scenario(self, scenario_id, do_trigger_lf_computation, cascading_failure, apply_cascading_output):
        # Retrieve the Scenario object associated to the desired id
        scenario = self.chronic.get_scenario(scenario_id)
        # Loads the next scenario: will load values and compute loadflow to compute real flows
        self.grid.load_scenario(scenario, do_trigger_lf_computation=do_trigger_lf_computation,
                                cascading_failure=cascading_failure, apply_cascading_output=apply_cascading_output)
        self.current_scenario_id = scenario_id

    def load_next_scenario(self, do_trigger_lf_computation, cascading_failure, apply_cascading_output):
        """ Loads the next scenario, in the sense that it loads the scenario with the smaller greater id (scenarios ids are
        not  necessarly consecutive).

        :return: :raise ValueError: raised in the case where they are no more scenarios available
        """
        # If there are no more scenarios to be played, raise NoMoreScenarios exception
        if self.current_scenario_id == self.scenarios_ids[-1]:
            raise NoMoreScenarios('No more scenarios available')

        # If no scenario has been loaded so far, loads the first one
        if self.current_scenario_id is None:
            next_scenario_id = self.scenarios_ids[0]
        else:  # Otherwise loads the next one in the list of scenarios
            next_scenario_id = self.scenarios_ids[self.scenarios_ids.index(self.current_scenario_id) + 1]

        # Update date
        self.current_date += self.timestep_date

        return self.load_scenario(next_scenario_id, do_trigger_lf_computation, cascading_failure,
                                  apply_cascading_output)

    def get_current_scenario_id(self):
        """ Retrieves the current index of scenario; this index might differs from a natural counter (some id may be
        missing within the chronic).

        :return: an integer of the id of the current scenario loaded
        """
        return self.current_scenario_id

    @property
    def get_scenarios_ids(self):
        """ Retrieves the list of all pre-computed scenarios.

        :return: a list of integer representing the ordered ids of scenarios to be played in total
        """
        return self.scenarios_ids

    @property
    def get_number_scenarios(self):
        """ Retrieves the number of scenarios of the chronic.

        :return: integer
        """
        return self.number_scenarios

    def get_observation(self):
        """ Retrieves an observation of the current state of the grid.

        :return: an instance of class pypownet.env.RunEnv.Observation
        """
        return self.grid.export_to_observation()

    def get_initial_topology(self, as_array=False):
        """ Retrieves the initial topology of the grid (when it was initially loaded). This is notably used to
        reinitialize the grid after a game over.

        :param as_array: True to return only one vector (concatenated-style), False to return the split topology (see
        class pypownet.grid.Topology
        :return: an instance of pypownet.grid.Topology or a list of integers
        """
        if as_array:
            return self.initial_topology.get_zipped()
        return self.initial_topology

    def get_date(self):
        """ Returns the current date of the game being played.

        :return:
        """
        return self.current_date

    def apply_action(self, action):
        """ Applies an action on the current grid (topology). The action is first into lists of same objects (e.g. nodes
        on which productions are connected), then the destination values are computed, such that the grid will replace
        its current topology with the latter. Since actions come from pypownet.env.RunEnv.Action, they are switches.
        Here, given the last values of the grid and the switches, this function computes the actual destination values
        (e.g. switch line status of line 10: if line 10 is on, put its status to off i.e. 0, otherwise put to on i.e. 1)

        :param action: an instance of pypownet.env.RunEnv.Action
        """
        self.timestep += 1
        # If there is no action, then no need to apply anything on the grid
        if action is None:
            return

        # Convert the action into the corresponding topology vector
        prods_nodes, loads_nodes, lines_or_nodes, line_ex_nodes, lines_service = self.grid.get_topology().get_unzipped()
        a_prods_nodes, a_loads_nodes, a_lines_or_nodes, a_line_ex_nodes, a_lines_service = \
            pypownet.grid.Topology.unzip(action, len(prods_nodes), len(loads_nodes), len(lines_service),
                                         self.grid.get_topology().invert_mapping_permutation)

        prods_nodes = np.where(a_prods_nodes, 1 - prods_nodes, prods_nodes)
        loads_nodes = np.where(a_loads_nodes, 1 - loads_nodes, loads_nodes)
        lines_or_nodes = np.where(a_lines_or_nodes, 1 - lines_or_nodes, lines_or_nodes)
        lines_ex_nodes = np.where(a_line_ex_nodes, 1 - line_ex_nodes, line_ex_nodes)
        lines_service = np.where(a_lines_service, 1 - lines_service, lines_service)
        new_topology = pypownet.grid.Topology(prods_nodes=prods_nodes,
                                              loads_nodes=loads_nodes,
                                              lines_or_nodes=lines_or_nodes,
                                              lines_ex_nodes=lines_ex_nodes,
                                              lines_service=lines_service)

        self.grid.apply_topology(new_topology)

    def compute_loadflow(self, cascading_failure, apply_cascading_output):
        """ Wrapper to call the computed a loadflow of the current grid state.

        :param cascading_failure: True to perform cascading failure
        :param apply_cascading_output: True to apply the lines breaks of cascading failure to future grid state
        :return: 0 failure, 1 success
        """
        return self.grid.compute_loadflow(perform_cascading_failure=cascading_failure,
                                          apply_cascading_output=apply_cascading_output)

    def reset_grid(self):
        """ Reinitialized the grid by applying the initial topology to the current state (topology).
        """
        self.grid.apply_topology(self.initial_topology)

    def reset(self, restart):
        """ Resets the game: put the grid topology to the initial one. Besides, if restart is True, then the game will
        load the first set of injections (i)_{t0}, otherwise the next set of injections of the chronics (i)_{t+1}

        :param restart: True to restart the chronic, else pursue with next timestep
        """
        self.reset_grid()
        self.epoch += 1
        if restart:  # If restart, put current id to None so that load_next will load first timestep
            self.current_scenario_id = None
            self.timestep = 1

        self.load_next_scenario(do_trigger_lf_computation=True, cascading_failure=False, apply_cascading_output=True)

    def step(self, action, cascading_failure, apply_cascading_output):
        self.apply_action(action)

        try:
            self.load_next_scenario(do_trigger_lf_computation=True,
                                    cascading_failure=cascading_failure, apply_cascading_output=apply_cascading_output)
        except (NoMoreScenarios, pypownet.grid.GridNotConnexeException, pypownet.grid.DivergingLoadflowException) as e:
            raise e

        # Compute the number of cut loads
        n_isolated_loads = self.grid._count_isolated_loads()

        return n_isolated_loads

    def simulate(self, action, cascading_failure, apply_cascading_output):
        before_topology = self.grid.get_topology()
        before_scenario_id = self.current_scenario_id

        # Step the action
        try:
            self.step(action, cascading_failure, apply_cascading_output)
        except Exception as e:
            # Put past values back for topo and injection
            self.grid.apply_topology(before_topology)
            self.load_scenario(before_scenario_id, do_trigger_lf_computation=True,
                               cascading_failure=cascading_failure, apply_cascading_output=apply_cascading_output)
            raise e

        # If no error raised, return the simulated output observation, such that reward can be computed, then
        # put topological and injections values back
        simulated_state = self.get_observation()
        # Put past values back for topo and injection
        self.grid.apply_topology(before_topology)
        self.load_scenario(before_scenario_id, do_trigger_lf_computation=True,
                           cascading_failure=cascading_failure, apply_cascading_output=apply_cascading_output)

        return simulated_state

    def _render(self, rewards, last_action, close=False, game_over=False):
        """ Initializes the renderer if not already done, then compute the necessary values to be carried to the
        renderer class (e.g. sum of consumptions).

        :param rewards: list of subrewards of the last timestep (used to plot reward per timestep)
        :param close: True to close the application
        :param game_over: True to plot a "Game over!" over the screen if game is over
        :return: :raise ImportError: pygame not found raises an error (it is mandatory for the renderer)
        """

        def initialize_renderer():
            """ initializes the pygame gui with the parameters necessary to e.g. plot colors of productions """
            pygame.init()

            # Compute an id mapping helper for line plotting
            mpcbus = self.grid.mpc['bus']
            half_nodes_ids = mpcbus[:len(mpcbus) // 2, 0]
            node_to_substation = lambda x: int(float(str(x).replace(ARTIFICIAL_NODE_STARTING_STRING, '')))
            # Retrieve true substations ids of origins and extremities
            nodes_or_ids = np.asarray(list(map(node_to_substation, self.grid.mpc['branch'][:, 0])))
            nodes_ex_ids = np.asarray(list(map(node_to_substation, self.grid.mpc['branch'][:, 1])))
            idx_or = [np.where(half_nodes_ids == or_id)[0][0] for or_id in nodes_or_ids]
            idx_ex = [np.where(half_nodes_ids == ex_id)[0][0] for ex_id in nodes_ex_ids]

            # Retrieve vector of size nodes with 0 if no prod (resp load) else 1
            mpcgen = self.grid.mpc['gen']
            nodes_ids = mpcbus[:, 0]
            prods_ids = mpcgen[:, 0]
            are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
                                      [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
            are_loads = np.logical_or(self.grid.are_loads[:len(mpcbus) // 2],
                                      self.grid.are_loads[len(nodes_ids) // 2:])

            from pypownet.renderer import Renderer

            return Renderer(self.grid_case, idx_or, idx_ex, are_prods, are_loads)

        try:
            import pygame
        except ImportError as e:
            raise ImportError("{}. (HINT: install pygame using `pip install pygame`)".format(e))

        if close:
            pygame.quit()

        if self.gui is None:
            self.gui = initialize_renderer()

        # Retrieve lines capacity usage (for plotting power lines with appropriate colors and widths)
        lines_capacity_usage = self.grid.export_lines_capacity_usage()
        prods_values = self.grid.mpc['gen'][:, 1]
        loads_values = self.grid.mpc['bus'][self.grid.are_loads, 2]
        lines_por_values = self.grid.mpc['branch'][:, 13]
        lines_service_status = self.grid.mpc['branch'][:, 10]

        substations_ids = self.grid.mpc['bus'][self.grid.n_nodes // 2:]
        # Based on the action, determine if substations has been touched (i.e. there was a topological change involved
        # in the associated substation)
        has_been_changed = np.zeros((len(substations_ids),))
        if last_action is not None:
            n_elements_substations = self.grid.number_elements_per_substations
            offset = 0
            for i, (substation_id, n_elements) in enumerate(zip(substations_ids, n_elements_substations)):
                has_been_changed[i] = np.any(last_action[offset:offset + n_elements] != 0)
                offset += n_elements

        self.gui.render(lines_capacity_usage, lines_por_values, lines_service_status,
                        self.epoch, self.timestep, self.current_scenario_id,
                        prods=prods_values, loads=loads_values, last_timestep_rewards=rewards,
                        date=self.current_date, are_substations_changed=has_been_changed, game_over=game_over)


# Exception to be risen when no more scenarios are available to be played (i.e. every scenario has been played)
class NoMoreScenarios(Exception):
    pass


class IllegalActionException(Exception):
    pass


class LoadCutException(Exception):
    pass
