__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import datetime
import logging

import os
import copy
import numpy as np
import pypownet.grid
import pypownet.environment
from pypownet.chronic import Chronic
from pypownet import root_path, ARTIFICIAL_NODE_STARTING_STRING


# Exception to be risen when no more scenarios are available to be played (i.e. every scenario has been played)
class NoMoreScenarios(Exception):
    pass


class Action(object):
    def __init__(self, topological_subaction, lines_status_subaction):
        if topological_subaction is None or lines_status_subaction is None:
            raise ValueError('Expected first argument of %s to be an array, got None' % self.__class__)
        self.topological_subaction = np.asarray(topological_subaction).astype(int)
        self.lines_status_subaction = np.asarray(lines_status_subaction).astype(int)

        self._topo_length = len(self.topological_subaction)
        self._linestat_length = len(self.lines_status_subaction)

    def get_topological_subaction(self):
        return self.topological_subaction

    def get_lines_status_subaction(self):
        return self.lines_status_subaction

    def as_array(self):
        return np.concatenate((self.topological_subaction, self.lines_status_subaction))

    def __str__(self):
        return 'topological subaction: %s; line status subaction: %s' % (
            '[%s]' % ', '.join(list(map(str, self.topological_subaction))),
            '[%s]' % ', '.join(list(map(str, self.lines_status_subaction))),)

    def __len__(self, do_sum=True):
        length_aslist = (len(self.topological_subaction), len(self.lines_status_subaction))
        return sum(length_aslist) if do_sum else length_aslist

    def __setitem__(self, item, value):
        item %= len(self)
        if item < self._topo_length:
            self.topological_subaction.__setitem__(item, value)
            #self.topological_subaction[item] = value
        else:
            self.lines_status_subaction.__setitem__(item % self._topo_length, value)
            #self.lines_status_subaction[item % self._topo_length] = value


class Game(object):
    def __init__(self, grid_case, start_id=0, seed=None):
        """ Initializes an instance of the game. This class is sufficient to play the game both as human and as AI.
        """
        dc_loadflow = True  # True if DC approximation for loadflow computation; False for AC
        if seed:
            np.random.seed(seed)

        # Check that the grid case is one of the expected
        if not isinstance(grid_case, int):
            raise ValueError('grid_case parameter should be an integer instead of', type(grid_case))

        reference_grid = os.path.join(root_path, 'input/reference_grid%d.m' % grid_case)
        chronic_folder = os.path.join(root_path, 'input/chronics/%d/' % grid_case)
        if not os.path.exists(reference_grid) or not os.path.exists(chronic_folder):
            raise FileNotFoundError('Grid case %d not currently handled by the software' % grid_case)

        # Todo: refacto, should not change new slack bus as should be mandatory in reference grid
        new_slack_bus = 69 if grid_case == 118 else 1

        self.grid_case = grid_case

        # Configuration parameters
        self.apply_cascading_output = True

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
        self.__chronic_folder = os.path.abspath(chronic_folder)
        self.__chronic = Chronic(source_folder=self.__chronic_folder)
        self.reference_grid_file = os.path.abspath(reference_grid)

        # Retrieve all the pertinent values of the chronic
        self.timesteps_ids = self.__chronic.get_timestep_ids()
        self.current_timestep_id = None

        # Loads the grid in a container for the EmulGrid object given the current scenario + current RL state container
        self.grid = pypownet.grid.Grid(src_filename=self.reference_grid_file,
                                       dc_loadflow=dc_loadflow,
                                       new_slack_bus=new_slack_bus,
                                       new_imaps=self.__chronic.get_imaps())
        # Save the initial topology (explicitely create another copy) + voltage angles and magnitudes of buses
        self.initial_topology = copy.deepcopy(self.grid.get_topology())
        self.initial_lines_service = copy.deepcopy(self.grid.get_lines_status())
        self.initial_voltage_magnitudes = copy.deepcopy(self.grid.mpc['bus'][:, 7])
        self.initial_voltage_angles = copy.deepcopy(self.grid.mpc['bus'][:, 8])

        # Instantiate the counter of timesteps before lines can be reconnected (one value per line)
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))
        self.timesteps_before_planned_maintenance = np.zeros((self.grid.n_lines,))

        self.gui = None
        self.epoch = 1
        self.timestep = 1

        self.coefficient_hard_thermal_limit = 1.5
        self.n_timesteps_thermal_limit_hard_broken = 10
        self.n_timesteps_thermal_limit_soft_broken = 5
        self.n_timesteps_overflowed_lines = np.zeros((self.grid.n_lines,))
        self.n_timesteps_overflowed_lines_break = 3

        self.logger = logging.getLogger('pypownet.' + __name__)

        # Loads first scenario
        self.load_entries_from_next_timestep(starting_timestep_id=start_id)
        self.compute_loadflow_cascading(fname_end='_init.m')

    def get_number_elements(self):
        return self.grid.get_number_elements()

    def get_current_timestep_id(self):
        """ Retrieves the current index of scenario; this index might differs from a natural counter (some id may be
        missing within the chronic).

        :return: an integer of the id of the current scenario loaded
        """
        return self.current_timestep_id

    def _get_planned_maintenance(self, horizon=10):
        timestep_id = self.current_timestep_id
        return self.__chronic.get_planned_maintenance(timestep_id, horizon)

    def get_initial_topology(self):
        """ Retrieves the initial topology of the grid (when it was initially loaded). This is notably used to
        reinitialize the grid after a game over.

        :return: an instance of pypownet.grid.Topology or a list of integers
        """
        return self.initial_topology.get_zipped()

    def load_entries_from_timestep_id(self, timestep_id):
        # Retrieve the Scenario object associated to the desired id
        timestep_entries = self.__chronic.get_timestep_entries(timestep_id)

        # Loads the next timestep injections: PQ and PV and gen status
        self.grid.load_timestep_injections(timestep_entries)

        # Integration of timestep maintenance: disco lines for which current maintenance not 0 (equal to time to wait)
        timestep_maintenance = timestep_entries.get_maintenance()
        mask_affected_lines = timestep_maintenance > 0
        self.grid.mpc['branch'][mask_affected_lines, 10] = 0
        assert not np.any(timestep_maintenance[mask_affected_lines] == 0), 'Line maintenance cant last for 0 timestep'

        # For cases when multiple events (mtn, hazards) can overlap, put the max time to wait as new value between new
        # time to wait and previous one
        self.timesteps_before_lines_reconnectable[mask_affected_lines] = np.max(
            np.vstack((self.timesteps_before_lines_reconnectable[mask_affected_lines],
                       timestep_maintenance[mask_affected_lines],)), axis=0)

        # Logs data about lines just put into maintenance
        n_maintained = sum(mask_affected_lines)
        if n_maintained > 0:
            self.logger.info(
                '  MAINTENANCE: switching off line%s %s for %s%s timestep%s' %
                ('s' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '#%.0f' % i, self.grid.ids_lines[mask_affected_lines]))),
                 'resp. ' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '%.0f' % i, timestep_maintenance[mask_affected_lines]))),
                 's' if n_maintained > 1 or np.any(timestep_maintenance[mask_affected_lines] > 1) else '')
            )

        # Integration of timestep hazards: disco freshly broken lines (just now)
        timestep_hazards = timestep_entries.get_hazards()
        mask_affected_lines = timestep_hazards > 0
        self.grid.mpc['branch'][mask_affected_lines, 10] = 0
        assert not np.any(timestep_hazards[mask_affected_lines] == 0), 'Line hazard cant last for 0 timestep'

        # For cases when multiple events (mtn, hazards) can overlap, put the max time to wait as new value between new
        # time to wait and previous one
        self.timesteps_before_lines_reconnectable[mask_affected_lines] = np.max(
            np.vstack((self.timesteps_before_lines_reconnectable[mask_affected_lines],
                       timestep_hazards[mask_affected_lines],)), axis=0)

        # Logs data about lines just broke from hazards
        n_maintained = sum(mask_affected_lines)
        if n_maintained > 0:
            self.logger.info(
                '  HAZARD: line%s %s broke; reparations will take %s%s timestep%s' %
                ('s' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '#%.0f' % i, self.grid.ids_lines[mask_affected_lines]))),
                 'resp. ' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '%.0f' % i, timestep_hazards[mask_affected_lines]))),
                 's' if n_maintained > 1 or np.any(timestep_hazards[mask_affected_lines] > 1) else '')
            )

        self.current_timestep_id = timestep_id  # Update id of current timestep after whole entries are in place

    def load_entries_from_next_timestep(self, starting_timestep_id=None, decrement_reconnectable_timesteps=False):
        """ Loads the next timestep injections (set of injections, maintenance, hazard etc for the next timestep id).

        :return: :raise ValueError: raised in the case where they are no more scenarios available
        """
        # If there are no more timestep to be played, raise NoMoreScenarios exception
        if self.current_timestep_id == self.timesteps_ids[-1]:
            raise NoMoreScenarios('All timesteps have been played.')

        # If no timestep injections has been loaded so far, loads the first one
        if self.current_timestep_id is None:
            next_timestep_id = self.timesteps_ids[0] if starting_timestep_id is None else starting_timestep_id
        else:  # Otherwise loads the next one in the list of timesteps injections
            next_timestep_id = self.timesteps_ids[self.timesteps_ids.index(self.current_timestep_id) + 1]

        # Update date
        self.current_date += self.timestep_date

        # If the method is not simulate, decrement the actual timesteps to wait for the crashed lines (real step call)
        if decrement_reconnectable_timesteps:
            self.timesteps_before_lines_reconnectable[self.timesteps_before_lines_reconnectable > 0] -= 1

        self.load_entries_from_timestep_id(next_timestep_id)

    def compute_loadflow_cascading(self, fname_end=None):
        try:
            self.grid.compute_loadflow(fname_end)
            ####### HACK GUI
            if self.gui is not None:
                self._render(None, self.last_action)
            ####### \HACK GUI
            self._compute_cascading_failure(apply_cascading_output=self.apply_cascading_output)
        except pypownet.grid.DivergingLoadflowException as e:
            raise e

    def _compute_cascading_failure(self, apply_cascading_output):
        mpc_before = self.grid.mpc if apply_cascading_output else copy.deepcopy(self.grid.mpc)

        depth = 1  # Count cascading depth
        is_done = False
        over_hard_thlim_lines = np.full(self.grid.n_lines, False)
        # Will loop undefinitely until an exception is raised (~outage) or the grid has no overflowed line
        while not is_done:
            is_done = True  # Reset is_done: if a line is broken bc of cascading failure, then is_done=False
            current_flows_a = self.grid.extract_flows_a()
            thermal_limits = self.grid.get_thermal_limits()

            over_thlim_lines = current_flows_a > thermal_limits  # Mask of overflowed lines
            # Computes the number of overflows: if 0, exit now for software speed
            if np.sum(over_thlim_lines) == 0:
                break

            # Checks for lines over hard nominal thermal limit
            over_hard_thlim_lines = current_flows_a > self.coefficient_hard_thermal_limit * thermal_limits
            if np.any(over_hard_thlim_lines):
                # Break lines over their hard thermal limits: set status to 0 and increment timesteps before reconn.
                self.grid.get_lines_status()[over_hard_thlim_lines] = 0
                if apply_cascading_output:
                    self.timesteps_before_lines_reconnectable[
                        over_hard_thlim_lines] = self.n_timesteps_thermal_limit_hard_broken
                is_done = False
            # Those lines have been treated so discard them for further depth process
            over_thlim_lines[over_hard_thlim_lines] = False

            # Checks for soft-overflowed lines among remaining lines
            if np.any(over_thlim_lines):
                time_limit = self.n_timesteps_overflowed_lines_break
                number_timesteps_over_thlim_lines = self.n_timesteps_overflowed_lines[over_thlim_lines]
                soft_broken_lines = number_timesteps_over_thlim_lines >= time_limit
                if np.any(soft_broken_lines):
                    # Soft break lines overflowed for more timesteps than the limit
                    self.grid.get_lines_status()[soft_broken_lines] = 0
                    if apply_cascading_output:
                        self.timesteps_before_lines_reconnectable[
                            soft_broken_lines] = self.n_timesteps_thermal_limit_soft_broken
                    is_done = False
                    # Do not consider those lines anymore
                    over_hard_thlim_lines[soft_broken_lines] = False

            try:
                self.grid.compute_loadflow(fname_end='_cascading%d.m' % depth)
            except pypownet.grid.DivergingLoadflowException as e:
                e.text += ': casading failure of depth %d has diverged' % depth
                raise e

            ####### HACK GUI
            if self.gui is not None:
                self._render(None, self.last_action)
            ####### \HACK GUI

            depth += 1

        # At the end of the cascading failure, decrement timesteps waited by overflowed lines
        self.n_timesteps_overflowed_lines[over_hard_thlim_lines] += 1
        self.n_timesteps_overflowed_lines[~over_hard_thlim_lines] = 0

        if not apply_cascading_output:
            self.grid.mpc = mpc_before

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
            raise ValueError('Cannot play None action')

        # Retrieve current grid nodes + mapping arrays for unshuffling the topological subaction of action
        grid_topology = self.grid.get_topology()
        grid_topology_mapping_array = grid_topology.mapping_array
        grid_topology_invert_mapping_function = grid_topology.invert_mapping_permutation
        prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes = grid_topology.get_unzipped()
        # Retrieve lines status service of current grid
        lines_service = self.grid.get_lines_status()

        action_lines_service = action.get_lines_status_subaction()
        action_topology = action.get_topological_subaction()
        # Split the action into 5 parts: prods nodes, loads nodes, lines or/ex nodes and lines status
        unzipped_action = pypownet.grid.Topology.unzip(action_topology,
                                                       len(prods_nodes), len(loads_nodes), len(lines_service),
                                                       grid_topology_invert_mapping_function)
        action_prods_nodes = unzipped_action[0]
        action_loads_nodes = unzipped_action[1]
        action_lines_or_nodes = unzipped_action[2]
        action_lines_ex_nodes = unzipped_action[3]

        # Verify that the player is not intended to reconnect not reconnectable lines (broken or being maintained);
        # here, we check for lines switches, because if a line is broken, then its status is already to 0, such that a
        # switch will switch on the power line
        to_be_switched_lines = np.equal(action_lines_service, 1)
        broken_lines = self.timesteps_before_lines_reconnectable > 0  # Mask of broken lines
        illegal_lines_reconnections = np.logical_and(to_be_switched_lines, broken_lines)

        # Raises an exception if there is some attempt to reconnect broken lines; Game.step should manage what to do in
        # such a case
        if np.any(illegal_lines_reconnections):
            timesteps_to_wait = self.timesteps_before_lines_reconnectable[illegal_lines_reconnections]
            assert not np.any(timesteps_to_wait <= 0), 'Should not happen'

            # Creates strings for log printing
            non_reconnectable_lines_as_str = ', '.join(
                list(map(str, np.arange(self.grid.n_lines)[illegal_lines_reconnections])))
            timesteps_to_wait_as_str = ', '.join(list(map(lambda x: str(int(x)), timesteps_to_wait)))

            number_invalid_reconnections = np.sum(illegal_lines_reconnections)
            if number_invalid_reconnections > 1:
                timesteps_to_wait_as_str = 'resp. ' + timesteps_to_wait_as_str

            raise pypownet.environment.IllegalActionException(
                'Trying to reconnect broken line%s %s, must wait %s timesteps. ' % (
                    's' if number_invalid_reconnections > 1 else '',
                    non_reconnectable_lines_as_str, timesteps_to_wait_as_str), illegal_lines_reconnections)

        # Compute the destination nodes of all elements + the lines service finale values: actions are switches
        prods_nodes = np.where(action_prods_nodes, 1 - prods_nodes, prods_nodes)
        loads_nodes = np.where(action_loads_nodes, 1 - loads_nodes, loads_nodes)
        lines_or_nodes = np.where(action_lines_or_nodes, 1 - lines_or_nodes, lines_or_nodes)
        lines_ex_nodes = np.where(action_lines_ex_nodes, 1 - lines_ex_nodes, lines_ex_nodes)
        new_topology = pypownet.grid.Topology(prods_nodes=prods_nodes, loads_nodes=loads_nodes,
                                              lines_or_nodes=lines_or_nodes, lines_ex_nodes=lines_ex_nodes,
                                              mapping_array=grid_topology_mapping_array)

        new_lines_service = np.where(action_lines_service, 1 - lines_service, lines_service)

        # Apply the newly computed destination topology to the grid
        self.grid.apply_topology(new_topology)
        # Apply the newly computed destination topology to the grid
        self.grid.set_lines_status(new_lines_service)

    def reset(self, restart):
        """ Resets the game: put the grid topology to the initial one. Besides, if restart is True, then the game will
        load the first set of injections (i)_{t0}, otherwise the next set of injections of the chronics (i)_{t+1}

        :param restart: True to restart the chronic, else pursue with next timestep
        """
        self.reset_grid()
        self.epoch += 1
        if restart:  # If restart, put current id to None so that load_next will load first timestep
            self.current_timestep_id = None
            self.timestep = 1

        try:
            self.load_entries_from_next_timestep()
            self.compute_loadflow_cascading(fname_end='_resetted.m')
        # If after reset there is a diverging loadflow, then recall reset w/o penalty (not the player's fault)
        except pypownet.grid.DivergingLoadflowException as e:
            self.logger.error(e)
            self.reset(restart)

    def reset_grid(self):
        """ Reinitialized the grid by applying the initial topology to the current state (topology).
        """
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))

        self.grid.apply_topology(self.initial_topology)
        self.grid.mpc['gen'][:, 7] = 1  # Put all prods status to 1 (ON)
        self.grid.set_lines_status(self.initial_lines_service)  # Reset lines status
        self.grid.discard_flows()  # Discard flows: not mandatory

        # Reset voltage magnitude and angles: they change when using AC mode
        self.grid.set_voltage_angles(self.initial_voltage_angles)
        self.grid.set_voltage_magnitudes(self.initial_voltage_magnitudes)

    def step(self, action, decrement_reconnectable_timesteps=True):
        # Apply action, or raises eception if some broken lines are attempted to be switched on
        try:
            self.last_action = action  # tmp
            self.apply_action(action)
        except pypownet.environment.IllegalActionException as e:
            e.text += ' Ignoring action switches of broken lines.'
            # If broken lines are attempted to be switched on, put the switches to 0
            illegal_lines_reconnections = e.illegal_lines_reconnections
            action.lines_status_subaction[illegal_lines_reconnections] = 0
            assert np.sum(action.get_lines_status_subaction()[illegal_lines_reconnections]) == 0

            # Resubmit step with modified valid action and return either exception of new step, or this exception
            obs, correct_step, done = self.step(action, decrement_reconnectable_timesteps)
            return obs, correct_step if correct_step else e, done  # Return done and not False because step might
            # diverge

        try:
            # Load next timestep entries, compute one loadflow, then potentially cascading failure
            self.load_entries_from_next_timestep(decrement_reconnectable_timesteps=decrement_reconnectable_timesteps)
            self.compute_loadflow_cascading(fname_end='_after_entries.m')
        except (NoMoreScenarios, pypownet.grid.DivergingLoadflowException) as e:
            return None, e, True

        return self.export_observation(), None, False

    def simulate(self, action, cascading_failure, apply_cascading_output):
        before_topology = copy.deepcopy(self.grid.get_topology())
        before_timestep_id = self.current_timestep_id

        # Step the action
        try:
            self.step(action, decrement_reconnectable_timesteps=False)
        except pypownet.grid.DivergingLoadflowException as e:
            # Put past values back for topo and injection
            self.grid.apply_topology(before_topology)
            self.load_entries_from_timestep_id(before_timestep_id)
            raise e

        # If no error raised, return the simulated output observation, such that reward can be computed, then
        # put topological and injections values back
        simulated_state = self.export_observation()
        # Put past values back for topo and injection
        self.grid.apply_topology(before_topology)
        self.load_entries_from_timestep_id(before_timestep_id)

        return simulated_state

    def export_observation(self):
        """ Retrieves an observation of the current state of the grid.

        :return: an instance of class pypownet.env.Observation
        """
        observation = self.grid.export_to_observation()
        # Fill additional parameters: starts with substations ids of all elements
        observation.timesteps_before_lines_reconnectable = self.timesteps_before_lines_reconnectable
        self.timesteps_before_planned_maintenance = self._get_planned_maintenance()
        observation.timesteps_before_planned_maintenance = self.timesteps_before_planned_maintenance

        return observation

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
        lines_capacity_usage = self.grid.export_lines_capacity_usage(safe_mode=True)

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
                has_been_changed[i] = np.any(
                    [l != 0 for l in last_action.get_topological_subaction()[offset:offset + n_elements]])
                offset += n_elements

        self.gui.render(lines_capacity_usage, lines_por_values, lines_service_status,
                        self.epoch, self.timestep, self.current_timestep_id,
                        prods=prods_values, loads=loads_values, last_timestep_rewards=rewards,
                        date=self.current_date, are_substations_changed=has_been_changed, game_over=game_over)
