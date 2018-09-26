__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import logging

from time import sleep
import copy
import numpy as np
import pypownet.grid
import pypownet.environment
from pypownet.chronic import Chronic, ChronicLooper
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
from pypownet.parameters import Parameters


# Exception to be risen when no more scenarios are available to be played (i.e. every scenario has been played)
class NoMoreScenarios(Exception):
    pass


class DivergingLoadflowException(pypownet.grid.DivergingLoadflowException):
    def __init__(self, last_observation, *args):
        super(DivergingLoadflowException, self).__init__(last_observation, *args)


class TooManyProductionsCut(Exception):
    def __init__(self, *args):
        super(TooManyProductionsCut, self).__init__(*args)
        self.text = args[0]


class TooManyConsumptionsCut(Exception):
    def __init__(self, *args):
        super(TooManyConsumptionsCut, self).__init__(*args)
        self.text = args[0]


class Action(object):
    def __init__(self, prods_switches_subaction, loads_switches_subaction,
                 lines_or_switches_subaction, lines_ex_switches_subaction, lines_status_subaction):
        if prods_switches_subaction is None:
            raise ValueError('Expected prods_switches_subaction to be array, got None')
        if loads_switches_subaction is None:
            raise ValueError('Expected loads_switches_subaction to be array, got None')
        if lines_or_switches_subaction is None:
            raise ValueError('Expected lines_or_switches_subaction to be array, got None')
        if lines_ex_switches_subaction is None:
            raise ValueError('Expected lines_ex_switches_subaction to be array, got None')
        if lines_status_subaction is None:
            raise ValueError('Expected lines_status_subaction to be array, got None')

        self.prods_switches_subaction = np.asarray(prods_switches_subaction).astype(int)
        self.loads_switches_subaction = np.asarray(loads_switches_subaction).astype(int)
        self.lines_or_switches_subaction = np.asarray(lines_or_switches_subaction).astype(int)
        self.lines_ex_switches_subaction = np.asarray(lines_ex_switches_subaction).astype(int)
        self.lines_status_subaction = np.asarray(lines_status_subaction).astype(int)

        self._prods_switches_length = len(self.prods_switches_subaction)
        self._loads_switches_length = len(self.loads_switches_subaction)
        self._lines_or_switches_length = len(self.lines_or_switches_subaction)
        self._lines_ex_switches_length = len(self.lines_ex_switches_subaction)
        self._lines_status_length = len(self.lines_status_subaction)

    def get_prods_switches_subaction(self):
        return self.prods_switches_subaction

    def get_loads_switches_subaction(self):
        return self.loads_switches_subaction

    def get_lines_or_switches_subaction(self):
        return self.lines_or_switches_subaction

    def get_lines_ex_switches_subaction(self):
        return self.lines_ex_switches_subaction

    def get_node_splitting_subaction(self):
        return np.concatenate((self.get_prods_switches_subaction(), self.get_loads_switches_subaction(),
                               self.get_lines_or_switches_subaction(), self.get_lines_ex_switches_subaction(),))

    def get_lines_status_subaction(self):
        return self.lines_status_subaction

    def as_array(self):
        return np.concatenate((self.get_node_splitting_subaction(), self.get_lines_status_subaction(),))

    def __str__(self):
        return ', '.join(list(map(str, self.as_array())))

    def __len__(self, do_sum=True):
        length_aslist = (self._prods_switches_length, self._loads_switches_length, self._lines_or_switches_length,
                         self._lines_ex_switches_length, self._lines_status_length)
        return sum(length_aslist) if do_sum else length_aslist

    def __setitem__(self, item, value):
        item %= len(self)
        if item < self._prods_switches_length:
            self.prods_switches_subaction.__setitem__(item, value)
        elif item < self._prods_switches_length + self._loads_switches_length:
            self.loads_switches_subaction.__setitem__(item % self._prods_switches_length, value)
        elif item < self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length:
            self.lines_or_switches_subaction.__setitem__(
                item % (self._prods_switches_length + self._loads_switches_length), value)
        elif item < self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length + \
                self._lines_ex_switches_length:
            self.lines_ex_switches_subaction.__setitem__(item % (self._prods_switches_length +
                                                                 self._loads_switches_length +
                                                                 self._lines_or_switches_length),
                                                         value)
        else:
            self.lines_status_subaction.__setitem__(
                item % (self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length +
                        self._lines_ex_switches_length), value)

    def __getitem__(self, item):
        item %= len(self)
        if item < self._prods_switches_length:
            self.prods_switches_subaction.__getitem__(item)
        elif item < self._prods_switches_length + self._loads_switches_length:
            self.loads_switches_subaction.__getitem__(item % self._prods_switches_length)
        elif item < self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length:
            self.lines_or_switches_subaction.__getitem__(
                item % (self._prods_switches_length + self._loads_switches_length))
        elif item < self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length + \
                self._lines_ex_switches_length:
            self.lines_ex_switches_subaction.__getitem__(
                item % (self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length))
        else:
            self.lines_status_subaction.__getitem__(
                item % (self._prods_switches_length + self._loads_switches_length + self._lines_or_switches_length +
                        self._lines_ex_switches_length))


class Game(object):
    def __init__(self, parameters_folder, game_level, chronic_looping_mode, chronic_starting_id,
                 game_over_mode, renderer_frame_latency=None):
        """ Initializes an instance of the game. This class is sufficient to play the game both as human and as AI.
        """
        self.logger = logging.getLogger('pypownet.' + __name__)

        # Read parameters
        self.__parameters = Parameters(parameters_folder, game_level)
        loadflow_backend = self.__parameters.get_loadflow_backend()
        self.is_mode_dc = self.__parameters.is_dc_mode()
        self.hard_overflow_coefficient = self.__parameters.get_hard_overflow_coefficient()
        self.n_timesteps_hard_overflow_is_broken = self.__parameters.get_n_timesteps_hard_overflow_is_broken()
        self.n_timesteps_consecutive_soft_overflow_breaks = \
            self.__parameters.get_n_timesteps_consecutive_soft_overflow_breaks()
        self.n_timesteps_soft_overflow_is_broken = self.__parameters.get_n_timesteps_soft_overflow_is_broken()
        self.n_timesteps_horizon_maintenance = self.__parameters.get_n_timesteps_horizon_maintenance()
        self.max_number_prods_game_over = self.__parameters.get_max_number_prods_game_over()
        self.max_number_loads_game_over = self.__parameters.get_max_number_loads_game_over()

        # Seek and load chronic
        self.__chronic_looper = ChronicLooper(chronics_folder=self.__parameters.get_chronics_path(),
                                              game_level=game_level, start_id=chronic_starting_id,
                                              looping_mode=chronic_looping_mode)
        self.__chronic = self.get_next_chronic()

        self.game_over_mode = game_over_mode

        # Seek and load starting reference grid
        self.grid = pypownet.grid.Grid(loadflow_backend=loadflow_backend,
                                       src_filename=self.__parameters.get_reference_grid_path(),
                                       dc_loadflow=self.is_mode_dc,
                                       new_imaps=self.__chronic.get_imaps())
        # Container that counts the consecutive timesteps lines are soft-overflows
        self.n_timesteps_soft_overflowed_lines = np.zeros((self.grid.n_lines,))

        # Retrieve all the pertinent values of the chronic
        self.current_timestep_id = None
        self.current_date = None
        self.current_timestep_entries = None
        self.previous_timestep = None  # Hack for renderer
        self.previous_date = None  # Hack for renderer
        self.n_loads_cut, self.n_prods_cut = 0, 0  # for renderer

        # Save the initial topology (explicitely create another copy) + voltage angles and magnitudes of buses
        self.initial_topology = copy.deepcopy(self.grid.get_topology())
        self.initial_lines_service = copy.deepcopy(self.grid.get_lines_status())
        self.initial_voltage_magnitudes = copy.deepcopy(self.grid.mpc['bus'][:, 7])
        self.initial_voltage_angles = copy.deepcopy(self.grid.mpc['bus'][:, 8])

        self.substations_ids = self.grid.mpc['bus'][:self.grid.n_nodes // 2, 0]

        # Instantiate the counter of timesteps before lines can be reconnected (one value per line)
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))

        # Renderer params
        self.renderer = None
        self.latency = renderer_frame_latency  # Sleep time after each frame plot (multiple frame plots per timestep)
        self.last_action = None
        self.epoch = 1
        self.timestep = 1

        self.get_reward_signal_class = self.__parameters.get_reward_signal_class()

        # Loads first scenario
        self.load_entries_from_next_timestep()
        self._compute_loadflow_cascading()

    def get_max_seconds_per_timestep(self):
        return self.__parameters.get_max_seconds_per_timestep()

    def get_number_elements(self):
        return self.grid.get_number_elements()

    def get_reward_signal_class(self):
        return self.get_reward_signal_class

    def get_substations_ids_prods(self):
        return np.asarray(list(map(lambda x: int(float(x)),
                                   list(map(lambda v: str(v).replace(ARTIFICIAL_NODE_STARTING_STRING, ''),
                                            self.grid.mpc['gen'][:, 0]))))).astype(int)

    def get_substations_ids_loads(self):
        return np.asarray(list(map(lambda x: int(float(x)),
                                   list(map(lambda v: str(v).replace(ARTIFICIAL_NODE_STARTING_STRING, ''),
                                            self.grid.mpc['bus'][self.grid.are_loads, 0]))))).astype(int)

    def get_substations_ids_lines_or(self):
        return np.asarray(list(map(lambda x: int(float(x)),
                                   list(map(lambda v: str(v).replace(ARTIFICIAL_NODE_STARTING_STRING, ''),
                                            self.grid.mpc['branch'][:, 0]))))).astype(int)

    def get_substations_ids_lines_ex(self):
        return np.asarray(list(map(lambda x: int(float(x)),
                                   list(map(lambda v: str(v).replace(ARTIFICIAL_NODE_STARTING_STRING, ''),
                                            self.grid.mpc['branch'][:, 1]))))).astype(int)

    def get_current_timestep_id(self):
        """ Retrieves the current index of scenario; this index might differs from a natural counter (some id may be
        missing within the chronic).

        :return: an integer of the id of the current scenario loaded
        """
        return self.current_timestep_id

    def _get_planned_maintenance(self):
        timestep_id = self.current_timestep_id
        return self.__chronic.get_planned_maintenance(timestep_id, self.n_timesteps_horizon_maintenance)

    def get_initial_topology(self):
        """ Retrieves the initial topology of the grid (when it was initially loaded). This is notably used to
        reinitialize the grid after a game over.

        :return: an instance of pypownet.grid.Topology or a list of integers
        """
        return self.initial_topology.get_unzipped()

    def get_substations_ids(self):
        return self.substations_ids

    def get_next_chronic(self):
        self.logger.info('Loading next chronic...')
        chronic = Chronic(self.__chronic_looper.get_next_chronic_folder())
        self.logger.info('  loaded chronic %s' % chronic.name)
        self.current_timestep_id = 0
        return chronic

    def load_entries_from_timestep_id(self, timestep_id, is_simulation=False, silence=False):
        # Retrieve the Scenario object associated to the desired id
        timestep_entries = self.__chronic.get_timestep_entries(timestep_id)
        self.current_timestep_entries = timestep_entries

        # Loads the next timestep injections: PQ and PV and gen status
        if not is_simulation:
            self.grid.load_timestep_injections(timestep_entries)
        else:
            self.grid.load_timestep_injections(timestep_entries,
                                               prods_p=self.current_timestep_entries.get_planned_prods_p(),
                                               prods_v=self.current_timestep_entries.get_planned_prods_v(),
                                               loads_p=self.current_timestep_entries.get_planned_loads_p(),
                                               loads_q=self.current_timestep_entries.get_planned_loads_q(), )

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
        if n_maintained > 0 and (not silence and not is_simulation):
            self.logger.info(
                '  MAINTENANCE: switching off line%s %s for %s%s timestep%s' %
                ('s' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '#%.0f' % i, self.grid.ids_lines[mask_affected_lines]))),
                 'resp. ' if n_maintained > 1 else '',
                 ', '.join(list(map(lambda i: '%.0f' % i, timestep_maintenance[mask_affected_lines]))),
                 's' if n_maintained > 1 or np.any(timestep_maintenance[mask_affected_lines] > 1) else '')
            )

        # No hazards for simulations
        if not is_simulation:
            # Integration of timestep hazards: disco freshly broken lines (just now)
            timestep_hazards = timestep_entries.get_hazards()
            mask_affected_lines = timestep_hazards > 0
            self.grid.mpc['branch'][mask_affected_lines, 10] = 0
            assert not np.any(timestep_hazards[mask_affected_lines] == 0), 'Line hazard cant last for 0 timestep'

            # For cases when multiple events (mtn, hazards) can overlap, put the max time to wait as new value between
            # new time to wait and previous one
            self.timesteps_before_lines_reconnectable[mask_affected_lines] = np.max(
                np.vstack((self.timesteps_before_lines_reconnectable[mask_affected_lines],
                           timestep_hazards[mask_affected_lines],)), axis=0)

            # Logs data about lines just broke from hazards
            n_maintained = sum(mask_affected_lines)
            if n_maintained > 0 and not silence:
                self.logger.info(
                    '  HAZARD: line%s %s broke; reparations will take %s%s timestep%s' %
                    ('s' if n_maintained > 1 else '',
                     ', '.join(list(map(lambda i: '#%.0f' % i, self.grid.ids_lines[mask_affected_lines]))),
                     'resp. ' if n_maintained > 1 else '',
                     ', '.join(list(map(lambda i: '%.0f' % i, timestep_hazards[mask_affected_lines]))),
                     's' if n_maintained > 1 or np.any(timestep_hazards[mask_affected_lines] > 1) else '')
                )

        self.previous_timestep = self.current_timestep_id
        self.current_timestep_id = timestep_id  # Update id of current timestep after whole entries are in place
        self.previous_date = copy.deepcopy(self.current_date)
        self.current_date = timestep_entries.get_datetime()

    def load_entries_from_next_timestep(self, is_simulation=False):
        """ Loads the next timestep injections (set of injections, maintenance, hazard etc for the next timestep id).

        :return: :raise ValueError: raised in the case where they are no more scenarios available
        """
        timesteps_ids = self.__chronic.get_timestep_ids()
        # If there are no more timestep to be played, loads next chronic
        if self.current_timestep_id == timesteps_ids[-1]:
            self.__chronic = self.get_next_chronic()

        # If no timestep injections has been loaded so far, loads the first one
        if self.current_timestep_id is None:
            next_timestep_id = timesteps_ids[0]
        else:  # Otherwise loads the next one in the list of timesteps injections
            next_timestep_id = timesteps_ids[timesteps_ids.index(self.current_timestep_id) + 1]

        # If the method is not simulate, decrement the actual timesteps to wait for the crashed lines (real step call)
        if not is_simulation:
            self.timesteps_before_lines_reconnectable[self.timesteps_before_lines_reconnectable > 0] -= 1

        self.load_entries_from_timestep_id(next_timestep_id, is_simulation)

    def _compute_loadflow_cascading(self):
        depth = 0  # Count cascading depth
        is_done = False
        over_thlim_lines = np.full(self.grid.n_lines, False)
        # Will loop undefinitely until an exception is raised (~outage) or the grid has no overflowed line
        while not is_done:
            is_done = True  # Reset is_done: if a line is broken bc of cascading failure, then is_done=False

            # Compute loadflow of current grid
            try:
                self.grid.compute_loadflow(fname_end='_cascading%d' % depth)
            except pypownet.grid.DivergingLoadflowException as e:
                e.text += ': cascading emulation of depth %d has diverged' % depth
                if self.renderer is not None:
                    self.render(None, game_over=True, cascading_frame_id=depth, date=self.previous_date,
                                timestep_id=self.previous_timestep)
                raise DivergingLoadflowException(e.last_observation, e.text)

            current_flows_a = self.grid.extract_flows_a()
            thermal_limits = self.grid.get_thermal_limits()

            over_thlim_lines = current_flows_a > thermal_limits  # Mask of overflowed lines
            # Computes the number of overflows: if 0, exit now for software speed
            if np.sum(over_thlim_lines) == 0:
                break

            # Checks for lines over hard nominal thermal limit
            over_hard_thlim_lines = current_flows_a > self.hard_overflow_coefficient * thermal_limits
            if np.any(over_hard_thlim_lines):
                # Break lines over their hard thermal limits: set status to 0 and increment timesteps before reconn.
                self.grid.get_lines_status()[over_hard_thlim_lines] = 0
                self.timesteps_before_lines_reconnectable[
                    over_hard_thlim_lines] = self.n_timesteps_hard_overflow_is_broken
                is_done = False
            # Those lines have been treated so discard them for further depth process
            over_thlim_lines[over_hard_thlim_lines] = False

            # Checks for soft-overflowed lines among remaining lines
            if np.any(over_thlim_lines):
                time_limit = self.n_timesteps_consecutive_soft_overflow_breaks
                number_timesteps_over_thlim_lines = self.n_timesteps_soft_overflowed_lines
                # Computes the soft-broken lines: they are overflowed and has been so for more than time_limit
                soft_broken_lines = np.logical_and(over_thlim_lines, number_timesteps_over_thlim_lines >= time_limit)
                if np.any(soft_broken_lines):
                    # Soft break lines overflowed for more timesteps than the limit
                    self.grid.get_lines_status()[soft_broken_lines] = 0
                    self.timesteps_before_lines_reconnectable[
                        soft_broken_lines] = self.n_timesteps_soft_overflow_is_broken
                    is_done = False
                    # Do not consider those lines anymore
                    over_thlim_lines[soft_broken_lines] = False

            depth += 1
            if self.renderer is not None:
                self.render(None, cascading_frame_id=depth, date=self.previous_date, timestep_id=self.previous_timestep)

        # At the end of the cascading failure, decrement timesteps waited by overflowed lines
        self.n_timesteps_soft_overflowed_lines[over_thlim_lines] += 1
        self.n_timesteps_soft_overflowed_lines[~over_thlim_lines] = 0

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
        prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes = grid_topology.get_unzipped()
        # Retrieve lines status service of current grid
        lines_service = self.grid.get_lines_status()

        action_lines_service = action.get_lines_status_subaction()
        action_prods_nodes = action.get_prods_switches_subaction()
        action_loads_nodes = action.get_loads_switches_subaction()
        action_lines_or_nodes = action.get_lines_or_switches_subaction()
        action_lines_ex_nodes = action.get_lines_ex_switches_subaction()

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

        # import pypower.api
        # import os
        # pypower.api.savecase(os.path.join('tmp', self.grid.filename[:-2]+'.py'), self.grid.mpc)

    def reset(self):
        """ Resets the game: put the grid topology to the initial one. Besides, if restart is True, then the game will
        load the first set of injections (i)_{t0}, otherwise the next set of injections of the chronics (i)_{t+1}
        """
        self.reset_grid()
        self.epoch += 1
        # Loads next chronics if game is hardcore mode
        if self.game_over_mode == 'hard':  # If restart, put current id to None so that load_next will load first timestep
            self.current_timestep_id = None
            self.timestep = 1
            self.__chronic = self.get_next_chronic()

        try:
            self.load_entries_from_next_timestep()
            self._compute_loadflow_cascading()
        # If after reset there is a diverging loadflow, then recall reset w/o penalty (not the player's fault)
        except pypownet.grid.DivergingLoadflowException:
            self.reset()

    def reset_grid(self):
        """ Reinitialized the grid by applying the initial topology to the current state (topology).
        """
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))

        self.grid.apply_topology(self.initial_topology)
        self.grid.mpc['gen'][:, 7] = 1  # Put all prods status to 1 (ON)
        self.grid.set_lines_status(self.initial_lines_service)  # Reset lines status
        #self.grid.discard_flows()  # Discard flows: not mandatory

        # Reset voltage magnitude and angles: they change when using AC mode
        self.grid.set_voltage_angles(self.initial_voltage_angles)
        self.grid.set_voltage_magnitudes(self.initial_voltage_magnitudes)

        self.grid.mpc = {k: v for k, v in self.grid.mpc.items() if k in ['bus', 'gen', 'branch', 'baseMVA', 'version']}

        #self.grid.set_flows_to_0()

    def step(self, action, _is_simulation=False):
        if _is_simulation:
            assert self.grid.dc_loadflow, "Cheating detected"

        # Apply action, or raises eception if some broken lines are attempted to be switched on
        try:
            self.last_action = action  # tmp
            self.apply_action(action)
            if self.renderer is not None:
                self.render(None, cascading_frame_id=-1)
        except pypownet.environment.IllegalActionException as e:
            e.text += ' Ignoring action switches of broken lines.'
            # If broken lines are attempted to be switched on, put the switches to 0
            illegal_lines_reconnections = e.illegal_lines_reconnections
            action.lines_status_subaction[illegal_lines_reconnections] = 0
            assert np.sum(action.get_lines_status_subaction()[illegal_lines_reconnections]) == 0

            # Resubmit step with modified valid action and return either exception of new step, or this exception
            obs, correct_step, done = self.step(action, _is_simulation=_is_simulation)
            return obs, correct_step if correct_step else e, done  # Return done and not False because step might
            # diverge

        try:
            # Load next timestep entries, compute one loadflow, then potentially cascading failure
            self.load_entries_from_next_timestep(is_simulation=_is_simulation)
            self._compute_loadflow_cascading()
        except pypownet.grid.DivergingLoadflowException as e:
            return None, DivergingLoadflowException(e.last_observation, e.text), True

        are_isolated_loads, are_isolated_prods, _ = pypownet.grid.Grid._count_isolated_loads(self.grid.mpc,
                                                                                             self.grid.are_loads)
        self.n_loads_cut, self.n_prods_cut = sum(are_isolated_loads), sum(are_isolated_prods)

        # Check whether max number of productions and load cut are not reached
        if np.sum(are_isolated_loads) > self.max_number_loads_game_over:
            observation = None
            flag = TooManyConsumptionsCut('There are %d isolated loads; at most %d tolerated' % (
                np.sum(are_isolated_loads), self.max_number_loads_game_over))
            done = True
            if self.renderer is not None:
                self.render(None, game_over=True, cascading_frame_id=None)
            return observation, flag, done
        if np.sum(are_isolated_prods) > self.max_number_prods_game_over:
            observation = None
            flag = TooManyProductionsCut('There are %d isolated productions; at most %d tolerated' % (
                np.sum(are_isolated_prods), self.max_number_prods_game_over))
            done = True
            if self.renderer is not None:
                self.render(None, game_over=True, cascading_frame_id=None)
            return observation, flag, done

        return self.export_observation(), None, False

    def simulate(self, action):
        # Copy variables of a step: timestep id, mpc (~grid), topology (stand-alone in self), and lists of overflows
        before_timestep_id = self.current_timestep_id
        before_mpc = copy.deepcopy(self.grid.mpc)
        before_topology = copy.deepcopy(self.grid.get_topology())
        before_n_timesteps_overflowed_lines = self.n_timesteps_soft_overflowed_lines
        before_timesteps_before_lines_reconnectable = self.timesteps_before_lines_reconnectable

        # Save grid AC or DC normal mode, and force DC mode for simulate
        before_dc = self.grid.dc_loadflow
        self.grid.dc_loadflow = True

        def reload_minus_1_timestep():
            self.grid.mpc = before_mpc  # Change grid mpc before apply topo
            self.grid.apply_topology(before_topology)  # Change topo before loading entries (reflects what happened)
            self.load_entries_from_timestep_id(before_timestep_id, silence=True)
            self.n_timesteps_soft_overflowed_lines = before_n_timesteps_overflowed_lines
            self.timesteps_before_lines_reconnectable = before_timesteps_before_lines_reconnectable
            return

        # Step the action
        try:
            simulated_obs, flag, done = self.step(action, _is_simulation=True)
        except pypownet.grid.DivergingLoadflowException as e:
            # Reset previous step
            reload_minus_1_timestep()
            # Put back on previous mode (should be AC)
            self.grid.dc_loadflow = before_dc
            raise DivergingLoadflowException(e.last_observation, e.text)

        # Reset previous timestep conditions (cancel previous step)
        reload_minus_1_timestep()

        # Put back on previous mode (should be AC)
        self.grid.dc_loadflow = before_dc

        return simulated_obs, flag, done

    def export_observation(self):
        """ Retrieves an observation of the current state of the grid.

        :return: an instance of class pypownet.env.Observation
        """
        observation = self.grid.export_to_observation()
        # Fill additional parameters: starts with substations ids of all elements
        observation.timesteps_before_lines_reconnectable = self.timesteps_before_lines_reconnectable
        observation.timesteps_before_planned_maintenance = self._get_planned_maintenance()

        current_timestep_entries = self.current_timestep_entries
        observation.planned_active_loads = current_timestep_entries.get_planned_loads_p()
        observation.planned_reactive_loads = current_timestep_entries.get_planned_loads_q()
        observation.planned_active_productions = current_timestep_entries.get_planned_prods_p()
        observation.planned_voltage_productions = self.grid.normalize_prods_voltages(
            current_timestep_entries.get_planned_prods_v())

        # Initial topology
        initial_topology = self.get_initial_topology()
        observation.initial_productions_nodes = initial_topology[0]
        observation.initial_loads_nodes = initial_topology[1]
        observation.initial_lines_or_nodes = initial_topology[2]
        observation.initial_lines_ex_nodes = initial_topology[3]

        observation.datetime = self.current_date

        return observation

    def render(self, rewards, game_over=False, cascading_frame_id=None, date=None, timestep_id=None):
        """ Initializes the renderer if not already done, then compute the necessary values to be carried to the
        renderer class (e.g. sum of consumptions).

        :param rewards: list of subrewards of the last timestep (used to plot reward per timestep)
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

            timestep_duration_seconds = self.__chronic.get_timestep_duration()

            from pypownet.renderer import Renderer

            return Renderer(len(self.grid.number_elements_per_substations), idx_or, idx_ex, are_prods, are_loads,
                            timestep_duration_seconds)

        try:
            import pygame
        except ImportError as e:
            raise ImportError(
                "{}. (HINT: install pygame using `pip install pygame` or refer to this package README)".format(e))

        # if close:
        #     pygame.quit()

        if self.renderer is None:
            self.renderer = initialize_renderer()

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
        if self.last_action is not None and cascading_frame_id is not None:
            last_action = self.grid.get_topology().mapping_permutation(
                self.last_action.as_array()[:-self.last_action.__len__(False)[-1]])
            n_elements_substations = self.grid.number_elements_per_substations
            offset = 0
            for i, (substation_id, n_elements) in enumerate(zip(substations_ids, n_elements_substations)):
                has_been_changed[i] = np.any([l != 0 for l in last_action[offset:offset + n_elements]])
                offset += n_elements

        are_isolated_loads, are_isolated_prods, _ = self.grid._count_isolated_loads(self.grid.mpc, self.grid.are_loads)
        number_unavailable_lines = sum(self.timesteps_before_lines_reconnectable > 0)

        initial_topo = self.initial_topology
        current_topo = self.grid.get_topology()
        distance_ref_grid = sum(np.asarray(initial_topo.get_zipped()) != np.asarray(current_topo.get_zipped()))

        max_number_isolated_loads = self.__parameters.get_max_number_loads_game_over()
        max_number_isolated_prods = self.__parameters.get_max_number_prods_game_over()

        self.renderer.render(lines_capacity_usage, lines_por_values, lines_service_status,
                             self.epoch, self.timestep, self.current_timestep_id if not timestep_id else timestep_id,
                             prods=prods_values, loads=loads_values, last_timestep_rewards=rewards,
                             date=self.current_date if date is None else date, are_substations_changed=has_been_changed,
                             number_loads_cut=sum(are_isolated_loads),
                             number_prods_cut=sum(are_isolated_prods),
                             number_nodes_splitting=sum(self.last_action.get_node_splitting_subaction())
                                        if self.last_action is not None else 0,
                             number_lines_switches=sum(self.last_action.get_lines_status_subaction())
                                        if self.last_action is not None else 0,
                             distance_initial_grid=distance_ref_grid,
                             number_off_lines=sum(self.grid.get_lines_status() == 0),
                             number_unavailable_lines=number_unavailable_lines,
                             max_number_isolated_loads=max_number_isolated_loads,
                             max_number_isolated_prods=max_number_isolated_prods,
                             game_over=game_over, cascading_frame_id=cascading_frame_id)

        if self.latency:
            sleep(self.latency)

    def parameters_environment_tostring(self):
        return self.__parameters.__str__()
