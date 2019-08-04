__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import logging

from time import sleep
import copy
import numpy as np
import pypownet.grid
from pypownet.chronic import Chronic, ChronicLooper
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
from pypownet.parameters import Parameters


# Exception to be risen when no more scenarios are available to be played (i.e. every scenario has been played)
class NoMoreScenarios(Exception):
    pass


class IllegalActionException(Exception):
    def __init__(self, text, has_too_much_activations, illegal_lines_reconnections, illegal_unavailable_lines_switches,
                 illegal_oncoolown_substations_switches, *args):
        super(IllegalActionException, self).__init__(*args)
        self.text = text
        self.has_too_much_activations = has_too_much_activations
        # size resp. n_lines, n_lines, n_substations; can be None if no illegal corresponding actions
        self.illegal_broken_lines_reconnections = illegal_lines_reconnections
        self.illegal_oncooldown_lines_switches = illegal_unavailable_lines_switches
        self.illegal_oncoolown_substations_switches = illegal_oncoolown_substations_switches

    def get_has_too_much_activations(self):
        return self.has_too_much_activations

    def get_illegal_broken_lines_reconnections(self):
        return self.illegal_broken_lines_reconnections

    def get_illegal_oncoolown_lines_switches(self):
        return self.illegal_oncooldown_lines_switches

    def get_illegal_oncoolown_substations_switches(self):
        return self.illegal_oncoolown_substations_switches

    @property
    def is_empty(self):
        return self.has_too_much_activations is False and self.illegal_broken_lines_reconnections is None \
               and self.illegal_oncooldown_lines_switches is None \
               and self.illegal_oncoolown_substations_switches is None


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


class ListExceptions(Exception):
    def __init__(self, exceptions):
        super(ListExceptions, self).__init__(exceptions)
        self.exceptions = exceptions


class Action(object):
    def __init__(self, prods_switches_subaction, loads_switches_subaction,
                 lines_or_switches_subaction, lines_ex_switches_subaction, lines_status_subaction,
                 substations_ids, prods_subs_ids, loads_subs_ids, lines_or_subs_id, lines_ex_subs_id,
                 elementtype):
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

        self.substations_ids = substations_ids
        self.prods_subs_ids = prods_subs_ids
        self.loads_subs_ids = loads_subs_ids
        self.lines_or_subs_id = lines_or_subs_id
        self.lines_ex_subs_id = lines_ex_subs_id
        self.elementtype = elementtype  # should be class enumerating the types of elements

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

    def set_node_splitting_subaction(self, new_node_splitting_subaction):
        assert len(new_node_splitting_subaction) == len(self.get_node_splitting_subaction())
        offset = 0
        # prods -> loads -> lines or -> lines ex
        self.prods_switches_subaction = new_node_splitting_subaction[offset:offset + self._prods_switches_length]
        offset += self._prods_switches_length
        self.loads_switches_subaction = new_node_splitting_subaction[offset:offset + self._loads_switches_length]
        offset += self._loads_switches_length
        self.lines_or_switches_subaction = new_node_splitting_subaction[offset:offset + self._lines_or_switches_length]
        offset += self._lines_or_switches_length
        self.lines_ex_switches_subaction = new_node_splitting_subaction[offset:]
        return

    def get_lines_status_subaction(self):
        return self.lines_status_subaction

    def get_substation_switches(self, substation_id, concatenated_output=True):
        assert substation_id in self.substations_ids, 'Substation with id %d does not exist' % substation_id

        # Save the type of each elements in the returned switches list
        elements_type = []

        # Retrieve switches associated with resp. production (max 1 per substation), consumptions (max 1 per substation),
        # origins of lines, extremities of lines; also saves each type inserted within the switches-values list
        prod_switches = self.prods_switches_subaction[
            np.where(self.prods_subs_ids == substation_id)] if substation_id in self.prods_subs_ids else []
        elements_type.extend([self.elementtype.PRODUCTION] * len(prod_switches))
        load_switches = self.loads_switches_subaction[
            np.where(self.loads_subs_ids == substation_id)] if substation_id in self.loads_subs_ids else []
        elements_type.extend([self.elementtype.CONSUMPTION] * len(load_switches))
        lines_origins_switches = self.lines_or_switches_subaction[
            np.where(self.lines_or_subs_id == substation_id)] if substation_id in self.lines_or_subs_id else []
        elements_type.extend([self.elementtype.ORIGIN_POWER_LINE] * len(lines_origins_switches))
        lines_extremities_switches = self.lines_ex_switches_subaction[
            np.where(self.lines_ex_subs_id == substation_id)] if substation_id in self.lines_ex_subs_id else []
        elements_type.extend([self.elementtype.EXTREMITY_POWER_LINE] * len(lines_extremities_switches))

        assert len(elements_type) == len(prod_switches) + len(load_switches) + len(lines_origins_switches) + len(
            lines_extremities_switches), "Mistmatch lengths for elements type and switches-value list; should not happen"

        return np.concatenate((prod_switches, load_switches, lines_origins_switches,
                               lines_extremities_switches)) if concatenated_output else \
                   (prod_switches, load_switches, lines_origins_switches, lines_extremities_switches), \
               np.asarray(elements_type)

    def set_substation_switches(self, substation_id, new_values):
        new_values = np.asarray(new_values)

        _, elements_type = self.get_substation_switches(substation_id, concatenated_output=False)
        expected_configuration_size = len(elements_type)
        assert expected_configuration_size == len(new_values), 'Expected new_values of size %d for' \
                                                               ' substation %d, got size %d' % (
                                                                   expected_configuration_size, substation_id,
                                                                   len(new_values))

        self.prods_switches_subaction[self.prods_subs_ids == substation_id] = new_values[
            elements_type == self.elementtype.PRODUCTION]
        self.loads_switches_subaction[self.loads_subs_ids == substation_id] = new_values[
            elements_type == self.elementtype.CONSUMPTION]
        self.lines_or_switches_subaction[self.lines_or_subs_id == substation_id] = new_values[
            elements_type == self.elementtype.ORIGIN_POWER_LINE]
        self.lines_ex_switches_subaction[self.lines_ex_subs_id == substation_id] = new_values[
            elements_type == self.elementtype.EXTREMITY_POWER_LINE]

        return self

    def set_as_do_nothing(self):
        # fill self values with values of do nothing action ie all 0
        self.prods_switches_subaction = np.zeros(len(self.prods_switches_subaction)).astype(int)
        self.loads_switches_subaction = np.zeros(len(self.loads_switches_subaction)).astype(int)
        self.lines_or_switches_subaction = np.zeros(len(self.lines_or_switches_subaction)).astype(int)
        self.lines_ex_switches_subaction = np.zeros(len(self.lines_ex_switches_subaction)).astype(int)
        self.lines_status_subaction = np.zeros(len(self.lines_status_subaction)).astype(int)
        return self

    def as_array(self):
        return np.concatenate((self.get_node_splitting_subaction(), self.get_lines_status_subaction(),))

    def __str__(self):
        return self.as_array().__str__()

    def __len__(self, do_sum=True):
        length_aslist = (self._prods_switches_length, self._loads_switches_length, self._lines_or_switches_length,
                         self._lines_ex_switches_length, self._lines_status_length)
        return sum(length_aslist) if do_sum else length_aslist

    def __setitem__(self, item, value):
        item %= len(self)

        if item < self._prods_switches_length:
            return self.prods_switches_subaction.__setitem__(item, value)
        item -= self._prods_switches_length

        if item < self._loads_switches_length:
            return self.loads_switches_subaction.__setitem__(item, value)
        item -= self._loads_switches_length

        if item < self._lines_or_switches_length:
            return self.lines_or_switches_subaction.__setitem__(item, value)
        item -= self._lines_or_switches_length

        if item < self._lines_ex_switches_length:
            return self.lines_ex_switches_subaction.__setitem__(item, value)
        item -= self._lines_ex_switches_length

        return self.lines_status_subaction.__setitem__(item, value)

    def __getitem__(self, item):
        item %= len(self)

        if item < self._prods_switches_length:
            return self.prods_switches_subaction.__getitem__(item)
        item -= self._prods_switches_length

        if item < self._loads_switches_length:
            return self.loads_switches_subaction.__getitem__(item)
        item -= self._loads_switches_length

        if item < self._lines_or_switches_length:
            return self.lines_or_switches_subaction.__getitem__(item)
        item -= self._lines_or_switches_length

        if item < self._lines_ex_switches_length:
            return self.lines_ex_switches_subaction.__getitem__(item)
        item -= self._lines_ex_switches_length

        return self.lines_status_subaction.__getitem__(item)


class Game(object):
    def __init__(self, parameters_folder, game_level, chronic_looping_mode, chronic_starting_id,
                 game_over_mode, renderer_frame_latency, without_overflow_cutoff):
        """ Initializes an instance of the game. This class is sufficient to play the game both as human and as AI.
        """
        self.logger = logging.getLogger('pypownet.' + __name__)

        # Read parameters
        # backend
        self.__parameters = Parameters(parameters_folder, game_level)
        loadflow_backend = self.__parameters.get_loadflow_backend()
        self.is_mode_dc = self.__parameters.is_dc_mode()
        # overflows
        self.hard_overflow_coefficient = self.__parameters.get_hard_overflow_coefficient()
        if without_overflow_cutoff:
            self.hard_overflow_coefficient = 1e9
        self.n_timesteps_hard_overflow_is_broken = self.__parameters.get_n_timesteps_hard_overflow_is_broken()
        self.n_timesteps_soft_overflow_is_broken = self.__parameters.get_n_timesteps_soft_overflow_is_broken()
        self.n_timesteps_consecutive_soft_overflow_breaks = \
            self.__parameters.get_n_timesteps_consecutive_soft_overflow_breaks()
        if without_overflow_cutoff:
            self.n_timesteps_consecutive_soft_overflow_breaks = 1e12
        # maintenance
        self.n_timesteps_horizon_maintenance = self.__parameters.get_n_timesteps_horizon_maintenance()
        # game over
        self.max_number_prods_game_over = self.__parameters.get_max_number_prods_game_over()
        self.max_number_loads_game_over = self.__parameters.get_max_number_loads_game_over()
        # illegal action
        self.n_timesteps_actionned_line_reactionable = self.__parameters.get_n_timesteps_actionned_line_reactionable()
        self.n_timesteps_actionned_node_reactionable = self.__parameters.get_n_timesteps_actionned_node_reactionable()
        self.n_timesteps_pending_line_reactionable_when_overflowed = \
            self.__parameters.get_n_timesteps_pending_line_reactionable_when_overflowed()
        self.n_timesteps_pending_node_reactionable_when_overflowed = \
            self.__parameters.get_n_timesteps_pending_node_reactionable_when_overflowed()
        self.max_number_actionned_substations = self.__parameters.get_max_number_actionned_substations()
        self.max_number_actionned_lines = self.__parameters.get_max_number_actionned_lines()
        self.max_number_actionned_total = self.__parameters.get_max_number_actionned_total()

        # Seek and load chronic
        self.__chronic_looper = ChronicLooper(chronics_folder=self.__parameters.get_chronics_path(),
                                              game_level=game_level, start_id=chronic_starting_id,
                                              looping_mode=chronic_looping_mode)
        self.__chronic = self.get_next_chronic()

        self.game_over_mode = game_over_mode

        # Seek and load starting reference grid
        self.grid = pypownet.grid.Grid(loadflow_backend=loadflow_backend,
                                       src_filename=self.__parameters.get_reference_grid_path(loadflow_backend),
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

        # Instantiate the counter of timesteps before lines can be reconnected (one value per line) or activated
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))
        self.timesteps_before_lines_reactionable = np.zeros((self.grid.n_lines,))
        self.timesteps_before_nodes_reactionable = np.zeros((len(self.substations_ids),))

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
        n_prods, n_loads, n_lines = self.grid.get_number_elements()
        return n_prods, n_loads, n_lines, len(self.get_substations_ids())

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

    def get_current_chronic_name(self):
        return self.__chronic_looper.get_current_chronic_name()

    def load_entries_from_timestep_id(self, timestep_id, is_simulation=False, silence=False):
        # Retrieve the Scenario object associated to the desired id
        timestep_entries = self.__chronic.get_timestep_entries(timestep_id)

        # Loads the next timestep injections: PQ and PV and gen status
        if not is_simulation:
            self.current_timestep_entries = timestep_entries  # do not apply for simulation or would reveal planned
                                                              # injections/maintenance in the outputted observation
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
        # If there are no more timestep to be played, loads next chronic except if this is called during a simulation
        if self.current_timestep_id == timesteps_ids[-1] and not is_simulation:
            self.__chronic = self.get_next_chronic()

        # If no timestep injections has been loaded so far, loads the first one
        if self.current_timestep_id is None:
            next_timestep_id = timesteps_ids[0]
        # Otherwise loads the next one in the list of timesteps injections; note the use of min to generalize to the
        # case of is_simulation=True which will load the last timstep indefinitely until a real timestep is played
        else:
            next_timestep_id = timesteps_ids[
                min(timesteps_ids.index(self.current_timestep_id) + 1, len(timesteps_ids) - 1)]

        # If the method is not simulate, decrement the actual timesteps to wait for the crashed lines (real step call)
        if not is_simulation:
            self.timesteps_before_lines_reconnectable[self.timesteps_before_lines_reconnectable > 0] -= 1
            self.timesteps_before_lines_reactionable[self.timesteps_before_lines_reactionable > 0] -= 1
            self.timesteps_before_nodes_reactionable[self.timesteps_before_nodes_reactionable > 0] -= 1

        self.load_entries_from_timestep_id(next_timestep_id, is_simulation)

    def _compute_loadflow_cascading(self, is_simulation=False):
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
                if self.renderer is not None and not is_simulation:
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

                # Log some info
                n_cut_off = sum(over_hard_thlim_lines)
                if not is_simulation:
                    self.logger.info('  AUTOMATIC HARD OVERFLOW CUT-OFF: switching off line%s %s for %s timestep%s due to '
                                     'hard overflow (%s%s of capacity)' %
                                     ('s' if n_cut_off > 1 else '',
                                      ', '.join(
                                          list(map(lambda i: '#%.0f' % i, self.grid.ids_lines[over_hard_thlim_lines]))),
                                      str(self.n_timesteps_hard_overflow_is_broken),
                                      's' if self.n_timesteps_hard_overflow_is_broken > 1 else '',
                                      'resp. ' if n_cut_off > 1 else '',
                                      ', '.join(list(map(lambda i: '%.0f%%' % i,
                                                         100. * current_flows_a[over_hard_thlim_lines] / thermal_limits[
                                                             over_hard_thlim_lines])))))
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

                    # Log some info
                    n_cut_off = sum(soft_broken_lines)
                    if not is_simulation:
                        self.logger.info('  AUTOMATIC SOFT OVERFLOW CUT-OFF: switching off line%s %s for %s timestep%s due '
                                         'to soft overflow' %
                                         ('s' if n_cut_off > 1 else '',
                                          ', '.join(
                                              list(map(lambda i: '#%.0f' % i,
                                                       self.grid.ids_lines[soft_broken_lines]))),
                                          str(self.n_timesteps_soft_overflow_is_broken),
                                          's' if self.n_timesteps_hard_overflow_is_broken > 1 else ''))

                    # Do not consider those lines anymore
                    over_thlim_lines[soft_broken_lines] = False

            depth += 1
            if self.renderer is not None and not is_simulation:
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

        try:
            to_be_raised_exception = self._verify_illegal_action(action)
        except (IllegalActionException, ValueError) as e:
            raise e

        # If illegal moves has been caught, raise exception
        if not to_be_raised_exception.is_empty:
            raise to_be_raised_exception

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

        # Put activated lines as unactivable for predetermined timestep
        has_lines_been_changed = action_lines_service == 1
        self.timesteps_before_lines_reactionable[has_lines_been_changed] = self.n_timesteps_actionned_line_reactionable
        # Compute and put activated nodes as unactivable for predetermined timestep
        has_nodes_been_changed = self.get_changed_substations(action)
        self.timesteps_before_nodes_reactionable[has_nodes_been_changed] = self.n_timesteps_actionned_node_reactionable

    def _verify_illegal_action(self, action):
        # Initialize exception container that gather illegal action moves to account for all eventual action errors
        # before returning
        to_be_raised_exception = IllegalActionException('', False, None, None, None)

        # If there is no action, then no need to apply anything on the grid
        if action is None:
            raise ValueError('Cannot play None action')

        action_lines_service = action.get_lines_status_subaction()

        # Compute the elements to be switched by the action (ie where >= 1 value is one for substation, or line is 1)
        to_be_switched_substations = self.get_changed_substations(action)
        to_be_switched_lines = action_lines_service == 1

        # First, check the number of activated elements (substations, lines) and compare with max tolerated numbers
        n_switched_substations = sum(to_be_switched_substations)
        n_switched_lines = sum(to_be_switched_lines)
        n_switched_total = n_switched_substations + n_switched_lines
        if n_switched_substations > self.max_number_actionned_substations \
                or n_switched_lines > self.max_number_actionned_lines \
                or n_switched_total > self.max_number_actionned_total:
            to_be_raised_exception.has_too_much_activations = True
            to_be_raised_exception.text += 'Action has too much activations simultaneously: ' \
                                           '{}/{} activated substations, {}/{} switched lines and ' \
                                           '{}/{} total switched elements (substations and lines).' \
                .format(n_switched_substations, self.max_number_actionned_substations, n_switched_lines,
                        self.max_number_actionned_lines, n_switched_total, self.max_number_actionned_total)
            # Raise error now because the following checks do not make sense since the action is virtually do-nothing at
            # this point
            raise to_be_raised_exception

        # Verify that the player is not intended to reconnect not reconnectable lines (broken or being maintained);
        # here, we check for lines switches, because if a line is broken, then its status is already to 0, such that a
        # switch will switch on the power line
        broken_lines = self.timesteps_before_lines_reconnectable > 0  # Mask of broken lines
        illegal_lines_reconnections = np.logical_and(to_be_switched_lines, broken_lines)
        # Raises an exception if there is some attempt to reconnect broken lines; Game.step should manage what to do in
        # such a case
        if np.any(illegal_lines_reconnections):
            timesteps_to_wait = self.timesteps_before_lines_reconnectable[illegal_lines_reconnections]
            assert not np.any(timesteps_to_wait <= 0), 'Should not happen'

            # Creates strings for log printing
            non_reconnectable_lines_as_str = ', '.join(
                list(map(str, np.arange(len(illegal_lines_reconnections))[illegal_lines_reconnections])))
            timesteps_to_wait_as_str = ', '.join(list(map(lambda x: str(int(x)), timesteps_to_wait)))

            number_invalid_reconnections = np.sum(illegal_lines_reconnections)
            if number_invalid_reconnections > 1:
                timesteps_to_wait_as_str = 'resp. ' + timesteps_to_wait_as_str

            # postponed exception raising for catching other action illegal moves below
            to_be_raised_exception.text += 'Trying to reconnect broken/on-maintenance line%s %s, must wait %s ' \
                                           'timesteps.' % ('s' if number_invalid_reconnections > 1 else '',
                                                           non_reconnectable_lines_as_str, timesteps_to_wait_as_str)
            to_be_raised_exception.illegal_broken_lines_reconnections = illegal_lines_reconnections

        # Verify that the player is not trying to switch lines or nodes topologies that are pending for reusage after
        # being used within the tolerated timeframe before another action can be operated on a line or a node
        ## lines
        unactionnable_lines = self.timesteps_before_lines_reactionable > 0
        illegal_activating_lines = np.logical_and(to_be_switched_lines, unactionnable_lines)
        if np.any(illegal_activating_lines):
            timesteps_to_wait = self.timesteps_before_lines_reactionable[illegal_activating_lines]
            assert not np.any(timesteps_to_wait <= 0), 'Should not happen'

            # Creates strings for log printing
            non_actionnable_lines_as_str = ', '.join(
                list(map(str, np.arange(len(illegal_activating_lines))[illegal_activating_lines])))
            timesteps_to_wait_as_str = ', '.join(list(map(lambda x: str(int(x)), timesteps_to_wait)))

            number_invalid_activations = np.sum(illegal_activating_lines)
            if number_invalid_activations > 1:
                timesteps_to_wait_as_str = 'resp. ' + timesteps_to_wait_as_str

            # postponed exception raising for catching other action illegal moves below
            to_be_raised_exception.text += 'Trying to action on-cooldown line%s %s, must wait resp. %s timesteps. ' % (
                's' if number_invalid_activations > 1 else '',
                non_actionnable_lines_as_str, timesteps_to_wait_as_str)
            to_be_raised_exception.illegal_oncooldown_lines_switches = illegal_activating_lines
        ## substations
        unactionnable_nodes = self.timesteps_before_nodes_reactionable > 0
        illegal_activating_nodes = np.logical_and(to_be_switched_substations, unactionnable_nodes)
        if np.any(illegal_activating_nodes):
            timesteps_to_wait = self.timesteps_before_nodes_reactionable[illegal_activating_nodes]
            assert not np.any(timesteps_to_wait <= 0), 'Should not happen'

            # Creates strings for log printing
            non_actionnable_nodes_as_str = ', '.join(
                list(map(str, np.arange(len(illegal_activating_nodes))[illegal_activating_nodes])))
            timesteps_to_wait_as_str = ', '.join(list(map(lambda x: str(int(x)), timesteps_to_wait)))

            number_invalid_activations = np.sum(illegal_activating_nodes)
            if number_invalid_activations > 1:
                timesteps_to_wait_as_str = 'resp. ' + timesteps_to_wait_as_str

            # postponed exception raising for catching other action illegal moves below
            to_be_raised_exception.text += 'Trying to action on-cooldown substation%s %s, must wait resp. %s ' \
                                           'timesteps.' % ('s' if number_invalid_activations > 1 else '',
                                                           non_actionnable_nodes_as_str, timesteps_to_wait_as_str)
            to_be_raised_exception.illegal_oncoolown_substations_switches = illegal_activating_nodes

        return to_be_raised_exception

    def is_action_valid(self, action):
        try:
            to_be_raised_exception = self._verify_illegal_action(action)
        except (IllegalActionException, ValueError):
            return False
        return to_be_raised_exception.is_empty

    def process_game_over(self):
        """ Handles game over behavior of the game: put the grid topology to the initial one
        + if restart is True, then the game will load the first set of injections (i)_{t0},
        otherwise the next set of injections of the chronics (i)_{t+1}.
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
            self.process_game_over()

    def reset_grid(self):
        """ Reinitialized the grid by applying the initial topology to the current state (topology).
        """
        self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))
        self.timesteps_before_lines_reactionable = np.zeros((self.grid.n_lines,))
        self.timesteps_before_nodes_reactionable = np.zeros((len(self.substations_ids),))

        self.grid.apply_topology(self.initial_topology)
        self.grid.mpc['gen'][:, 7] = 1  # Put all prods status to 1 (ON)
        self.grid.set_lines_status(self.initial_lines_service)  # Reset lines status

        # Reset voltage magnitude and angles: they change when using AC mode
        self.grid.set_voltage_angles(self.initial_voltage_angles)
        self.grid.set_voltage_magnitudes(self.initial_voltage_magnitudes)

        self.grid.mpc = {k: v for k, v in self.grid.mpc.items() if k in ['bus', 'gen', 'branch', 'baseMVA', 'version']}

    def step(self, action, _is_simulation=False):
        #if _is_simulation: #let's run everything in the same mode for now because we did not assess its impact
        #    assert self.grid.dc_loadflow, "Cheating detected"

        # Apply action, or raises exception if some broken lines are attempted to be switched on
        try:
            self.last_action = action  # tmp
            self.apply_action(action)
            if self.renderer is not None and not _is_simulation:
                self.render(None, cascading_frame_id=-1)
        except IllegalActionException as e:
            # First, check if the action does not overpass the max number of activated elements, and stop further
            # process since the other checks have not been done
            if e.get_has_too_much_activations():
                action = action.set_as_do_nothing()
                assert sum(action.as_array()) == 0
            else:
                # If broken/on-maintenance lines are attempted to be switched, put the switches to 0
                illegal_broken_lines_switches = e.get_illegal_broken_lines_reconnections()
                if illegal_broken_lines_switches is not None:
                    if sum(illegal_broken_lines_switches) > 0:
                        action.lines_status_subaction[illegal_broken_lines_switches] = 0  # cancel illegal moves
                        e.text += ' Ignoring action switches of broken/on-maintenance lines: %s.' % \
                                  ', '.join(list(map(str, np.where(illegal_broken_lines_switches)[0])))
                    assert np.sum(action.get_lines_status_subaction()[illegal_broken_lines_switches]) == 0

                # Similarly if on-cooldown lines are attempted to be switched, put the switches to 0
                illegal_oncooldown_lines_switches = e.get_illegal_oncoolown_lines_switches()
                if illegal_oncooldown_lines_switches is not None:
                    if sum(illegal_oncooldown_lines_switches) > 0:
                        action.lines_status_subaction[illegal_oncooldown_lines_switches] = 0  # cancel illegal moves
                        e.text += ' Ignoring action switches of on-cooldown lines: %s.' % \
                                  ', '.join(list(map(str, np.where(illegal_oncooldown_lines_switches)[0])))
                    assert np.sum(action.get_lines_status_subaction()[illegal_oncooldown_lines_switches]) == 0

                # Similarly for on-cooldown node splitting
                illegal_oncooldown_nodes_switches = e.get_illegal_oncoolown_substations_switches()
                if illegal_oncooldown_nodes_switches is not None:
                    substations_changed = self.substations_ids[illegal_oncooldown_nodes_switches]
                    if sum(illegal_oncooldown_nodes_switches) > 0:
                        # put all switches to 0 for illegal oncooldown substation use
                        for substation_changed in substations_changed:
                            expected_subaction_length = len(
                                action.get_substation_switches(substation_changed, False)[1])
                            action.set_substation_switches(substation_changed, np.zeros(expected_subaction_length))
                        # self.cancel_action_from_has_been_changed(action, illegal_oncooldown_nodes_switches)  # cancel illegal moves
                        e.text += ' Ignoring node switches of on-cooldown substations: %s.' % \
                                  ', '.join(list(map(str, np.where(substations_changed)[0])))

            # Resubmit step with modified valid action and return either exception of new step, or this exception
            obs, correct_step, done = self.step(action, _is_simulation=_is_simulation)

            # for flag, return the info of step with corrected action if there is an exception, since it would be more
            # important that the illegal moves which would not be raised on second step call with corrected action
            flag = correct_step if correct_step else e
            return obs, flag, done  # Return done, not False, because step might diverge

        try:
            # Load next timestep entries, compute one loadflow, then potentially cascading failure
            self.load_entries_from_next_timestep(is_simulation=_is_simulation)
            self._compute_loadflow_cascading(is_simulation=_is_simulation)
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
            if self.renderer is not None and not _is_simulation:
                self.render(None, game_over=True, cascading_frame_id=None)
            return observation, flag, done
        if np.sum(are_isolated_prods) > self.max_number_prods_game_over:
            observation = None
            flag = TooManyProductionsCut('There are %d isolated productions; at most %d tolerated' % (
                np.sum(are_isolated_prods), self.max_number_prods_game_over))
            done = True
            if self.renderer is not None and not _is_simulation:
                self.render(None, game_over=True, cascading_frame_id=None)
            return observation, flag, done

        return self.export_observation(), None, False

    def simulate(self, action):
        # Copy variables of a step: timestep id, mpc (~grid), topology (stand-alone in self), and lists of overflows
        before_timestep_id = self.current_timestep_id
        before_mpc = copy.deepcopy(self.grid.mpc)
        before_grid_topology = copy.deepcopy(self.grid.topology)
        before_are_loads = copy.deepcopy(self.grid.are_loads)
        before_n_timesteps_overflowed_lines = copy.deepcopy(self.n_timesteps_soft_overflowed_lines)
        before_n_timesteps_soft_overflowed_lines = copy.deepcopy(self.n_timesteps_soft_overflowed_lines)
        before_timesteps_before_lines_reconnectable = copy.deepcopy(self.timesteps_before_lines_reconnectable)
        before_timesteps_before_lines_reactionable = copy.deepcopy(self.timesteps_before_lines_reactionable)
        before_timesteps_before_nodes_reactionable = copy.deepcopy(self.timesteps_before_nodes_reactionable)
        before_previous_timestep = self.previous_timestep
        before_current_date = copy.deepcopy(self.current_date)
        before_previous_date = copy.deepcopy(self.previous_date)
        before_n_loads_cut = self.n_loads_cut
        before_n_prods_cut = self.n_prods_cut
        before_grid_filename = self.grid.filename

        # Save grid AC or DC normal mode, and force DC mode for simulate
        before_dc = self.grid.dc_loadflow
        #self.grid.dc_loadflow = True #let's run everything in the same mode for now because we did not assess its impact

        def reload_minus_1_timestep():
            self.grid.mpc = before_mpc  # Change grid mpc before apply topo
            self.grid.are_loads = before_are_loads
            self.current_timestep_id = before_timestep_id
            self.grid.topology = copy.deepcopy(before_grid_topology)
            self.current_date = before_current_date
            self.n_timesteps_soft_overflowed_lines = before_n_timesteps_overflowed_lines
            self.timesteps_before_lines_reconnectable = before_timesteps_before_lines_reconnectable
            self.timesteps_before_lines_reactionable = before_timesteps_before_lines_reactionable
            self.timesteps_before_nodes_reactionable = before_timesteps_before_nodes_reactionable
            self.previous_timestep = before_previous_timestep
            self.previous_date = before_previous_date
            self.n_timesteps_soft_overflowed_lines = before_n_timesteps_soft_overflowed_lines
            self.n_loads_cut = before_n_loads_cut
            self.n_prods_cut = before_n_prods_cut
            self.grid.filename = before_grid_filename
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
        observation.timesteps_before_lines_reactionable = self.timesteps_before_lines_reactionable
        observation.timesteps_before_nodes_reactionable = self.timesteps_before_nodes_reactionable
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

        observation.date_year = self.current_date.year
        observation.date_month = self.current_date.month
        observation.date_day = self.current_date.day
        observation.date_hour = self.current_date.hour
        observation.date_minute = self.current_date.minute
        observation.date_second = self.current_date.second

        return observation

    def get_current_datetime(self):
        return self.current_date

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
        # pygame.quit()

        if self.renderer is None:
            self.renderer = initialize_renderer()

        # Retrieve lines capacity usage (for plotting power lines with appropriate colors and widths)
        lines_capacity_usage = self.grid.export_lines_capacity_usage(safe_mode=True)

        prods_values = self.grid.mpc['gen'][:, 1]
        loads_values = self.grid._consistent_ordering_loads()(self.grid.mpc['bus'][self.grid.are_loads, 2])
        lines_por_values = self.grid.mpc['branch'][:, 13]
        lines_service_status = self.grid.mpc['branch'][:, 10]

        substations_ids = self.grid.mpc['bus'][self.grid.n_nodes // 2:]
        # Based on the action, determine if substations has been touched (i.e. there was a topological change involved
        # in the associated substation)
        if self.last_action is not None and cascading_frame_id is not None:
            has_been_changed = self.get_changed_substations(self.last_action)
        else:
            has_been_changed = np.zeros((len(substations_ids),))

        are_isolated_loads, are_isolated_prods, _ = self.grid._count_isolated_loads(self.grid.mpc, self.grid.are_loads)
        number_unavailable_lines = sum(self.timesteps_before_lines_reconnectable > 0) + \
                                   sum(self.timesteps_before_lines_reactionable > 0)
        number_unavailable_nodes = sum(self.timesteps_before_nodes_reactionable > 0)

        initial_topo = self.initial_topology
        current_topo = self.grid.get_topology()
        distance_ref_grid = sum(np.asarray(initial_topo.get_zipped()) != np.asarray(current_topo.get_zipped()))

        max_number_isolated_loads = self.__parameters.get_max_number_loads_game_over()
        max_number_isolated_prods = self.__parameters.get_max_number_prods_game_over()

        # Compute the number of used nodes per substation
        current_observation = self.export_observation()
        n_nodes_substations = []
        for substation_id in self.substations_ids:
            substation_conf = current_observation.get_nodes_of_substation(substation_id)[0]
            n_nodes_substations.append(1 + int(len(list(set(substation_conf))) == 2))

        self.renderer.render(lines_capacity_usage, lines_por_values, lines_service_status, self.epoch, self.timestep,
                             self.current_timestep_id if not timestep_id else timestep_id, prods=prods_values,
                             loads=loads_values, date=self.current_date if date is None else date,
                             are_substations_changed=has_been_changed, number_nodes_per_substation=n_nodes_substations,
                             number_loads_cut=sum(are_isolated_loads), number_prods_cut=sum(are_isolated_prods),
                             number_nodes_splitting=sum(self.last_action.get_node_splitting_subaction())
                             if self.last_action is not None else 0,
                             number_lines_switches=sum(self.last_action.get_lines_status_subaction())
                             if self.last_action is not None else 0, distance_initial_grid=distance_ref_grid,
                             number_off_lines=sum(self.grid.get_lines_status() == 0),
                             number_unavailable_lines=number_unavailable_lines,
                             number_unactionable_nodes=number_unavailable_nodes,
                             max_number_isolated_loads=max_number_isolated_loads,
                             max_number_isolated_prods=max_number_isolated_prods, game_over=game_over,
                             cascading_frame_id=cascading_frame_id)

        if self.latency:
            sleep(self.latency)

    def get_changed_substations(self, action):
        """ Computes the boolean array of changed substations from an Action.
        """
        assert isinstance(action, Action), 'Should not happen'
        has_been_changed = np.zeros((len(self.substations_ids),))
        reordered_action = self.grid.get_topology().mapping_permutation(action.get_node_splitting_subaction())
        n_elements_substations = self.grid.number_elements_per_substations
        offset = 0
        for i, (substation_id, n_elements) in enumerate(zip(self.substations_ids, n_elements_substations)):
            has_been_changed[i] = np.any([l != 0 for l in reordered_action[offset:offset + n_elements]])
            offset += n_elements

        return has_been_changed.astype(bool)

    def parameters_environment_tostring(self):
        return self.__parameters.__str__()
