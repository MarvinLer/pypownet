__author__ = 'marvinler'
import os
import numpy as np
import copy
from oct2py import octave
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
from pypownet.scenarios_chronic import Scenario
import pypownet.env


class DivergingLoadflowException(Exception):
    pass


class GridNotConnexeException(Exception):
    pass


def compute_flows_a(active, reactive, voltage):
    # TODO: verify that the formula is correct
    return np.asarray(np.sqrt(active ** 2. + reactive ** 2.) / (3 ** .5 * voltage))


class Grid(object):
    def __init__(self, src_filename, dc_loadflow, new_slack_bus, new_imaps, verbose=False):
        self.filename = src_filename
        self.dc_loadflow = dc_loadflow
        self.save_io = False
        self.verbose = verbose

        # Container output of Matpower usual functions (mpc structure); contains all grid params/values
        self.mpc = octave.loadcase(self.filename, verbose=False)
        # Change thermal limits
        self.mpc['branch'][:, 5] = new_imaps
        self.mpc['branch'][:, 6] = new_imaps
        self.mpc['branch'][:, 7] = new_imaps

        self.new_slack_bus = new_slack_bus
        self.are_loads = np.logical_or(self.mpc['bus'][:, 2] != 0, self.mpc['bus'][:, 3] != 0)

        # Instantiate once and for all matpower options, and change algo if AC
        self.mpoption = octave.mpoption() if dc_loadflow else octave.mpoption('pf.alg', 'NR')

        # Fixed ids of substations associated with prods, loads and lines (init.,  all elements on real substation id)
        self.loads_ids = self.mpc['bus'][self.are_loads, 0]

        self.n_nodes = len(self.mpc['bus'])
        self.n_prods = len(self.mpc['gen'])
        self.n_loads = np.sum(self.are_loads)
        self.n_lines = len(self.mpc['branch'])

        # Topology container: initially, all elements are on the node 0
        self.topology = Topology(prods_nodes=np.zeros((self.n_prods,)), loads_nodes=np.zeros((self.n_loads,)),
                                 lines_or_nodes=np.zeros((self.n_lines,)), lines_ex_nodes=np.zeros((self.n_lines,)),
                                 lines_service=self.mpc['branch'][:, 10],
                                 mapping_permutation=self.compute_topological_mapping_permutation())

    def compute_topological_mapping_permutation(self):
        """ Compute a permutation that shuffles the construction order of a topology (prods->loads->lines or->lines ex) into a
            representation where all elements of a substation are consecutives values (same order, but locally)."""
        prods_ids = self.mpc['gen'][:, 0]
        loads_ids = self.mpc['bus'][self.are_loads, 0]
        lines_or = self.mpc['branch'][:, 0]
        lines_ex = self.mpc['branch'][:, 0]
        loads_offset = self.n_prods
        lines_or_offset = self.n_prods + self.n_loads
        lines_ex_offset = self.n_prods + self.n_loads + self.n_lines

        mapping = []
        for node_id in self.mpc['bus'][:self.n_nodes // 2, 0]:
            node_mapping = []
            if node_id in prods_ids:
                node_index = np.where(prods_ids == node_id)[0][0]  # Only one prod per substation
                node_mapping.append(node_index)
            if node_id in loads_ids:
                node_index = np.where(loads_ids == node_id)[0][0] + loads_offset  # Only one load per subst.
                node_mapping.append(node_index)
            if node_id in lines_or:
                node_index = np.where(lines_or == node_id)[0] + lines_or_offset  # Possible multiple lines per subst
                node_mapping.extend(node_index)
            if node_id in lines_ex:
                node_index = np.where(lines_ex == node_id)[0] + lines_ex_offset
                node_mapping.extend(node_index)
            mapping.append(node_mapping)

        # Verify that the mapping array has unique values
        assert len(np.concatenate(mapping)) == len(
            np.unique(np.concatenate(mapping))), 'Mapping does not have unique values, should not happen'
        assert len(mapping) == self.n_nodes // 2, 'Mapping does not have one configuration per substation'

        return mapping

    def _synchronize_bus_types(self):
        """ Checks for every bus if there are any active line connected to it; if not, set the associated mpc.bus bus
        type to 4 for the current self. """
        mpc = self.mpc
        bus = mpc['bus']
        gen = mpc['gen']
        branch = mpc['branch']

        # Compute the buses origin or extremity of at least one connected line
        branch_online = branch[branch[:, 10] != 0]  # Retrieves non-disconnected branches
        bus_orex = np.unique(branch_online[:, [0, 1]])
        bus_prods = gen[:, 0].astype(int)
        bus_loads = bus[self.are_loads, 0].astype(int)
        non_isolated_bus = np.unique(np.concatenate((bus_prods, bus_loads, bus_orex)).astype(int))
        # Replace type of isolated bus (i.e. not extremity of line, and does not have prod or load) by 4
        for b, bus_id in enumerate(bus[:, 0].astype(int)):
            # If bus is isolated, put the value 4 for its type
            if bus_id not in non_isolated_bus:
                bus[b, 1] = 4
            else:  # Otherwise, put 2 for PV node, 1 else
                if bus_id in bus_prods:
                    # If this is the slack bus and a production, then put its type to 3 (slack bus)
                    if bus_id == self.new_slack_bus:
                        bus[b, 1] = 3
                        continue
                    bus[b, 1] = 2
                else:
                    bus[b, 1] = 1

    def __vanilla_matpower_callback(self, mpc, pprint=None, fname=None, verbose=False):
        # DC loadflow
        if self.dc_loadflow:
            if pprint and fname:
                octave.savecase(fname, mpc)
                output = octave.rundcpf(mpc, octave.mpoption(), pprint, fname, verbose=verbose)
            else:
                output = octave.rundcpf(mpc, octave.mpoption(), verbose=verbose)
        else:  # AC loadflow
            if pprint and fname:
                output = octave.runpf(mpc, octave.mpoption(), pprint, fname, verbose=verbose)
            else:
                output = octave.runpf(mpc, octave.mpoption(), verbose=verbose)

        loadflow_success = output['success']
        return output, loadflow_success

    def compute_loadflow(self, perform_cascading_failure):
        # Ensure that all isolated bus has their type put to 4 (otherwise matpower diverged)
        """ Given the current state of the grid (topology + injections), compute the new loadflow of the grid. This function
        subtreats the Octave pipeline to self.__vanilla_matpower_callback.

        :param perform_cascading_failure: True to compute a loadflow after loading the new injections
        :return: :raise GridNotConnexeException: If the grid is not connex, matpower raises an error; when an error is
            raised after a matpower callback, this exception is raised.
        """
        self._synchronize_bus_types()

        mpc = self.mpc

        # Compute one matpower loadflow computation given the current grid state
        if self.save_io:  # Paths for grid s_t+0.5 and s_t+1
            pprint = os.path.abspath(os.path.join('tmp', 'pp' + os.path.basename(self.filename)))
            fname = os.path.abspath(os.path.join('tmp', os.path.basename(self.filename)))
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
        else:
            pprint, fname = None, None
        output, loadflow_success = self.__vanilla_matpower_callback(mpc, pprint, fname)

        # If matpower returned a diverging computation, raise proper exception
        if not loadflow_success:
            raise GridNotConnexeException('The grid is not connexe')

        # Simulate cascading failure: success. switches off overflowed lines, then compute loadflow and loop until
        # no lines are overflowed or an error is raised
        if perform_cascading_failure:
            if self.verbose:
                print('  Simulating cascading failure')
            pf = copy.deepcopy(output)
            depth = 0
            while True:
                bus = pf['bus']
                branch = pf['branch']
                # Compute the per-line Ampere values; column 13 is Pf, 14 Qf
                voltage = np.array([bus[np.where(bus[:, 0] == origin), 7] for origin in branch[:, 0]]).flatten()
                #voltage = bus[branch[:, 0], 7]
                branches_flows_a = compute_flows_a(active=branch[:, 13], reactive=branch[:, 14], voltage=voltage)
                branches_thermal_limits = branch[:, 5]

                # Sanity check: check flows and angles are not NaN
                if np.isnan(branch[:, 13]).any() or np.isnan(bus[:, 8]).any():
                    raise DivergingLoadflowException('Loadflow of has diverged')

                # If no lines are overflowed, end the cascading failure simulation (flows a > 0)
                if np.sum(branches_flows_a > branches_thermal_limits) == 0:
                    if self.verbose:
                        print('  ok')
                    break
                if self.verbose:
                    print('    depth %d: %d overflowed lines' % (depth, np.sum(branches_flows_a > branches_thermal_limits)))

                # Otherwise, switch off overflowed lines
                branch[branches_flows_a > branches_thermal_limits, 10] = 0

                if self.save_io:
                    fname = fname[:-2] + '_cascading%d.m' % depth
                    pprint = pprint[:-2] + '_cascading%d.m' % depth
                else:
                    fname, pprint = None, None
                pf, pf_success = self.__vanilla_matpower_callback(pf, pprint, fname, False)

                if not pf_success:
                    raise GridNotConnexeException(
                        'Cascading failure of depth %d lead to a non-connexe grid' % (depth + 1))
                depth += 1

        self.mpc = output
        return loadflow_success

    def load_scenario(self, scenario, do_trigger_lf_computation=True, cascading_failure=False):
        """ Loads a scenario from class Scenario: contains P and V values for prods, and P and Q values for loads.

        :param scenario: an instance of class Scenario
        :param do_trigger_lf_computation: True to compute a loadflow after loading the new injections
        :param cascading_failure: True to simulate a cascading failure after loading the new injections
        :return: if do_trigger_lf_computation then the result of self.compute_loadflow else nothing
        """
        assert isinstance(scenario, Scenario), 'Trying to load a scenario which is not an instance of class Scenario'

        # Change the filename of self to pretty print middle-end created temporary files
        self.filename = 'scenario%d.m' % (scenario.get_id())

        mpc = self.mpc
        gen = mpc['gen']
        bus = mpc['bus']

        # Import new productions values
        prods_p = scenario.get_prods_p()
        prods_v = scenario.get_prods_v()
        # Check that there are the same number of productions names and values
        assert len(prods_v) == len(prods_p), 'Not the same number of active prods values than reactives prods'
        gen[:, 1] = prods_p
        gen[:, 5] = prods_v

        # Import new loads values
        loads_p = scenario.get_loads_p()
        loads_q = scenario.get_loads_q()
        # Check that there are the same number of productions names and values
        assert len(loads_q) == len(loads_p), 'Not the same number of active loads values than reactives loads'
        bus[self.are_loads, 2] = loads_p
        bus[self.are_loads, 3] = loads_q

        if do_trigger_lf_computation:
            return self.compute_loadflow(perform_cascading_failure=cascading_failure)
        return

    def disconnect_line(self, id_line):
        self.mpc['branch'][id_line, 10] = 0

    def reconnect_line(self, id_line):
        self.mpc['branch'][id_line, 10] = 1

    def apply_topology(self, new_topology):
        # Verify new specified topology is of good number of elements and only 0 or 1
        """ Applies a new topology to self. topology should be an instance of class Topology, with computed values to
        be replaced in self.

        :param new_topology: an instance of Topology, with destination values for the nodes values/lines service status
        """
        assert new_topology.get_length() == self.get_topology().get_length()
        assert set(new_topology.get_zipped()).issubset([0, 1])

        # Split topology vector into the four chunks
        new_prods_nodes, new_loads_nodes, new_lines_or_nodes, new_lines_ex_nodes, new_lines_service = \
            new_topology.get_unzipped()

        # Function to find the true id of the substation associated with one node
        node_to_substation = lambda node_id: str(node_id).replace(ARTIFICIAL_NODE_STARTING_STRING, '')

        # Change nodes ids of productions
        gen = self.mpc['gen']
        for p, (prod_id, new_prod_node) in enumerate(zip(self.mpc['gen'][:, 0], new_prods_nodes)):
            prod_substation = node_to_substation(prod_id)
            if new_prod_node == 1:
                gen[p, 0] = float(ARTIFICIAL_NODE_STARTING_STRING + prod_substation)
            else:
                gen[p, 0] = float(node_to_substation(prod_id))

        # Change nodes ids of lines (1 origin and 1 extremity per line)
        branch = self.mpc['branch']
        for li, (line_or_id, new_line_or_node) in enumerate(zip(branch[:, 0], new_lines_or_nodes)):
            line_substation = node_to_substation(line_or_id)
            if new_line_or_node == 1:
                branch[li, 0] = float(ARTIFICIAL_NODE_STARTING_STRING + line_substation)
            else:
                branch[li, 0] = float(line_substation)
        for li, (line_ex_id, new_line_ex_node) in enumerate(zip(branch[:, 1], new_lines_ex_nodes)):
            line_substation = node_to_substation(line_ex_id)
            if new_line_ex_node == 1:
                branch[li, 1] = float(ARTIFICIAL_NODE_STARTING_STRING + line_substation)
            else:
                branch[li, 1] = float(line_substation)

        # Change nodes ids of loads
        bus = self.mpc['bus']
        for lo, (load_node, new_load_node) in enumerate(zip(self.topology.loads_nodes, new_loads_nodes)):
            # If the node on which a load is connected is swap, then swap P and Q values for both nodes
            if new_load_node != load_node:
                are_loads = np.where(self.are_loads[:self.n_nodes // 2], self.are_loads[:self.n_nodes // 2],
                                     self.are_loads[self.n_nodes // 2:])
                id_bus = np.where(are_loads)[0][lo] % (self.n_nodes // 2)
                # Copy first node P and Q
                tmp = copy.deepcopy(bus[id_bus, [2, 3]])
                # Replace their values with the one of its associated node
                bus[id_bus, 2] = bus[(lo + self.n_nodes // 2) % self.n_nodes, 2]
                bus[id_bus, 3] = bus[(lo + self.n_nodes // 2) % self.n_nodes, 3]
                # Paste tmp values into asso. node
                bus[(id_bus + self.n_nodes // 2) % self.n_nodes, 2] = tmp[0]
                bus[(id_bus + self.n_nodes // 2) % self.n_nodes, 3] = tmp[1]
                # Change the ids of current loads
                tmp_id = id_bus if self.are_loads[id_bus] else id_bus + self.n_nodes // 2
                self.are_loads[tmp_id] = False
                self.are_loads[(tmp_id + self.n_nodes // 2) % self.n_nodes] = True

        # Change line status (equivalent to reco/disco calls as a function of values of new_line_status)
        branch[:, 10] = new_lines_service

        self.topology = new_topology

    def get_topology(self):
        return self.topology

    def export_to_observation(self):
        """ Exports the current grid state into an observation. """
        mpc = self.mpc
        bus = mpc['bus']
        gen = mpc['gen']
        branch = mpc['branch']

        to_array = lambda array: np.asarray(array)
        # Generators data
        active_prods = to_array(gen[:, 1])  # Pg
        reactive_prods = to_array(gen[:, 2])  # Qg
        voltage_prods = to_array(gen[:, 5])  # Vg

        # Branch data
        active_flows_origin = to_array(branch[:, 13])  # Pf
        reactive_flows_origin = to_array(branch[:, 14])  # Qf
        active_flows_extremity = to_array(branch[:, 15])  # Pt
        reactive_flows_extremity = to_array(branch[:, 16])  # Qt
        voltage_origin = to_array([bus[np.where(bus[:, 0] == origin), 7] for origin in branch[:, 0]]).flatten()
        voltage_extremity = to_array([bus[np.where(bus[:, 0] == origin), 7] for origin in branch[:, 1]]).flatten()
        flows_a = compute_flows_a(active_flows_origin, reactive_flows_origin, voltage_origin)
        relative_thermal_limit = to_array(flows_a / branch[:, 5])  # elementwise division of flow a and rateA

        # Loads data
        loads_buses = bus[self.are_loads, :]  # Select lines of loads buses
        active_loads = to_array(loads_buses[:, 2])
        reactive_loads = to_array(loads_buses[:, 3])
        voltage_loads = to_array(loads_buses[:, 7])

        # Topology vector
        topology = self.get_topology().get_zipped()  # Retrieve concatenated version of topology

        return pypownet.env.RunEnv.Observation(active_loads, reactive_loads, voltage_loads,
                                               active_prods, reactive_prods, voltage_prods,
                                               active_flows_origin, reactive_flows_origin, voltage_origin,
                                               active_flows_extremity, reactive_flows_extremity, voltage_extremity,
                                               relative_thermal_limit, topology)

    def export_relative_thermal_limits(self):
        """ Computes and returns the relative thermal limits, i.e. the elementwise division of the flows in Ampere by the
        lines nominal thermal limit.

        :return: a list of size the number of lines of positive values
        """
        mpc = self.mpc
        bus = mpc['bus']
        branch = mpc['branch']
        to_array = lambda array: np.asarray(array)

        # Retrieve P, Q and V value
        active_flows_origin = to_array(branch[:, 13])  # Pf
        reactive_flows_origin = to_array(branch[:, 14])  # Qf
        voltage_origin = to_array([bus[np.where(bus[:, 0] == origin), 7] for origin in branch[:, 0]]).flatten()
        # Compute flows in Ampere using formula compute_flows_a
        flows_a = compute_flows_a(active_flows_origin, reactive_flows_origin, voltage_origin)
        relative_thermal_limits = to_array(flows_a / branch[:, 5])  # elementwise division of flow a and rateA

        return relative_thermal_limits

    def _snapshot(self, dst_fname=None):
        """ Saves a snapshot of current grid state into IEEE format file with path dst_fname. """
        self._synchronize_bus_types()
        if dst_fname is None:
            dst_fname = os.path.abspath(os.path.join('tmp', 'snapshot_' + os.path.basename(self.filename)))
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
        print('Saved snapshot at', dst_fname)
        return octave.savecase(dst_fname, self.mpc)


class Topology(object):
    """
    This class is a container for the topology lists defining the current topological state of a grid. Topology should
    be manipulated using this class, as it maintains the adopted convention consistently.
    """

    def __init__(self, prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes, lines_service,
                 mapping_permutation=None):
        self.prods_nodes = prods_nodes
        self.loads_nodes = loads_nodes
        self.lines_or_nodes = lines_or_nodes
        self.lines_ex_nodes = lines_ex_nodes
        self.lines_service = lines_service

        # Function that sorts the internal topological array into a more intuitive representation: the nodes of the
        # elements of a substation are consecutive (first prods, then loads, then lines origin, then line ext.)
        if not mapping_permutation:  # Identity by default
            self.mapping_permutation = lambda x: x
            self.invert_mapping_permutation = lambda x: x
        else:
            concatenated_mapping_permutation = np.concatenate(mapping_permutation)
            self.mapping_permutation = lambda array: np.concatenate(
                ([array[c] for c in concatenated_mapping_permutation],
                 array[-len(lines_ex_nodes):]))
            invert_indexes = [np.where(concatenated_mapping_permutation == i)[0][0] for i in
                              range(len(concatenated_mapping_permutation))]
            self.invert_mapping_permutation = lambda array: np.concatenate(
                ([array[c] for c in invert_indexes],
                 array[-len(lines_ex_nodes):]))

    def get_zipped(self):
        return self.mapping_permutation(np.concatenate(
            (self.prods_nodes, self.loads_nodes, self.lines_or_nodes, self.lines_ex_nodes, self.lines_service)))

    def get_unzipped(self):
        return self.prods_nodes, self.loads_nodes, self.lines_or_nodes, self.lines_ex_nodes, self.lines_service

    @staticmethod
    def unzip(topology, n_prods, n_loads, n_lines, invert_mapping_function):
        # Shuffle topology parameter based on index positions; invert_mapping_function should be the same as the
        # one used by the environment to convert the sorted topology into its internal representation
        topology_shuffled = invert_mapping_function(topology)

        prods_nodes = topology_shuffled[:n_prods]
        loads_nodes = topology_shuffled[n_prods:n_prods + n_loads]
        lines_or_nodes = topology_shuffled[-3 * n_lines:-2 * n_lines]
        lines_ex_nodes = topology_shuffled[-2 * n_lines:-n_lines]
        lines_service = topology_shuffled[-n_lines:]
        return prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes, lines_service

    def get_length(self):
        return len(self.prods_nodes) + len(self.loads_nodes) + len(self.lines_ex_nodes) + len(
            self.lines_or_nodes) + len(self.lines_service)


if __name__ == '__main__':
    # For testing purposes... might be outdated
    ###########################################

    source = '/home/marvin/Documents/pro/stagemaster_inria/PowerGrid-UI/input/reference_grid118.m'
    grid = Grid(source, False, 69, new_imaps=3900)
    grid._snapshot(os.path.join('tmp', 'snapshot_before.m'))
    top = grid.get_topology()
    topology = grid.get_topology().get_zipped()
    print(topology)
    print(np.argmax(grid.get_topology().get_zipped()))
    print(np.argmax(grid.get_topology().mapping_permutation(topology)))

    offset = 1
    new_topo = copy.deepcopy(topology)
    new_topo[offset] = 1
    new_topo = Topology(*(Topology.unzip(new_topo, grid.n_prods, grid.n_loads, grid.n_lines,
                                         top.invert_mapping_permutation)))
    new_topo.mapping_permutation = grid.get_topology().mapping_permutation
    grid.apply_topology(new_topo)
    grid._snapshot(os.path.join('tmp', 'snapshot_middle.m'))
    print(np.argmax(grid.get_topology().get_zipped()))
    print(grid.get_topology().get_zipped())

    new_topo = copy.deepcopy(topology)
    new_topo[offset] = 0
    new_topo = Topology(*Topology.unzip(new_topo, grid.n_prods, grid.n_loads, grid.n_lines,
                                        top.invert_mapping_permutation))
    new_topo.mapping_permutation = grid.get_topology().mapping_permutation
    grid.apply_topology(new_topo)
    #grid.compute_loadflow(perform_cascading_failure=True)
    grid._snapshot(os.path.join('tmp', 'snapshot_after.m'))
    print(np.argmax(grid.get_topology().get_zipped()))
    print(np.argmax(grid.get_topology().mapping_permutation(topology)))

