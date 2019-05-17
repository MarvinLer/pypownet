"""This file constains tests where we use simulate before acting"""

from pypownet.environment import RunEnv, ElementType
from pypownet.runner import Runner
from pypownet.agent import *
from pypownet.game import TooManyProductionsCut, TooManyConsumptionsCut, DivergingLoadflowException
from tests.test_basic import WrappedRunner, get_verbose_node_topology
import math
import pprint


def get_n_first_lines_from_csv(nb_lines, csv_file_path):
    content = np.genfromtxt(csv_file_path, np.float32, delimiter=";")
    return content[1:nb_lines + 1]


class DoNothing(Agent):
    def act(self, observation):
        action_length = self.environment.action_space.action_length
        return np.zeros(action_length)


class CustomGreedySearch(Agent):
    """ This agent is a copy of the Agent.GreedySearch, so it simulates many different things but in the end returns a
    do nothing action, for more information check description of function: test_simulate_Agent_CustomGreedySearch.
    """
    def __init__(self, environment):
        super().__init__(environment)
        print("CustomGreedySearch created...")
        self.verbose = True
        self.consos_save = {
            "do_nothing": {
                1: [], 2: [], 3: []
            },
            "line_opening": {
                1: [], 2: [], 3: []
            },
            "node_change" : {
                1: [], 2: [], 3: []
            }
        }
        self.real_consos_save = []

        self.current_step = 1
        self.ioman = ActIOnManager(destination_path='saved_actions.csv')

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        import itertools

        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        number_lines = action_space.lines_status_subaction_length
        # Will store reward, actions, and action name, then eventually pick the maximum reward and retrieve the
        # associated values
        rewards, actions, names = [], [], []

        # Test doing nothing
        if self.verbose:
            print(' Simulation with no action', end='')
        action = action_space.get_do_nothing_action()
        res = self.environment.simulate(action, do_sum=False, obs_for_tests=True)
        simulated_obs = self.environment.observation_space.array_to_observation(res[0])
        reward = res[1]
        rewards.append(reward)
        actions.append(action)
        names.append('no action')
        self.consos_save["do_nothing"][self.current_step].append(simulated_obs.active_loads.astype(int))

        # Test every line opening
        for l in range(number_lines):
            if self.verbose:
                print(' Simulation with switching status of line %d' % l, end='')
            action = action_space.get_do_nothing_action(as_class_Action=True)
            action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
            res = self.environment.simulate(action, do_sum=False, obs_for_tests=True)
            simulated_obs = self.environment.observation_space.array_to_observation(res[0])
            reward = res[1]
            rewards.append(reward)
            actions.append(action)
            names.append('switching status of line %d' % l)
            self.consos_save["line_opening"][self.current_step].append(simulated_obs.active_loads.astype(int))

        # For every substation with at least 4 elements, try every possible configuration for the switches
        for substation_id in action_space.substations_ids:
            substation_n_elements = action_space.get_number_elements_of_substation(substation_id)
            if 8 > substation_n_elements > 3:
                # Look through all configurations of n_elements binary vector with first value fixed to 0
                for configuration in list(itertools.product([0, 1], repeat=substation_n_elements - 1)):
                    new_configuration = [0] + list(configuration)
                    if self.verbose:
                        print(' Simulation with change in topo of sub. %d with switches %s' % (
                            substation_id, repr(new_configuration)), end='')
                    # Construct action
                    action = action_space.get_do_nothing_action(as_class_Action=True)
                    action_space.set_substation_switches_in_action(action=action, substation_id=substation_id,
                                                                   new_values=new_configuration)
                    res = self.environment.simulate(action, do_sum=False, obs_for_tests=True)
                    if res[0] is not None:
                        simulated_obs = self.environment.observation_space.array_to_observation(res[0])
                        reward = res[1]
                        rewards.append(reward)
                        actions.append(action)
                        names.append('change in topo of sub. %d with switches %s' % (substation_id,
                                                                                     repr(new_configuration)))
                    if simulated_obs is not None:
                        self.consos_save["node_change"][self.current_step].append(simulated_obs.active_loads.astype(int))

        # Take the best reward, and retrieve the corresponding action
        best_reward = max(rewards)
        best_index = rewards.index(best_reward)
        best_action = actions[best_index]
        best_action_name = names[best_index]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        self.real_consos_save.append(observation.active_loads.astype(int))
        self.current_step += 1
        return action_space.get_do_nothing_action()


class Agent_test_LineChangePersistance(Agent):
    """This agent cuts a line and checks for 9 steps that it is still cut"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LineChangePersistance created...")

        self.current_step = 1
        self.line_to_cut = 18

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        print("lines_status = ", list(observation.lines_status.astype(int)))

        if self.current_step == 1:
            # SWITCH OFF LINE 18
            print("we switch off line {}".format(self.line_to_cut))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
            # here we would like to simulate before submitting the action
            # reward, simulated_obs = self.environment.simulate(action, obs_for_tests=True)

            res = self.environment.simulate(action)
            # for elem in res:
            #     print("------> ", elem)
            # simulated_observation = self.environment.observation_space.array_to_observation(res[0])
            # print("res = ")
            # print(simulated_observation)
            # print("simulated obs = ", simulated_obs)

            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
        else:
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  0, 1])

        self.current_step += 1

        return action


class small_Agent_test_RewardError(Agent):
    """For more info, check function small_test_simulate_Agent_test_RewardError description.
    """
    def __init__(self, environment, sim_bool):
        super().__init__(environment)
        print("small_Agent_test_RewardError created...")

        self.current_step = 1
        self.line_to_cut = 18
        self.sim_bool = sim_bool
        self.node_to_change = 3

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        print("lines_status = ", list(observation.lines_status.astype(int)))
        res = self.environment.simulate(action)
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # get current configuration
        current_conf, types = observation.get_nodes_of_substation(self.node_to_change)

        if self.current_step == 1:
            # SWITCH OFF LINE 18
            print("we switch off line {}".format(self.line_to_cut))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
            # here we would like to simulate before submitting the action
            if self.sim_bool:
                print("***** we simulate *****")
                # reward, simulated_obs = self.environment.simulate(action, obs_for_tests=True)
                res = self.environment.simulate(action)

        if self.current_step == 2:
            # we connect the fourth element to busbar 1
            target_configuration[-1] = 1
            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)
            if self.sim_bool:
                print("***** we simulate *****")
                # reward, simulated_obs = self.environment.simulate(action, obs_for_tests=True)
                res = self.environment.simulate(action)

        self.current_step += 1
        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())

        return action


class Agent_test_AccurateConsumptionLoadings(Agent):
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_InputLoadValues created...")

        self.current_step = 1
        self.line_to_cut = None
        self.path = "tests/parameters/default14_for_tests/level0/chronics/a/_N_loads_p.csv"
        self.path_planned = "tests/parameters/default14_for_tests/level0/chronics/a/_N_loads_p_planned.csv"
        self.abspath = os.path.abspath(self.path)
        self.abspath_planned = os.path.abspath(self.path_planned)
        self.from_file_loads_content = get_n_first_lines_from_csv(5, self.abspath)
        self.from_file_loads_planned_content = get_n_first_lines_from_csv(5, self.abspath_planned)
        print("loads from file = ", self.from_file_loads_content)
        print("loads planned from file = ", self.from_file_loads_planned_content)

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
         try:
             observation = self.environment.observation_space.array_to_observation(observation)
         except Exception as e:
             raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        current_load_powers = observation.active_loads
        print("current_load_powers = ", current_load_powers)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            expected = self.from_file_loads_content[1]
            print("expected = ", expected)
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

            res = self.environment.simulate(action, obs_for_tests=True)
            simulated_obs = self.environment.observation_space.array_to_observation(res[0])
            simulated_load_powers = simulated_obs.active_loads
            print("simulated load power = ", simulated_load_powers)
            expected_planned = self.from_file_loads_planned_content[2]
            for expected_elem, core_elem in zip(expected_planned, simulated_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

        if self.current_step == 2:
            expected = self.from_file_loads_content[2]
            print("expected = ", expected)
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

            res = self.environment.simulate(action, obs_for_tests=True)
            simulated_obs = self.environment.observation_space.array_to_observation(res[0])
            simulated_load_powers = simulated_obs.active_loads
            print("simulated load power = ", simulated_load_powers)
            expected_planned = self.from_file_loads_planned_content[3]
            for expected_elem, core_elem in zip(expected_planned, simulated_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

        if self.current_step == 3:
            expected = self.from_file_loads_content[3]
            print("expected = ", expected)
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

            res = self.environment.simulate(action, obs_for_tests=True)
            simulated_obs = self.environment.observation_space.array_to_observation(res[0])
            simulated_load_powers = simulated_obs.active_loads
            print("simulated load power = ", simulated_load_powers)
            expected_planned = self.from_file_loads_planned_content[4]
            for expected_elem, core_elem in zip(expected_planned, simulated_load_powers):
                error_diff = core_elem - expected_elem
                assert(error_diff == 0.0)
                print("error diff = ", error_diff)

        self.current_step += 1

        return action



########################################################################################################################
########################################################################################################################
########################################################################################################################


def test_simulate_Agent_test_SimulateThenAct():
    """This function creates an Agent that cut a line and checks for 9 steps that it is still cut, while using the
    function env.simulate."""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 10

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_LineChangePersistance(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False])
    assert(list(actions_recap) == [None, None, None, None, None, None, None, None, None, None])


def test_simulate_Agent_test_RewardError():
    """ Function to test if a reward is the same whether we simulate during our work or not.
    This function creates a small_Agent_test_RewardError which works for 3 steps.
    first instance ==> the agent simulates in addition to cutting a line and change a node's topology
    second instance ==> the agent just cuts a line and changes a node's topology, without simulation.
    then we compare the reward. They must be equal.
    """
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 3

    env_class = RunEnv

    res = []

    for i in range(2):
        print("############################# current INSTANCE = {} #############################".format(i))
        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        if i == 0:
            agent = small_Agent_test_RewardError(env, True)
        else:
            agent = small_Agent_test_RewardError(env, False)
        # Instantiate game runner and loop
        runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
        res.append(final_reward)
        print("Obtained a final reward of {}".format(final_reward))
        print("game_overs = ", game_overs)
        print("actions_recap = ", actions_recap)
        assert(niter == len(game_overs) == len(actions_recap))
        assert(list(game_overs) == [False, False, False])
        assert(list(actions_recap) == [None, None, None])

    for reward in res[1:]:
        assert(reward == res[0])


def test_simulate_Agent_exhaustive_test_RewardError():
    """ Function to test if a reward is the same whether we simulate during our work or not.
    This function creates first a CustomGreedyAgent that will simulate all possible changes, but do nothing in the end.
    Then, second instance ==> A do nothing Agent.
    Finally, we compare the rewards. They must be equal.
    """
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 3

    env_class = RunEnv

    res = []

    for i in range(2):
        print("############################# current INSTANCE = {} #############################".format(i))
        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        if i == 0:
            agent = CustomGreedySearch(env)
        else:
            agent = DoNothing(env)
        # Instantiate game runner and loop
        runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
        res.append(final_reward)
        print("Obtained a final reward of {}".format(final_reward))
        print("game_overs = ", game_overs)
        print("actions_recap = ", actions_recap)
        assert(niter == len(game_overs) == len(actions_recap))
        assert(list(game_overs) == [False, False, False])
        assert(list(actions_recap) == [None, None, None])

    for reward in res[1:]:
        assert(reward == res[0])


def test_simulate_Agent_CustomGreedySearch():
    """This function creates an Agent that does all possible actions while simulating. It checks that WITHIN a timestep,
    the Observation of all Consumptions(Observation.are_loads) are still the same.
    """
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = CustomGreedySearch(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Agent's data = ")
    pprint.pprint(agent.consos_save)
    print("real observation after  = ", agent.real_consos_save)

    # Post processing - verif that real obs step 1 == simulated_obs step 1, etc...
    results = {
        "do_nothing": {
            1: [], 2: [], 3: []
        },
        "line_opening": {
            1: [], 2: [], 3: []
        },
        "node_change" : {
            1: [], 2: [], 3: []
        }
    }

    for i in range(1, 4): # 1, 2, 3.
        first_flows = agent.consos_save["line_opening"][i][0]
        for j, flow in enumerate(agent.consos_save["line_opening"][i]):
            if list(flow) == list(first_flows):
                results["line_opening"][i].append((True))
            else:
                results["line_opening"][i].append((False))
        # ALL THE TESTS ARE HERE, MUST BE TRUE EACH TIME
        nparray = np.array(results["line_opening"][i])
        assert(nparray.all())

    for i in range(1, 4): # 1, 2, 3.
        first_flows = agent.consos_save["node_change"][i][0]
        for j, flow in enumerate(agent.consos_save["node_change"][i]):
            if list(flow) == list(first_flows):
                results["node_change"][i].append((True))
            else:
                results["node_change"][i].append((False))
        # ALL THE TESTS ARE HERE, MUST BE TRUE EACH TIME
        nparray = np.array(results["node_change"][i])
        assert(nparray.all())

    print("=========================== FINAL RESULTS ===========================")
    pprint.pprint(results)

    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False])
    assert(list(actions_recap) == [None, None, None])


def test_simulate_Agent_test_AccurateConsumptionLoadings():
    """This function creates an Agent
    """
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_AccurateConsumptionLoadings(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False])
    assert(list(actions_recap) == [None, None, None])


# test_simulate_Agent_test_SimulateThenAct()
# test_simulate_Agent_test_RewardError()
# test_simulate_Agent_exhaustive_test_RewardError()
# test_simulate_Agent_CustomGreedySearch()
# test_simulate_Agent_test_AccurateConsumptionLoadings()

