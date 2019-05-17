# This Tests folder contains test functions for software Pypownet.

Each file contains specific tests:

## in the file: test_basic.py

test_differencial_topology PASSED                      
test_Agent_test_LineTimeLimitSwitching PASSED          
test_Agent_test_NodeTimeLimitSwitching PASSED          
test_Agent_test_MaxNumberActionnedLines PASSED         
test_Agent_test_MaxNumberActionnedNodes PASSED         
test_Agent_test_BasicSubstationTopologyChange PASSED   
test_Agent_test_AdvancedSubstationTopologyChange PASSED

## in the file: test_core.py

test_core_Agent_test_ObsToArrayAndBack PASSED
test_core_Agent_test_limitOfProdsLost PASSED         
test_core_Agent_test_limitOfLoadsLost PASSED         
test_core_Agent_test_InputProdValues PASSED          
test_core_Agent_test_InputLoadValues PASSED          
test_core_Agent_test_method_obs_are_prods_cut PASSED 
test_core_Agent_test_method_obs_are_loads_cut PASSED 
test_core_Agent_test_Loss_Error PASSED               
test_core_Agent_test_SoftOverflowBreakLimit PASSED   
test_core_Agent_test_SoftOverflowIsBroken PASSED     
test_core_Agent_test_NodeTopoChangePersistence PASSED
test_core_Agent_test_LineChangePersistence PASSED    
test_core_Agent_test_HardOverflowCoefTest PASSED     

## in the file: test_simulate.py

tests/test_simulate.py::test_simulate_Agent_test_SimulateThenAct PASSED            
tests/test_simulate.py::test_simulate_Agent_test_RewardError PASSED                
tests/test_simulate.py::test_simulate_Agent_exhaustive_test_RewardError PASSED     
tests/test_simulate.py::test_simulate_Agent_CustomGreedySearch PASSED              
tests/test_simulate.py::test_simulate_Agent_test_AccurateConsumptionLoadings PASSED  

### Below is presented an extensive description of each function.
### ---------------------------------- test_basic.py ----------------------------------

#### test_differencial_topology():
Simple check to verify that the function **get_differencial_topology(stateA, stateB)** works as intended.
Function **get_differencial_topology(stateA, stateB)** returns the action needed to be taken to reach a node
configuration state B from a node configuration state A.

#### test_Agent_test_LineTimeLimitSwitching():
This function creates an agent that switches a line, then tries to switch it again. (But should be nullified because
 of input param **"n_timesteps_actionned_line_reactionable: 3"**, then after 3 steps, we switch it back up.
The agent's plan is as follows:
+ t = 1, the agent switches off line X,
+ t = 2, he observes that the line X has been switched off
+ t = 2, he tries to switch the line back on, but should be dropped because of the restriction n_timesteps_actionned_line_reactionable: 3
+ t = 3, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
+ t = 3, he tries to switch it on again
+ t = 4, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
+ t = 4, he tries to switch it on again
+ t = 5, THE "SWITCH BACK ON" WORKED
+ t = 5, he tries to cut it again
+ t = 6, must be restricted again. Should still be back on.
 
#### test_Agent_test_NodeTimeLimitSwitching():
This function creates an agent that switches a node topology, then tries to switch it again.
(But should be nullified because of input param **"n_timesteps_actionned_node_reactionable: 3"**, then after 3 steps,
we switch it back up.
The agent's plan is as follows:
+ t == 1, [0, 0, 0] change one element. SHOULD WORK.     (restriction_step = 1)
+ t == 2, [1, 0, 0] try to change again should NOT work  (restriction_step = 2)
+ t == 3, [1, 0, 0] try to change again should NOT work  (restriction_step = 3)
+ t == 4, [1, 0, 0] try to change node. SHOULD WORK.
+ t == 5, [1, 1, 0] verify change.

#### test_Agent_test_MaxNumberActionnedLines():
This function creates an Agent that tests the restriction param: **"max_number_actionned_lines: 2"**.
The agent's plan is as follows:
+ t = 1, Agent switches off 3 lines, ==> should be rejected because of the restriction.
+ t = 2, check if all lines are ON.
+ t = 2, Agent switches off 2 lines, ==> should work.
+ t = 3, check if 2 lines are OFF and rest of lines are still ON
+ t = 3, Agent switches of 4 lines, ==> should be rejected because of the restriction.
+ t = 4, check if 2 lines are still OFF and rest of lines are still ON

#### test_Agent_test_MaxNumberActionnedNodes():
This function creates an Agent that tests the restriction param: **"max_number_actionned_substations: 2"**
The agent's plan is as follows:
+ t = 1, Agent changes the topology of 3 nodes, ==> should be rejected because of the restriction.
+ t = 2, check if substation configurations are identical to t == 1
+ t = 2, Agent changes the topology of 2 nodes, ==> should work.
+ t = 3, check if substation configurations of node [X, X] changed and the rest is identical to t == 1

#### test_Agent_test_BasicSubstationTopologyChange():
This function creates an Agent that tests all the Topological changes of all the Substations.
The agent changes all the possible connections on a given node, and checks from Observations that it did occur.
We check this way all changes, PRODS, LOADS, OR, EX
This test has a specific folder with **"n_timesteps_actionned_node_reactionable = 0"**, in order to be able to change
nodes at each time steps

#### test_Agent_test_AdvancedSubstationTopologyChange():
This function creates an Agent that tests all the Topological changes of all the Substations **with back and
 forth intermediate steps**. For example, on a node with 3 elements, the sequence of tested topologies will be as follows:
 + [000]
 + [100]
 + [000]
 + [010]
 + [000]
 + [001]
 + [000]
 
 The test is similar to the test BasicSubstationTopologyChange but with intermediate "go back to default configuration" actions.

### ---------------------------------- test_core.py ----------------------------------

#### test_core_Agent_test_ObsToArrayAndBack():
This function creates an Agent that tests the transformation of the object Observation into a list ==> and back,
transformation from list of observation ==> to object Observation.
The agent's plan is as follows:
+ Get current object observation1. Create list_obs1 from observation1
+ Create object observation2 from list_obs1.
+ Create list_obs2 from observation2
+ Compare list_obs1 and list_obs2.

#### test_core_Agent_test_limitOfProdsLost():
This function creates an Agent that tests the config variable: **"max_number_prods_game_over: 1"**
The agent's plan is as follows:
+ t = 1, it disconnects 1 prod
+ t = 2, it disconnects second prod, ==> causes a Game Over
+ t = 3, it checks that obs.productions_nodes = [0, 0, 0, 0, 0], ie, that the game reset.
This function checks that the game ended because of TooManyProductionsCut.

#### test_core_Agent_test_limitOfLoadsLost():
This function creates an Agent that tests the config variable: **"max_number_loads_game_over: 1"**
The agent's plan is as follows:
+ t = 1, it disconnects 1 load
+ t = 2, it disconnects second load, ==> causes a Game Over
+ t = 3, it checks that obs.loads_nodes = [0, 0, ... , 0, 0], ie, that the game reset.
This function checks that the game ended because of TooManyConsumptionsCut.

#### test_core_Agent_test_InputProdValues():
This function creates an Agent that tests the correct loading of input Prod values.
The agent compares the input PROD values found in the chronics and internal observations for 3 steps.

#### test_core_Agent_test_InputLoadValues():
This function creates an Agent that tests the correct loading of input Load values.
This agent compares the input LOAD values found in the chronics and internal observations for 3 steps.

#### test_core_Agent_test_method_obs_are_prods_cut():
This function tests the method: observation.are_prods_cut.

#### test_core_Agent_test_method_obs_are_loads_cut():
This function tests the method: observation.are_loads_cut.

#### test_core_Agent_test_Loss_Error():
This function creates an Agent that compares the expected loss (from chronics) and real loss (from observation)
for the first 3 iterations.

#### test_core_Agent_test_SoftOverflowBreakLimit():
This function creates an Agent that checks variable: **"n_timesteps_consecutive_soft_overflow_breaks = 2"**,
with thermal limit = 300 for line 6.
The agent's plan is as follows:
+ at t = 9,  line's 6 flow in ampere > 300, 322
+ at t = 10, line's 6 flow in ampere > 300, 347
+ at t = 11, it is the third consecutive timestep so we should have a line that is CUT because of SOFT OVERFLOW

#### test_core_Agent_test_SoftOverflowIsBroken():
This function creates an Agent that checks variable: **"n_timesteps_soft_overflow_is_broken: 2"**
\- number of timesteps a soft overflow broken line is broken. It is a follow up test for SoftOverflowBreakLimit
+ at t = 9,  line's 6 flow in ampere > 300, 322
+ at t = 10, line's 6 flow in ampere > 300, 347
+ at t = 11, it is the third consecutive timestep so we should have a line that is CUT because of SOFT OVERFLOW
from this point, for 2 more steps we will try to set the line back up, and we should get Illegal Actions exception
until t = 13
+ at t = 12 down for 2 consecutive steps
+ at t = 13, 3 consecutive steps > n_timesteps_soft_overflow_is_broken: 2, so we should be able to reconnect
+ at t = 14, we check line is BACK ONLINE

#### test_core_Agent_test_NodeTopoChangePersistence():
This function creates an Agent that switches a nodes topology and checks for 9 steps that it is still the same.

#### test_core_Agent_test_LineChangePersistence():
This function creates an Agent that cuts a line and checks for 9 steps that it is still cut.

#### test_core_Agent_test_HardOverflowCoefTest():
This function creates an Agent that checks variable: **"hard_overflow_coefficient: 1.5"** which is % of line capacity
usage above which a line will break bc of hard overflow, check Agent description for more info.
The flow of line 6 for each 15 steps is =[244, 210, 223, 214, 214, 237, 244, 286, 322, 347, 381, 310, 303, 324, 275]
with the thermal limits of the line 6 being: 200, * (overflow_coeff) 1.5 = 300.
So at step 9, the flow value of line 6 being 322, the line should break and we should have 0.
we expected the result to be            = [244, 210, 223, 214, 214, 237, 244, 286, 0, 0, 0, 0]
by trying to switch the line back up at each step, for step >= 9, we make sur the variable 
**"n_timesteps_hard_overflow_is_broken: 2"**. The number of timesteps a hard overflow broken line is broken, works.
The agent's plan is as follows:
+ t = 9,  line's 6 flow in ampere > 300, 322
+ t == 9, line just broke so switching it back up doesnt work. So we should have illegal action
+ t == 10, n_timesteps_hard_overflow_is_broken: 2, so still illegal action,
+ t == 11, we switch, it works but flow > 300 so it breaks again
+ t == 12, broken consec timestep = 1
+ t == 13, broken consec timestep = 2,
+ t == 14, we can switch back up,
+ t == 15, since flow < 300, it didnt break, and we end up with all line that are ON.

Expected results are = 
[None, None, None, None, None, None, None, None, IllegalActionException(), IllegalActionException(), None,
 IllegalActionException(), IllegalActionException(), None, None]
 
 
### ---------------------------------- test_simulate.py ----------------------------------

#### test_simulate_Agent_test_SimulateThenAct():
This function creates an Agent that cuts a line and checks for 9 steps that it is still cut, while using the
function env.simulate.

#### test_simulate_Agent_test_RewardError():
Function to test if a reward is the same whether we simulate during our work or not.
This function creates a small_Agent_test_RewardError which works for 3 steps.
first instance ==> the agent simulates in addition to cutting a line and change a node's topology
second instance ==> the agent just cuts a line and changes a node's topology, without simulation.
then we compare the rewards. They must be equal.

#### test_simulate_Agent_exhaustive_test_RewardError():
Function to test if a reward is the same whether we simulate during our work or not.
This function creates first a CustomGreedyAgent that will simulate all possible changes, but do nothing in the end.
Then, second instance ==> A do nothing Agent.
Finally, we compare the rewards. They must be equal.

#### test_simulate_Agent_CustomGreedySearch():
This function creates an Agent that does all possible actions while simulating. It checks that WITHIN a timestep,
the Observation of all Consumptions(Observation.are_loads) are still the same.

#### test_simulate_Agent_test_AccurateConsumptionLoadings():
This function creates an Agent that tests the correct loading of input Load values and planned Load values.
This agent compares the input LOAD values found in the chronics and internal observations for 3 steps,
and the input LOAD values from _N_loads_p_planned, which are read by the function env.simulate. It also checks that 
they are in fact the t+1 values compared to current observations.

