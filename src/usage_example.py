__author__ = 'marvinler'
from src.env import RunEnv
import time
import os
import numpy as np

# Create an instance of the game using APIgame
env = RunEnv()

observation = env._get_obs()
done = False
epoch = 0
step = 0
total_reward = 0.
start = time.time()
# Plays all the scenarios, one at a time
while not done:
    print('Playing epoch', epoch, 'scenario', "%06d" % env.get_current_scenario_id(), '; total reward', total_reward)

    action = None
    # action = env.action_space.sample_line_switch()
    # print('Switching %d' % np.argmax(action))

    # Apply action on the grid and retrieve the resulting state, reward, done status and info
    observation, reward, done, info = env.step(action)
    t = 2000
    # if time.time()-start>t:
    #     time.sleep(10000)
    # env._render(close=time.time()-start>t)
    env._render()

    # if step > 100:
    #     break

    # Debugging purposes: check whether the model has made an illegal action
    if info is not None:
        print('Game over!', info)
        pass  # Here you can do things to debug your solution
    if done:
        epoch += 1
        observation = env.reset(restart=False)
        done = False

    total_reward += reward
    n_overflowed_lines = np.sum(observation.relative_thermal_limits >= 1.)
    step += 1

    # Saving rewards
    # res_folder = 'results/'
    # if not os.path.exists(res_folder):
    #     os.makedirs(res_folder)
    # current_filename = 'do_nothing_reward_epoch%d.txt' % epoch
    # with open(os.path.join(res_folder, current_filename), 'a') as f:
    #     f.write('%5f\n' % reward)
    # current_filename = 'do_nothing_noverflowed_epoch%d.txt' % epoch
    # with open(os.path.join(res_folder, current_filename), 'a') as f:
    #     f.write('%d\n' % n_overflowed_lines)

end = time.time()
print('time', end - start)
