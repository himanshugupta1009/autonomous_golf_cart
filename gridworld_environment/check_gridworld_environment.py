import gym
import gym_gridworlds
from gridworld_environment.gw_env import GridworldEnv
import time

env = GridworldEnv()

# 0 stay, 1 up, 2 down, 3 left, 4 right
A = env.action_space

print(A)

env.verbose = True

env._render()

env.move_humans_to_goals()

# env.step(2)

env.temp_path_print()








