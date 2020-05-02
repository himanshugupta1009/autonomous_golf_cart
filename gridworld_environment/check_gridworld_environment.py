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


# #traj-1
# cart_poses = [[8,30],[8,29],[7,29],[7,27],[7,24],[5,22],[4,19],[4,14],[4,10],[5,6],[6,2],[7,1]]

# #traj-2
# cart_poses = [[8,30], [ 8, 29], [ 7, 28],[ 7, 25],[ 6, 22],[ 4, 20],[ 4, 16],[ 4, 12],[ 4,  8],[5, 5],[6, 2],[7, 1]]

# #traj-3
cart_poses = [[8,30], [7, 29], [7, 26],[6, 24],[5, 21],[4, 19],[4, 16],[4, 14],[4, 13],[5, 12],[4, 11],[4, 10],[4, 9],[4, 8],[4, 7],[4, 6],[5, 5],[6, 4],[6, 2],[7, 1]]

# env.move_humans_to_goals(cart_poses)


env.move_cart_to_goal(cart_poses)

# env.step(2)

# env.temp_path_print()


#
# array([[ 8, 29],[ 7, 29]]),
#             array([[ 7, 28],[ 7, 27],[ 7, 26]]),
#             array([[ 7, 25],[ 7, 24],[ 6, 24]]),
#             array([[ 5, 24],[ 5, 23],[ 5, 22],[ 5, 21]]),
#             array([[ 4, 21],[ 4, 20],[ 4, 19]]),
#             array([[ 4, 18],[ 4, 17],[ 4, 16]]),
#             array([[ 4, 15],[ 4, 14]]),
#             array([[ 4, 13]]),
#             array([[ 5, 13],[ 5, 12]]),
#             array([[ 5, 11],[ 4, 11]]),
#             array([[ 4, 10]]),
#             array([[4, 9]]),
#             array([[4, 8]]),
#             array([[4, 7]]),
#             array([[4, 6]]),
#             array([[4, 5],[5, 5]]),
#             array([[5, 4],[6, 4]]),
#             array([[6, 3],[6, 2]]),
#             array([[7, 2],[7, 1]])



