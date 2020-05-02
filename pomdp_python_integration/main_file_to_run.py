from gridworld_environment.gw_env import GridworldEnv
from utils.belief_updator import belief_update, goal_update_based_on_new_belief, flatten_2d_list
from pomdp_python_integration.command_line_hack import update_file_with_new_data, run_julia_and_get_output_from_command_line
from pomdp_python_integration.reward_functions import get_total_reward
import pdb

env = GridworldEnv()

actual_path = []
curr_idx_in_path = 0

total_reward = 0

h_time_step = 0

# initial belief
initial_belief = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]

initial_vel = 2

# move humans one time step
humans_list = env.move_humans_one_time_step(h_time_step)
h_time_step += 1
# print(humans_list[3].get_state())

human_curr_poses = [humans_list[i].get_state() for i in range(len(humans_list))]
# print(human_curr_poses)
# human_goals = [humans_list[i].get_goal() for i in range(len(humans_list))]
# print(human_goals)
human_goals = [[1,1], [1,30], [14,30], [14,1]]


# and update belief
new_belief = belief_update(human_goals, human_curr_poses, initial_belief)
# print(new_belief)

# update new goald for pomdp based on new belief
new_human_goal_pair_list = goal_update_based_on_new_belief(new_belief, human_curr_poses, human_goals)
# print(new_human_goal_pair_list)

# loop until Cart reaches it's goal at [7,1]
robot_curr_state = env._get_robot_state()
goal_state = [7,1]
print(robot_curr_state)
robot_prev_state = robot_curr_state

curr_vel = initial_vel

i = 0
#robot_curr_state != goal_state and
# pdb.set_trace()
while robot_curr_state != goal_state:
    i += 1

    # get A star path
    astar_path = env.plan_astar_path_at_current_time()
    # print(astar_path)
    print('got A-star')

    # solve pomdp and get action
    # prepare pomdp inputs:
    #       cart_state_data [8,30,0,2],
    cart_state_data = [robot_curr_state[0], robot_curr_state[1], 0, curr_vel]
    cart_state_data_string = 'cart_start_state_list = ' + str(cart_state_data)
    #       pedestrians_list_data [4,3,14,30,3,22,14,1,11,22,1,30,12,4,1,1],
    pedestrians_list_data = new_human_goal_pair_list
    pedestrians_list_data_string = 'pedestrians_list = ' + str(pedestrians_list_data)
    #       belief_data [0.15,0.3,0.5,0.05,0.1,0.3,0.1,0.5,0.5,0.25,0.1,0.15,0.05,0.5,0.35,0.1],
    new_belief_flat = flatten_2d_list(new_belief)
    new_belief = [ round(elem, 2) for elem in new_belief_flat]
    belief_data_string = 'initial_human_goal_distribution_list = ' + str(new_belief_flat)
    #       astar_path_data [8, 30,7, 30,7, 29,7, 28,7, 27,7, 26,7, 25, 7, 24,7, 23,7, 22,6, 22,6, 21,5, 21,5, 20,5, 19, 5, 18,5, 17,5, 16,5, 15,5, 14,5, 13,5, 12,5, 11,5, 10,5, 9,5, 8,5, 7,5, 6,5, 5,5, 4,5, 3,5, 2,5, 1,6, 1,7, 1]
    astar_path_flat = flatten_2d_list(astar_path)
    astar_path_string = 'given_astar_path = ' + str(astar_path_flat)

    # update the data into julia file
    update_file_with_new_data(cart_state_data_string, pedestrians_list_data_string, belief_data_string, astar_path_string)

    # execute and get result
    result_action = run_julia_and_get_output_from_command_line()
    print('got result: ', result_action)

    # update  velocity based on action
    curr_vel = curr_vel + int(result_action)

    robot_prev_state = robot_curr_state
    # update global path
    if len(astar_path) > curr_vel:
        actual_path.append(astar_path[1 : curr_vel+1])
        # curr_idx_in_path = curr_idx_in_path + curr_vel
        env.robot_state = astar_path[curr_vel]
        robot_curr_state = astar_path[curr_vel]
    else:
        actual_path.append(astar_path[0 : len(astar_path)])
        env.robot_state = astar_path[-1]
        robot_curr_state = astar_path[-1]



    # move humans one time step
    humans_list = env.move_humans_one_time_step(h_time_step)
    h_time_step += 1

    print('got human moving')

    # and update belief
    new_belief = belief_update(human_goals, human_curr_poses, initial_belief)
    # print(new_belief)

    # update new goald for pomdp based on new belief
    new_human_goal_pair_list = goal_update_based_on_new_belief(new_belief, human_curr_poses, human_goals)
    # print(new_human_goal_pair_list)
    print('got belief update')

    # print(type(goal_state))
    # print(type(robot_curr_state))
    # print(robot_curr_state)
    robot_curr_state = list(robot_curr_state)

    # robot_pose, pedestrian_poses, coll_threshold, robot_prev_pose, robot_goal, robot_speed
    total_reward += get_total_reward(robot_curr_state, human_curr_poses, 3, robot_prev_state, goal_state, curr_vel)
    print('\n #################### \n')


print(actual_path)
print(total_reward)




# #Traj-1
# [[ 8, 29]]), array([[ 7, 29]]), array([[ 7, 28],
#        [ 7, 27]]), array([[ 7, 26],
#        [ 7, 25],
#        [ 7, 24]]), array([[ 6, 24],
#        [ 6, 23],
#        [ 6, 22],
#        [ 5, 22]]), array([[ 5, 21],
#        [ 4, 21],
#        [ 4, 20],
#        [ 4, 19]]), array([[ 4, 18],
#        [ 4, 17],
#        [ 4, 16],
#        [ 4, 15],
#        [ 4, 14]]), array([[ 4, 13],
#        [ 4, 12],
#        [ 4, 11],
#        [ 4, 10]]), array([[4, 9],
#        [4, 8],
#        [5, 8],
#        [5, 7],
#        [5, 6]]), array([[5, 5],
#        [5, 4],
#        [6, 4],
#        [6, 3],
#        [6, 2]]), array([[6, 2],
#        [7, 2],
#        [7, 1]])]

# [array([[ 8, 29]]), array([], shape=(0, 2), dtype=int64), array([[ 7, 29]]), array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64), array([[ 7, 28]]), array([[ 7, 27],
#        [ 7, 26]]), array([[ 7, 25],
#        [ 7, 24]]), array([[ 6, 24],
#        [ 6, 23],
#        [ 6, 22]]), array([[ 5, 22],
#        [ 5, 21],
#        [ 4, 21],
#        [ 4, 20]]), array([[ 4, 19],
#        [ 4, 18],
#        [ 4, 17],
#        [ 4, 16]]), array([[ 4, 15],
#        [ 4, 14],
#        [ 4, 13]]), array([[ 4, 12],
#        [ 4, 11]]), array([[ 4, 10]]), array([[4, 9]]), array([[4, 8]]), array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64)]




#
# # Traj-2
# [array([[ 8, 29]]), array([[ 7, 29],
#        [ 7, 28]]), array([[ 7, 27],
#        [ 7, 26],
#        [ 7, 25]]), array([[ 7, 24],
#        [ 6, 24],
#        [ 6, 23],
#        [ 6, 22]]), array([[ 5, 22],
#        [ 5, 21],
#        [ 4, 21],
#        [ 4, 20]]), array([[ 4, 19],
#        [ 4, 18],
#        [ 4, 17],
#        [ 4, 16]]), array([[ 4, 15],
#        [ 4, 14],
#        [ 4, 13],
#        [ 4, 12]]), array([[ 4, 11],
#        [ 4, 10],
#        [ 4,  9],
#        [ 4,  8]]), array([[4, 7],
#        [4, 6],
#        [4, 5],
#        [5, 5]]), array([[5, 4],
#        [6, 4],
#        [6, 3],
#        [6, 2]]), array([[6, 2],
#        [7, 2],
#        [7, 1]])]



#Traj-3
# [array([[ 8, 29],
#        [ 7, 29]]), array([[ 7, 28],
#        [ 7, 27],
#        [ 7, 26]]), array([[ 7, 25],
#        [ 7, 24],
#        [ 6, 24]]), array([[ 5, 24],
#        [ 5, 23],
#        [ 5, 22],
#        [ 5, 21]]), array([[ 4, 21],
#        [ 4, 20],
#        [ 4, 19]]), array([[ 4, 18],
#        [ 4, 17],
#        [ 4, 16]]), array([[ 4, 15],
#        [ 4, 14]]), array([[ 4, 13]]), array([[ 5, 13],
#        [ 5, 12]]), array([[ 5, 11],
#        [ 4, 11]]), array([[ 4, 10]]), array([[4, 9]]), array([[4, 8]]), array([[4, 7]]), array([[4, 6]]), array([[4, 5],
#        [5, 5]]), array([[5, 4],
#        [6, 4]]), array([[6, 3],
#        [6, 2]]), array([[7, 2],
#        [7, 1]])]
