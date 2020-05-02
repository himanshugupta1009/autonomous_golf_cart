def calc_distance(human, human_goals):
    distances = []
    for goal in human_goals:
        dist = ((human[0] - goal[0])**2 + (human[1]-goal[1])**2)**0.5
        if dist != 0:
            distances.append(1/dist)
        else:
            distances.append(0)
    return distances


def belief_update(human_goals, human_curr_poses, old_belief):
    new_belief = []
    for i, human in enumerate(human_curr_poses):
        belief = old_belief[i]
        human_to_goal_distances = calc_distance(human, human_goals)

        temp_array = [belief[i]*human_to_goal_distances[i] for i in range(len(belief))]
        prob_sum = sum(temp_array)

        temp_array = [temp_array[i]/prob_sum for i in range(len(temp_array))]

        new_belief.append(temp_array)

    return new_belief


def goal_update_based_on_new_belief(new_belief, human_curr_poses, human_goals):
    human_goal_pair_list= []
    for i in range(len(human_goals)):
        new_b = new_belief[i]
        idx = new_b.index(max(new_b))
        human_goal_pair_list.append(human_curr_poses[i][0])
        human_goal_pair_list.append(human_curr_poses[i][1])
        human_goal_pair_list.append(human_goals[idx][0])
        human_goal_pair_list.append(human_goals[idx][1])
    return human_goal_pair_list


def flatten_2d_list(d_list):
    flat_list = []
    for item in d_list:
        for elem in item:
            flat_list.append(elem)
    return flat_list
