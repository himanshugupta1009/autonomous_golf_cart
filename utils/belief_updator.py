def calc_distance(human, human_goals):
    distances = []
    for goal in human_goals:
        dist = ((human[0] - goal[0])**2 + (human[1]-goal[1])**2)**0.5
        distances.append(1/dist)
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
