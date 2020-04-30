from utils.belief_updator import belief_update

human_goals = [[1,1], [1,30], [14,30], [14,1]]
human_curr_poses = [[4,4], [3,21], [11,23], [12,3]]
old_belief = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
new_belief = belief_update(human_goals, human_curr_poses, old_belief)

print(new_belief)
