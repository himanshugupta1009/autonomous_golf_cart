# reward model

def collision_reward(robot_pose, pedestrian_poses, coll_threshold):
    coll_r = 0





# collision reward
function
collision_reward(sp, coll_threshold)
total_reward = 0
cart_pose_x = sp.cart.x
cart_pose_y = sp.cart.y
for human in sp.pedestrians
    dist = ((human.x - cart_pose_x) ^ 2 + (human.y - cart_pose_y) ^ 2) ^ 0.5
    if dist < coll_threshold
        total_reward = total_reward + m.collision_reward
    end
end
return total_reward
end

# # goal reward
# function
# goal_reward(sp, s, goal_state_reward)
# total_reward = -1
# cart_new_pose_x = sp.cart.x
# cart_new_pose_y = sp.cart.y
#
# cart_goal = m.cart_goal_position
# new_dist = ((cart_goal.x - cart_new_pose_x) ^ 2 + (cart_goal.y - cart_new_pose_y) ^ 2) ^ 0.5
#
# cart_old_pose_x = s.cart.x
# cart_old_pose_y = s.cart.y
# old_dist = ((cart_goal.x - cart_old_pose_x) ^ 2 + (cart_goal.y - cart_old_pose_y) ^ 2) ^ 0.5
#
# if new_dist < old_dist & & new_dist != 0
#     total_reward = goal_state_reward / new_dist
# elseif
# new_dist == 0
# total_reward = goal_state_reward
# end
# return total_reward
# end
#
# # speed reward
# function
# speed_reward(sp, max_speed)
# return (sp.cart.v - max_speed) / max_speed
# end
#
# r = collision_reward(sp, m.collision_threshold) + goal_reward(sp, s, m.goal_reward) + speed_reward(sp, m.max_cart_speed)