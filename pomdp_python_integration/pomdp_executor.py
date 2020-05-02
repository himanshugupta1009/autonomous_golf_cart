from julia import Julia
jul = Julia(compiled_modules=False)

# func = jul.include("speed_planner.jl")

jul.eval('include("speed_planner.jl")')


get_best_possible_action = jul.eval('get_best_possible_action')

print(get_best_possible_action)


cart_start_state_list = [8,30,0,2]

cart_goal_position = [7,1]

pedestrians_list = [4,3,14,30,3,22,14,1,11,22,1,30,12,4,1,1]

possible_goal_positions = [1,1,1,30,14,30,14,1]

initial_human_goal_distribution_list = [0.15,0.3,0.5,0.05,0.1,0.3,0.1,0.5,
    0.5,0.25,0.1,0.15,0.05,0.5,0.35,0.1]

given_astar_path = [8, 30,7, 30,7, 29,7, 28,7, 27,7, 26,7, 25,
            7, 24,7, 23,7, 22,6, 22,6, 21,5, 21,5, 20,5, 19,
            5, 18,5, 17,5, 16,5, 15,5, 14,5, 13,5, 12,5, 11,
            5, 10,5, 9,5, 8,5, 7,5, 6,5, 5,5, 4,5, 3,5, 2,5, 1,6, 1,7, 1]


result = get_best_possible_action(cart_start_state_list, cart_goal_position, pedestrians_list,
                            possible_goal_positions, initial_human_goal_distribution_list, given_astar_path)

print(result)


# get_best_possible_action


# typeof(cart_start_state_list)                     = Array{Int64,1}
# typeof(cart_goal_position)                        = Array{Int64,1}
# typeof(pedestrians_list)                          = Array{Any,2}
# typeof(possible_goal_positions)                   = Array{Int64,2}
# typeof(initial_human_goal_distribution_list)      = Array{Float64,2}
# typeof(given_astar_path)                          = Array{Tuple{Int64,Int64},1}


# action_to_be_taken = get_best_possible_action(cart_start_state_list, cart_goal_position, pedestrians_list,
# possible_goal_positions, initial_human_goal_distribution_list, given_astar_path);
# @show(action_to_be_taken)