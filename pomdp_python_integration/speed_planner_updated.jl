using POMDPSimulators#nofixdeps
using POMDPs
using POMDPModels
using ARDESPOT
using POMDPPolicies
using POMDPModelTools
using BasicPOMCP


using ParticleFilters
using Distributions: Normal
using Random
import POMDPs: initialstate_distribution, actions, gen, discount, isterminal
Random.seed!(1);


sleep(10)

struct goal_location
    x:: Int64
    y:: Int64
end

struct pedestrian_state
    x:: Int64
    y:: Int64
    goal:: goal_location
end

struct cart_state
    x:: Int64
    y:: Int64
    theta:: Int64
    v:: Int64
end

struct golfcart_observations
    observed_human_positions:: Array{goal_location}
end

struct SP_POMDP_state
    cart:: cart_state
    pedestrians:: Array{pedestrian_state}
    pedestrian_goals:: Array{goal_location}
    path_covered_index:: Int64
end

struct human_goal_probability
    distribution::Array{Float64}
end


mutable struct Speed_Planner_POMDP <: POMDPs.POMDP{SP_POMDP_state,Int,golfcart_observations}
    discount_factor::Float64
    step_size::Int64
    collision_threshold::Float64
    collision_reward::Int64
    goal_reward::Int64
    max_cart_speed::Int64
    cart_goal_position::goal_location
    starting_cart_state::cart_state
    starting_human_states::Array{pedestrian_state}
    fixed_goal_locations::Array{goal_location}
    human_goals_prob_distribution::Array{human_goal_probability}
    astar_path::Array{Tuple{Int64,Int64}}
    start_path_index::Int64
end


function POMDPs.gen(m::Speed_Planner_POMDP, s, a, rng)

    # transition model

    function calculate_theta(current_state, previous_state)
        theta = 0
        x_diff = current_state[1] - previous_state[1]
        y_diff = current_state[2] - previous_state[2]
        if x_diff != 0
            if x_diff < 0
                theta = 90
            else
                theta = 270
            end
        end
        if y_diff != 0
            if y_diff < 0
                theta = 0
            else
                theta = 180
            end
        end
        return theta
    end

    function update_human_state(human, human_goals, rng)
        goal = human.goal
        human_fixed_goals = copy(human_goals)
        deleteat!(human_fixed_goals, findall(x -> x==goal, human_fixed_goals)[1])
        rand_num = rand(rng)
        function move_human_towards_goal(human, goal)
            temp_human_x = human.x
            temp_human_y = human.y
            if temp_human_x < goal.x
                temp_human_x = temp_human_x + 1
            elseif temp_human_x > goal.x
                temp_human_x = temp_human_x - 1
            end

            if temp_human_y < goal.y
                temp_human_y = temp_human_y + 1
            elseif temp_human_y > goal.y
                temp_human_y = temp_human_y - 1
            end
            temp_human_x = clamp(temp_human_x,1,14)
            temp_human_y = clamp(temp_human_y,1,30)
            return pedestrian_state(temp_human_x, temp_human_y, goal), goal_location(temp_human_x, temp_human_y)
        end
        if rand_num <= 0.7
            # move human towards goal
            new_human,observed_location = move_human_towards_goal(human, goal)
        elseif rand_num > 0.7 && rand_num <= 0.8
            new_human,observed_location = move_human_towards_goal(human, human_fixed_goals[1])
        elseif rand_num > 0.8 && rand_num <= 0.9
            new_human,observed_location = move_human_towards_goal(human, human_fixed_goals[2])
        elseif rand_num > 0.9
            new_human,observed_location = move_human_towards_goal(human, human_fixed_goals[3])
        end
        return new_human,observed_location
    end

    new_pedestrians = pedestrian_state[]
    observed_positions = goal_location[]

    # action 0
    if a == 0
        # kart state +2 steps based on path
        # x = new state in path's X
        # y = new state in path's Y
        # theta = new states - one previous state {if change in x or change in y}
        # v = v
        new_v = s.cart.v
        new_position = m.astar_path[clamp(s.path_covered_index + new_v,1,length(m.astar_path))]
        new_theta = calculate_theta(new_position, m.astar_path[clamp(s.path_covered_index + new_v - 1,1,length(m.astar_path))])
        cart_new_state = cart_state(new_position[1], new_position[2], new_theta, new_v)

        # pedestrians state +1 step in their path for all pedestrians
        # change x
        # change y
        for human in s.pedestrians
            new_human,observed_location = update_human_state(human, s.pedestrian_goals, rng)
            push!(new_pedestrians, new_human)
            push!(observed_positions, observed_location)
        end
        # path {need to change now/later based on A* from kart's current position to goal}
        new_path_index = s.path_covered_index + new_v

    # action 1
    elseif a == 1
        # kart state +3 steps based on path
        # x = new state in path's X
        # y = new state in path's Y
        # theta = new states - one previous state {if change in x or change in y}
        # v = v +1
        new_v = s.cart.v + a
        if(new_v>m.max_cart_speed)
            new_v = m.max_cart_speed
        end
        new_position = m.astar_path[clamp(s.path_covered_index + new_v,1,length(m.astar_path))]
        new_theta = calculate_theta(new_position, m.astar_path[clamp(s.path_covered_index + new_v - 1,1,length(m.astar_path))])
        cart_new_state = cart_state(new_position[1], new_position[2], new_theta, new_v)

        # pedestrians state +1 step in their path for all pedestrians
        # change x
        # change y
        for human in s.pedestrians
            new_human,observed_location = update_human_state(human, s.pedestrian_goals, rng)
            push!(new_pedestrians, new_human)
            push!(observed_positions, observed_location)
        end

        # path {need to change now/later based on A* from kart's current position to goal}
        new_path_index = s.path_covered_index + new_v

    # action -1
    elseif a == -1
        # kart state +1 steps based on path
        # x = new state in path's X
        # y = new state in path's Y
        # theta = new states - one previous state {if change in x or change in y}
        # v = v -1
        new_v = s.cart.v + a
        if new_v < 0
            new_v = 0
        end
        new_position = m.astar_path[clamp(s.path_covered_index + new_v,1,length(m.astar_path))]
        new_theta = calculate_theta(new_position, m.astar_path[clamp(s.path_covered_index + new_v - 1,1,length(m.astar_path))])
        cart_new_state = cart_state(new_position[1], new_position[2], new_theta, new_v)

        # pedestrians state +1 step in their path for all pedestrians
        # change x
        # change y
        for human in s.pedestrians
            new_human,observed_location = update_human_state(human, s.pedestrian_goals, rng)
            push!(new_pedestrians, new_human)
            push!(observed_positions, observed_location)
        end

        # path {need to change now/later based on A* from kart's current position to goal}
        new_path_index = s.path_covered_index + new_v

    end

    # update the state object
    sp = SP_POMDP_state(cart_new_state, new_pedestrians, s.pedestrian_goals, new_path_index)

    # observation model
    o = golfcart_observations(observed_positions)

    # reward model

    # collision reward
    function collision_reward(sp, coll_threshold)
        total_reward = 0
        cart_pose_x = sp.cart.x
        cart_pose_y = sp.cart.y
        for human in sp.pedestrians
            dist = ((human.x - cart_pose_x)^2 + (human.y - cart_pose_y)^2)^0.5
            if dist < coll_threshold
                total_reward = total_reward + m.collision_reward
            end
        end
        return total_reward
    end

    # goal reward
    function goal_reward(sp, s, goal_state_reward)
        total_reward = -1
        cart_new_pose_x = sp.cart.x
        cart_new_pose_y = sp.cart.y

        cart_goal = m.cart_goal_position
        new_dist = ((cart_goal.x - cart_new_pose_x)^2 + (cart_goal.y - cart_new_pose_y)^2)^0.5

        cart_old_pose_x = s.cart.x
        cart_old_pose_y = s.cart.y
        old_dist = ((cart_goal.x - cart_old_pose_x)^2 + (cart_goal.y - cart_old_pose_y)^2)^0.5

        if new_dist < old_dist && new_dist != 0
            total_reward = goal_state_reward/new_dist
        elseif new_dist == 0
            total_reward = goal_state_reward
        end
        return total_reward
    end

    # speed reward
    function speed_reward(sp, max_speed)
        return (sp.cart.v - max_speed)/max_speed
    end

    r = collision_reward(sp, m.collision_threshold) + goal_reward(sp, s, m.goal_reward) + speed_reward(sp, m.max_cart_speed)
    #@show("Action is ", a)
    # create and return a NamedTuple
    return (sp=sp, o=o, r=r)

end

#Discount and terminal state function

function isgoalstate(s,cart_goal)
    cart_x = s.cart.x
    cart_y = s.cart.y
    if(cart_goal.x == cart_x && cart_goal.y == cart_y)
        return true
    end
    for human in s.pedestrians
        if(cart_x == human.x && cart_y == human.y)
            #display("Collision")
            return true
        end
    end
    return false
end

discount(p::Speed_Planner_POMDP) = p.discount_factor
isterminal(p::Speed_Planner_POMDP, s::SP_POMDP_state) = isgoalstate(s,p.cart_goal_position);

#Action Space for the POMDP
actions(::Speed_Planner_POMDP) = [-1, 0, 1] # Decelerate Maintain Accelerate

#Initial state distribution for the POMDP

function initialstate_distribution(m::Speed_Planner_POMDP)
    initial_cart_state = m.starting_cart_state
    all_human_goal_locations = m.fixed_goal_locations
    initial_human_states = m.starting_human_states
    initial_path_start_index = m.start_path_index
    initial_human_goal_probability = m.human_goals_prob_distribution
    num_goals = length(all_human_goal_locations)

    all_256_possible_states = []
    all_256_probability_values = Float64[]

    for goal_human1_index in (1:num_goals)
        for goal_human2_index in (1:num_goals)
            for goal_human3_index in (1:num_goals)
                for goal_human4_index in (1:num_goals)
                    sampled_human1_state = pedestrian_state(initial_human_states[1].x,initial_human_states[1].y,all_human_goal_locations[goal_human1_index])
                    sampled_human2_state = pedestrian_state(initial_human_states[2].x,initial_human_states[2].y,all_human_goal_locations[goal_human2_index])
                    sampled_human3_state = pedestrian_state(initial_human_states[3].x,initial_human_states[3].y,all_human_goal_locations[goal_human3_index])
                    sampled_human4_state = pedestrian_state(initial_human_states[4].x,initial_human_states[4].y,all_human_goal_locations[goal_human4_index])
                    sampled_humans = [sampled_human1_state, sampled_human2_state, sampled_human3_state, sampled_human4_state]
                    generated_state = SP_POMDP_state(initial_cart_state,sampled_humans,all_human_goal_locations,initial_path_start_index)
                    push!(all_256_possible_states,generated_state)

                    human1_prob = initial_human_goal_probability[1].distribution[goal_human1_index]
                    human2_prob = initial_human_goal_probability[2].distribution[goal_human2_index]
                    human3_prob = initial_human_goal_probability[3].distribution[goal_human3_index]
                    human4_prob = initial_human_goal_probability[4].distribution[goal_human4_index]
                    probability_for_generated_state =  human1_prob*human2_prob*human3_prob*human4_prob
                    push!(all_256_probability_values,probability_for_generated_state)
                end
            end
        end
    end
    d = SparseCat(all_256_possible_states, all_256_probability_values)
    #@show(eltype(d.probs))
    return d
end

#Upper bound for DESPOT

function golf_cart_upper_bound(m, b)
    value_sum = 0.0
    function is_collision_state(s)
        is_collision_flag = false
        for human in s.pedestrians
            dist = ((human.x - s.cart.x)^2 + (human.y - s.cart.y)^2)^0.5
            if dist < m.collision_threshold
                is_collision_flag = true
            end
        end
        return is_collision_flag
    end
    function time_to_goal(s)
        curr_vel = m.max_cart_speed
        remaining_path_length = clamp(length(m.astar_path) - s.path_covered_index,1,100)
        time_needed_at_curr_vel = ceil(remaining_path_length/curr_vel)
        return time_needed_at_curr_vel
    end
    for (s, w) in weighted_particles(b)
        if(s.cart.x == 7 && s.cart.y==1)
            value_sum += w*m.goal_reward
        elseif (is_collision_state(s))
            value_sum += w*m.collision_reward*(-1)
        else
            value_sum += w*((discount(m)^time_to_goal(s))*m.goal_reward)
            #value_sum += w*m.goal_reward
        end
    end
    #@show(value_sum)
    return (value_sum)/weight_sum(b)
end

function get_best_possible_action(cart_start_state_list, cart_goal_position, pedestrians_list, possible_goal_positions, initial_human_goal_distribution_list, given_astar_path)

    #@show(typeof(cart_start_state_list))
    #@show(typeof(cart_goal_position))
    #@show(typeof(pedestrians_list))
    #@show(typeof(possible_goal_positions))
    #@show(typeof(initial_human_goal_distribution_list))
    #@show(typeof(given_astar_path))


    cart_start_state = cart_state(cart_start_state_list[1],cart_start_state_list[2],
                                cart_start_state_list[3],cart_start_state_list[4])

    cart_goal = goal_location(cart_goal_position[1],cart_goal_position[2])

    ps1 = pedestrian_state(pedestrians_list[1],pedestrians_list[2],
        goal_location(pedestrians_list[3],pedestrians_list[4]))
    ps2 = pedestrian_state(pedestrians_list[5],pedestrians_list[6],
        goal_location(pedestrians_list[7],pedestrians_list[8]))
    ps3 = pedestrian_state(pedestrians_list[9],pedestrians_list[10],
        goal_location(pedestrians_list[11],pedestrians_list[12]))
    ps4 = pedestrian_state(pedestrians_list[13],pedestrians_list[14],
        goal_location(pedestrians_list[15],pedestrians_list[16]))
    human_state_start_list = [ps1,ps2,ps3,ps4]


    g1 =  goal_location(possible_goal_positions[1],possible_goal_positions[2])
    g2 =  goal_location(possible_goal_positions[3],possible_goal_positions[4])
    g3 =  goal_location(possible_goal_positions[5],possible_goal_positions[6])
    g4 =  goal_location(possible_goal_positions[7],possible_goal_positions[8])
    all_goals_list = [g1,g2,g3,g4]


    h1_dis = human_goal_probability([initial_human_goal_distribution_list[1],initial_human_goal_distribution_list[2],initial_human_goal_distribution_list[3],initial_human_goal_distribution_list[4]])
    h2_dis = human_goal_probability([initial_human_goal_distribution_list[5],initial_human_goal_distribution_list[6],initial_human_goal_distribution_list[7],initial_human_goal_distribution_list[8]])
    h3_dis = human_goal_probability([initial_human_goal_distribution_list[9],initial_human_goal_distribution_list[10],initial_human_goal_distribution_list[11],initial_human_goal_distribution_list[12]])
    h4_dis = human_goal_probability([initial_human_goal_distribution_list[13],initial_human_goal_distribution_list[14],initial_human_goal_distribution_list[15],initial_human_goal_distribution_list[16]])
    human_dis_list = [h1_dis,h2_dis,h3_dis,h4_dis]

    robot_path = Tuple{Int64,Int64}[]

    modified_astar_path = []
    path_len = length(given_astar_path)
    for i in (1:2:path_len)
        push!(modified_astar_path,[given_astar_path[i],given_astar_path[i+1]])
    end

    for position in modified_astar_path
        push!(robot_path,(position[1],position[2]))
    end

    golfcart_pomdp() = Speed_Planner_POMDP(0.9,1,3,-100,100,5,cart_goal,cart_start_state,
        human_state_start_list,all_goals_list,human_dis_list,robot_path,1)

    m = golfcart_pomdp()
    #solver = POMCPSolver(tree_queries=1000, c=10)
    #solver = DESPOTSolver(bounds=(DefaultPolicyLB(RandomSolver()), golf_cart_upper_bound))
    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->1)),golf_cart_upper_bound, check_terminal=true, consistency_fix_thresh=10.0),D=100)
    planner = solve(solver, m)
    b = initialstate_distribution(m);
    #@show("check works till here")
    a = action(planner, b)
    return a
end

cart_start_state_list = [4, 7, 0, 4]

cart_goal_position = [7,1]

pedestrians_list = [11, 2, 14, 1, 8, 25, 14, 30, 8, 17, 14, 30, 11, 11, 14, 1]

possible_goal_positions = [1,1,1,30,14,30,14,1]

initial_human_goal_distribution_list = [0.20521470430751101, 0.06936542165149051, 0.07323734121250129, 0.6521825328284971, 0.1231879351736534, 0.3580076646415972, 0.3943149716252608, 0.12448942855948873, 0.22600902037117318, 0.267329844477873, 0.27567587519020825, 0.23098525996074556, 0.26677690123061865, 0.17571658682413618, 0.19613827862162636, 0.3613682333236189]

given_astar_path = [4, 7, 4, 6, 4, 5, 5, 5, 5, 4, 6, 4, 6, 3, 6, 2, 7, 2, 7, 1]


result = get_best_possible_action(cart_start_state_list, cart_goal_position, pedestrians_list, possible_goal_positions, initial_human_goal_distribution_list, given_astar_path)

@show(result)

