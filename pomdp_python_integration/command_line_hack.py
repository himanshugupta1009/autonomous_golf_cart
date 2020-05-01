import subprocess


# returns the action need to be taken
def run_julia_and_get_output_from_command_line():
    process = subprocess.Popen(['julia', 'speed_planner_updated.jl'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    output, error = process.communicate()
    output = output.decode('utf-8')
    error = error.decode('utf-8')

    output_array = output.split()
    return output_array[-1]

# run_julia_and_get_output_from_command_line()


# write the new data to the file
def update_file_with_new_data(cart_state_data, pedestrians_list_data, belief_data, astar_path_data):
    fi = open('speed_planner_updated.jl','r')

    lines = fi.readlines()
    fi.close()

    lines[426] = cart_state_data+'\n'
    lines[430] = pedestrians_list_data+'\n'
    lines[434] = belief_data+'\n'
    lines[436] = astar_path_data+'\n'

    fo = open('speed_planner_updated.jl','w')

    for l in lines:
        fo.write(l)
    fo.close()


# update_file_with_new_data('dfdfd', 'dfsdfdsath_data', 'dgya4dgshfgds', 'ghe4tghthjtrjrrg')


