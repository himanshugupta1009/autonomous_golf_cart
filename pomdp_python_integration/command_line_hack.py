import subprocess


# returns the action need to be taken
def run_julia_and_get_output_from_command_line():

    action = 0
    process = subprocess.Popen(['julia', 'speed_planner_updated.jl'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    output, error = process.communicate()
    output = output.decode('utf-8')
    error = error.decode('utf-8')

    error_array = error.split()
    if len(error_array) > 0:
        print('Julia ERROR')
    output_array = output.split()

    if len(output_array) > 0:
        action = output_array[-1]


    return action

# run_julia_and_get_output_from_command_line()


# write the new data to the file
def update_file_with_new_data(cart_state_data, pedestrians_list_data, belief_data, astar_path_data):
    fi = open('speed_planner_updated.jl','r')

    lines = fi.readlines()
    fi.close()

    lines[429] = cart_state_data+'\n'
    lines[433] = pedestrians_list_data+'\n'
    lines[437] = belief_data+'\n'
    lines[439] = astar_path_data+'\n'

    fo = open('speed_planner_updated.jl','w')

    for l in lines:
        fo.write(l)
    fo.close()


# update_file_with_new_data('dfdfd', 'dfsdfdsath_data', 'dgya4dgshfgds', 'ghe4tghthjtrjrrg')


