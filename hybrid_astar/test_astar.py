import cv2
import numpy as np
import hybrid_astar.pyastar as pyastar
from time import time
import sys
import os
from os.path import basename, join, splitext
import matplotlib.pyplot as plt


def _read_grid_map(grid_map_path):
    with open(grid_map_path, 'r') as f:
        grid_map = f.readlines()
    grid_map_array = np.array(
        list(map(
            lambda x: list(map(
                lambda y: int(y),
                x.split(' ')
            )),
            grid_map
        ))
    )
    return grid_map_array


def convert_grid_for_astar():
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    grid_map_path = os.path.join(this_file_path, 'environment.txt')
    start_grid_map = _read_grid_map(grid_map_path)

    ## convert 0's to inf
    start_grid_map[start_grid_map == 1] = 99
    start_grid_map[start_grid_map == 0] = 1
    return start_grid_map


grid_map = convert_grid_for_astar()


def main():

    grid = convert_grid_for_astar()
    grid = np.asarray(grid, dtype=np.float32)

    start = np.asarray([8, 30], dtype=np.int)
    end = np.asarray([7, 1], dtype=np.int)

    # # start is the first white block in the top row
    # start_j, = np.where(grid[0, :] == 1)
    # start = np.array([0, start_j[0]])
    #
    # # end is the first white block in the final column
    # end_i, = np.where(grid[:, -1] == 1)
    # end = np.array([end_i[0], grid.shape[0] - 1])

    t0 = time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar.astar_path(grid, start, end, allow_diagonal=False)
    dur = time() - t0

    # if path.shape[0] > 0:
    #     print('found path of length %d in %.6fs' % (path.shape[0], dur))
    #     maze[path[:, 0], path[:, 1]] = (0, 0, 255)
    #
    #     print('plotting path to %s' % (OUTP_FPATH))
    #     cv2.imwrite(OUTP_FPATH, maze)
    # else:
    #     print('no path found')

    print(path)
    print('done')


if __name__ == '__main__':
    main()
