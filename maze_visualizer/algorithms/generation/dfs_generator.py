# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:01:40 2020
@author: Raul Ortega Ochoa
"""
import pygame, argparse, csv, time
import argparse
import numpy as np
from time import sleep
from numpy.random import randint
from pathlib import Path
from maze_visualizer.core.maze import Maze




BASE_DIR = Path(__file__).resolve().parents[2]   # .../ai-maze-python
PKG_DIR = BASE_DIR / "maze_visualizer"
DATA_DIR = PKG_DIR / "data"
MAZES_DIR = DATA_DIR / "mazes"



def is_in_map(pos, grid_dim):
    """
    pos: (x, y) index in the grid
    grid_dim: (num_rows, num_cols)
    """
    (max_x, max_y) = grid_dim       # these are *dimensions*, not max indices
    (x, y) = pos

    # valid indices: 0 <= x < max_x, 0 <= y < max_y
    x_in = (x >= 0) & (x < max_x)
    y_in = (y >= 0) & (y < max_y)
    return bool(x_in * y_in)


# ===========================
def possible_next_steps(grid_dim, last_pos):
    """
    Parameters
    ----------
    grid_dim : tuple of 2 ints
        dimensions of the grid
    last_pos : tuple of 2 ints
        x, y coordinates of current position

    Returns
        possible_steps: list of list of tuples (x,y) denoting the
        next 2 movements possible in every direction possible
    """
    x_pos, y_pos = last_pos # unroll coordinates
    
    possible_steps = []
    operations_1 = [(0,1), (0,-1), (1,0), (-1,0)]
    operations_2 = [(0,2), (0,-2), (2,0), (-2,0)]
    num_operations = len(operations_1)
    
    for i in range(num_operations):
        op1_x, op1_y = operations_1[i]
        op2_x, op2_y = operations_2[i]
        
        if (is_in_map((x_pos + op1_x, y_pos + op1_y), grid_dim)) and (is_in_map((x_pos + op2_x, y_pos + op2_y), grid_dim)):
            possible_steps.append([(x_pos + op1_x, y_pos + op1_y), (x_pos + op2_x, y_pos + op2_y)])
    return possible_steps
# ===========================
def generate_step(grid, last_pos, pos_history, back_step):
    """
    Parameters
    ----------
    grid : list of list of ints
        the grid, it is filled with 0, 1, 2, 3 that correspond
        to different colors
    last_pos : tuple of 2 ints
        x, y coordinates of current position
    pos_history : list of tuples of 2 ints
        coordinates of last visited nodes, only add when see for the
        first time

    Returns
        changes grid[x][y] to white through the path the algorithm is going
        and paints the last_pos on the grid blue
        returns grid, last_pos, back_step, done
    """
    (x, y) = last_pos
    grid[x, y] = 1
    
    grid_dim = (len(grid), len(grid[0]))
    possible_steps = possible_next_steps(grid_dim, last_pos)
    
    valid_steps = []
    for step in possible_steps:
        (x1, y1) = step[0]
        (x2, y2) = step[1]
        
        not_white = (grid[x1, y1] != 1) & (grid[x2, y2] != 1)
        not_green = (grid[x1, y1] != 2) & (grid[x2, y2] != 2)
        
        if bool(not_white * not_green):
            valid_steps.append(step)
    
    #print(f"Valid steps: {valid_steps}")
    
    if (len(valid_steps) == 0): # if it is a dead end
        last_pos = pos_history[-2 - back_step]
        if last_pos == (0,0):
            done = True
            return grid, last_pos, back_step, done
        back_step += 1
        done = False
        return grid, last_pos, back_step, done
    
    else:
        back_step = 0 # reset it
        # choose a valid step at random
        if (len(valid_steps) == 1):
            last_pos = valid_steps[0]
            (x1, y1) = last_pos[0]
            (x2, y2) = last_pos[1]
            grid[x1, y1] = 1
            grid[x2, y2] = 4
            last_pos = last_pos[1]
            done = False
            return grid, last_pos, back_step, done
        else:
            index = randint(0, len(valid_steps))
            # print(f"valid: {len(valid_steps)}, chose {index}")
            last_pos = valid_steps[index]
            (x1, y1) = last_pos[0]
            (x2, y2) = last_pos[1]
            grid[x1, y1] = 1
            grid[x2, y2] = 4
            last_pos = last_pos[1]
            done = False
            return grid, last_pos, back_step, done
        

# Dynamic DFS Maze Generation Algorithm
def generate_maze_dfs(width: int, height: int, seed: int | None = None) -> Maze:
    """
    Generate a maze using the original DFS-based algorithm.

    Returns a Maze object with a numpy 2D grid.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize grid full of zeros
    grid = np.zeros((height, width), dtype=int)


    # Start position and history (same logic as original main)
    last_pos = (0, 0)
    pos_history = [last_pos]
    back_step = 0
    done = False

    # Define start and goal cells, as in the original script
    grid[0, 0] = 2    # start
    grid[-1, -1] = 3  # goal

    # Core DFS carving loop – uses the existing generate_step function
    while not done:
        grid, last_pos, back_step, done = generate_step(grid, last_pos, pos_history, back_step)
        if last_pos not in pos_history:
            pos_history.append(last_pos)

    return Maze(grid=grid)
 
#==============================================================================
#==============================================================================


def dfs_step_sequence(width: int, height: int, seed: int | None = None):
    """
    Generator that yields the grid at each DFS step.

    This uses the same logic as generate_maze_dfs, but instead of
    returning only the final maze, it yields intermediate grids for animation.
    """
    if seed is not None:
        np.random.seed(seed)

    # Integer grid
    grid = np.zeros((height, width), dtype=int)

    last_pos = (0, 0)
    pos_history = [last_pos]
    back_step = 0
    done = False

    # start & goal markers (same convention)
    grid[0, 0] = 2
    grid[-1, -1] = 3

    # yield initial state
    yield grid.copy()

    # DFS carving loop
    while not done:
        grid, last_pos, back_step, done = generate_step(grid, last_pos, pos_history, back_step)
        if last_pos not in pos_history:
            pos_history.append(last_pos)

        # yield a copy for visualization
        yield grid.copy()

    # final state (optional extra yield, but safe)
    yield grid.copy()


#==============================================================================
#==============================================================================





if __name__ == "__main__":
    import argparse
    import csv
    import time

    start_t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mazes", type=int, default=1, help="Number of mazes to generate")
    parser.add_argument("--rows", type=int, default=40, help="Maze height (rows)")
    parser.add_argument("--cols", type=int, default=40, help="Maze width (columns)")
    parser.add_argument("--display", type=int, default=0, help="Show pygame window (1) or not (0)")
    args = parser.parse_args()

    MAZES_DIR.mkdir(parents=True, exist_ok=True)

    for iter_maze in range(args.num_mazes):
        print(f"Generating Maze {iter_maze}/{args.num_mazes - 1}...", end=" ")

        maze = generate_maze_dfs(args.cols, args.rows)
        grid = maze.grid

        out_path = MAZES_DIR / f"maze_{iter_maze}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(grid)

        print(f"saved to {out_path}")

    print(f"--- finished {time.time() - start_t0:.3f} s---")
