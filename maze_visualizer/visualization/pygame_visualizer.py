# -*- coding: utf-8 -*-
"""
Original visualize.py, adapted to new folder layout.

Loads a maze or solved-maze CSV and displays it with pygame.
"""
import time
import argparse
import pygame
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    start_t0 = time.time()

    # ---------------- Paths ----------------
    # .../ai-maze-python/maze_visualizer
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    MAZES_DIR = DATA_DIR / "mazes"
    SOLUTIONS_DIR = DATA_DIR / "solutions"

    # ---------------- CLI ----------------
    # example: python -m maze_visualizer.visualization.pygame_visualizer --maze_file=maze_1.csv --algorithm=aStar
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        help="Implemented: bfs, aStar, dfs, <empty> (or unsolved)",
        default="",
        type=str,
    )
    parser.add_argument(
        "--maze_file",
        help="filename (csv) of the maze to load.",
        default="maze_1.csv",
        type=str,
    )
    args = parser.parse_args()

    algo = args.algorithm

    # ---------------- Choose file to load ----------------
    if algo == "":
        # raw maze
        address = MAZES_DIR / args.maze_file

    elif algo == "bfs":
        address = SOLUTIONS_DIR / f"bfs_{args.maze_file}"

    elif algo == "dfs":
        address = SOLUTIONS_DIR / f"dfs_{args.maze_file}"

    elif algo == "aStar":
        address = SOLUTIONS_DIR / f"aStar_{args.maze_file}"

    else:
        raise Exception("Not valid --algorithm parameter. (e.g bfs, aStar, dfs, <empty>)")

    try:
        grid = np.genfromtxt(address, delimiter=",", dtype=int)
    except OSError:
        raise Exception(f"Maze {address} not found.")

    # ---------------- Grid + colors ----------------
    num_rows = len(grid)
    num_columns = len(grid[0])

    # define colors of the grid RGB
    black = (0, 0, 0)       # grid == 0
    white = (255, 255, 255) # grid == 1
    green = (50, 205, 50)   # grid == 2
    red = (255, 99, 71)     # grid == 3
    grey = (211, 211, 211)  # background
    blue = (153, 255, 255)  # grid == 4, explored
    magenta = (255, 0, 255) # grid == 5, solution

    # cell geometry
    height = 7
    width = height
    margin = 1

    # ---------------- Pygame setup ----------------
    pygame.init()

    WINDOW_SIZE = [330, 330]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    pygame.display.set_caption(f"Visualizing: {address}")
    clock = pygame.time.Clock()

    idx_to_color = [black, white, green, red, blue, magenta]
    finish = False

    # draw initial grid
    screen.fill(grey)

    for row in range(num_rows):
        for column in range(num_columns):
            color = idx_to_color[grid[row, column]]
            pygame.draw.rect(
                screen,
                color,
                [
                    (margin + width) * column + margin,
                    (margin + height) * row + margin,
                    width,
                    height,
                ],
            )

    clock.tick(60)
    pygame.display.flip()

    # ---------------- Event loop ----------------
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True

    pygame.quit()
    print(f"--- finished {time.time() - start_t0:.3f} s ---")
