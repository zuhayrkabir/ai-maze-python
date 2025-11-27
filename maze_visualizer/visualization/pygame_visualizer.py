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
from maze_visualizer.algorithms.generation import generate_maze




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


    parser.add_argument(
    "--source",
    choices=["csv", "live"],
    default="csv",
    help="Where to get the maze from: csv file or live generation",
    )
    parser.add_argument(
        "--rows", type=int, default=40, help="Maze rows when using live source"
    )
    parser.add_argument(
        "--cols", type=int, default=40, help="Maze cols when using live source"
    )
    parser.add_argument(
        "--gen_algo",
        choices=["dfs"],  # later: ["dfs", "prim", "kruskal"]
        default="dfs",
        help="Maze generation algorithm when using live source",
    )


    args = parser.parse_args()

    # ---------------- Get maze grid ----------------
    if args.source == "csv":
        algo = args.algorithm

        # Choose file to load based on algorithm
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

        caption_text = f"Visualizing file: {address}"

    else:
        # Live generation using your algorithms
        maze = generate_maze(args.gen_algo, args.cols, args.rows)
        grid = maze.grid
        grid = grid.astype(int)

        address = f"live:{args.gen_algo}"
        caption_text = f"Live {args.gen_algo} maze ({args.rows}x{args.cols})"

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

    pygame.display.set_caption(caption_text)
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
