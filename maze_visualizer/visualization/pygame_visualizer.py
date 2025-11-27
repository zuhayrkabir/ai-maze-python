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
from maze_visualizer.algorithms.generation.dfs_generator import dfs_step_sequence





if __name__ == "__main__":

    start_t0 = time.time()

    # ---------------- Paths ----------------
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    MAZES_DIR = DATA_DIR / "mazes"
    SOLUTIONS_DIR = DATA_DIR / "solutions"

    # ---------------- CLI ----------------
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
        choices=["dfs", "prim", "kruskal"],
        default="dfs",
        help="Maze generation algorithm when using live source",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate live maze generation",
    )
    parser.add_argument(
        "--solve",
        choices=["bfs", "dfs", "astar"],
        default=None,
        help="Run an animated pathfinding algorithm on the maze (after it is loaded/generated).",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Advance animation one step at a time on keypress instead of time-based playback.",
    )


    args = parser.parse_args()

    # ---------------- Get maze grid or mode ----------------
    if args.source == "csv":
        algo = args.algorithm

        if algo == "":
            address = MAZES_DIR / args.maze_file
        elif algo == "bfs":
            address = SOLUTIONS_DIR / f"bfs_{args.maze_file}"
        elif algo == "dfs":
            address = SOLUTIONS_DIR / f"dfs_{args.maze_file}"
        elif algo == "aStar":
            address = SOLUTIONS_DIR / f"aStar_{args.maze_file}"
        else:
            raise Exception(
                "Not valid --algorithm parameter. (e.g bfs, aStar, dfs, <empty>)"
            )

        try:
            grid = np.genfromtxt(address, delimiter=",", dtype=int)
        except OSError:
            raise Exception(f"Maze {address} not found.")

        caption_text = f"Visualizing file: {address}"

    else:
        # Live generation
        if args.animate:
            # We'll animate generation later; no static grid yet
            grid = None
            caption_text = f"Animating {args.gen_algo.upper()} maze generation ({args.rows}x{args.cols})"
        else:
            # Static live maze
            maze = generate_maze(args.gen_algo, args.cols, args.rows)
            grid = maze.grid.astype(int)
            address = f"live:{args.gen_algo}"
            caption_text = f"Live {args.gen_algo} maze ({args.rows}x{args.cols})"

    # ---------------- Colors & Pygame setup ----------------
    black   = (0, 0, 0)       # 0
    white   = (255, 255, 255) # 1
    green   = (50, 205, 50)   # 2
    red     = (255, 99, 71)   # 3
    grey    = (211, 211, 211) # background
    blue    = (153, 255, 255) # 4
    magenta = (255, 0, 255)   # 5

    height = 7
    width  = height
    margin = 1

    pygame.init()
    WINDOW_SIZE = [330, 330]
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(caption_text)
    clock = pygame.time.Clock()
    idx_to_color = [black, white, green, red, blue, magenta]

    finish = False

    def draw_grid(grid_array: np.ndarray):
        screen.fill(grey)
        num_rows, num_columns = grid_array.shape

        for row in range(num_rows):
            for column in range(num_columns):
                val = int(grid_array[row, column])
                color = idx_to_color[val]
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
        pygame.display.flip()


    def wait_for_step_or_quit():
        """
        In step mode: wait until user presses space/enter/right, or closes the window.
        Returns True if we should continue, False if user requested quit.
        """
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT):
                        waiting = False
                        break
            clock.tick(60)
        return True



    # ---------------- Generation animation (live source) ----------------
        # ---------------- Generation animation (live source) ----------------
    if args.source == "live" and args.animate:
        if args.gen_algo == "dfs":
            from maze_visualizer.algorithms.generation.dfs_generator import dfs_step_sequence
            step_gen = dfs_step_sequence(args.cols, args.rows)
        elif args.gen_algo == "prim":
            from maze_visualizer.algorithms.generation.prim_generator import prim_step_sequence
            step_gen = prim_step_sequence(args.cols, args.rows)
        elif args.gen_algo == "kruskal":
            from maze_visualizer.algorithms.generation.kruskal_generator import kruskal_step_sequence
            step_gen = kruskal_step_sequence(args.cols, args.rows)
        else:
            raise Exception("Animation not supported for this generation algorithm.")

        # Play generation animation
        for step_grid in step_gen:
            if args.step:
                # Step-by-step mode: wait for key each frame
                should_continue = wait_for_step_or_quit()
                if not should_continue:
                    finish = True
                    break
            else:
                # Time-based mode: allow quitting and run at fixed FPS
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        finish = True
                        break
                if finish:
                    break
                clock.tick(10)  # FPS for generation animation

            draw_grid(step_grid.astype(int))

        # After animation finishes, capture final grid for possible solving
        current_grid = step_grid.astype(int)


    else:
        # Static display (CSV or live)
        current_grid = grid.astype(int)
        # Only draw immediately if we're NOT going to run a solver
        if args.solve is None:
            draw_grid(current_grid)


    # ---------------- Pathfinding animation (on current_grid) ----------------
    if args.solve is not None:
        if args.solve == "bfs":
            from maze_visualizer.algorithms.pathfinding.bfs_solver import bfs_step_sequence
            step_gen = bfs_step_sequence(current_grid.copy())
        elif args.solve == "dfs":
            from maze_visualizer.algorithms.pathfinding.dfs_solver import dfs_step_sequence
            step_gen = dfs_step_sequence(current_grid)
        elif args.solve == "astar":
            # placeholder: astar_step_sequence(grid)
            raise NotImplementedError("A* solve animation not implemented yet.")
        else:
            raise Exception("Invalid solver selected.")

        for step_grid in step_gen:
            # Allow quitting during solving animation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    finish = True
                    break
            if finish:
                break

            draw_grid(step_grid.astype(int))
            clock.tick(30)  # 10 FPS; lower = slower, more visible

        # After solving, keep the last frame
        current_grid = step_grid.astype(int)


    # ---------------- Event loop (hold final frame) ----------------
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True
        clock.tick(30)

    pygame.quit()
    print(f"--- finished {time.time() - start_t0:.3f} s ---")
