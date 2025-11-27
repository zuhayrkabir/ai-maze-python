# maze_visualizer/algorithms/generation/prim_generator.py

from __future__ import annotations

from typing import Tuple, List, Set
import numpy as np

from maze_visualizer.core.maze import Maze
from .dfs_generator import possible_next_steps  # reuse neighbor logic

Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


def generate_maze_prim(width: int, height: int, seed: int | None = None) -> Maze:
    """
    Generate a maze using randomized Prim's algorithm on a grid
    with the same connectivity pattern as the DFS generator.

    Grid encoding (consistent with DFS / visualizer):
      0 = wall
      1 = open path
      2 = start
      3 = goal
    """
    if seed is not None:
        np.random.seed(seed)

    # All walls initially
    grid = np.zeros((height, width), dtype=int)

    # Dimensions as used by possible_next_steps
    grid_dim = (height, width)

    # Start at top-left
    start: Coord = (0, 0)
    visited: Set[Coord] = set()
    visited.add(start)

    # Mark start cell as "start" (2)
    sx, sy = start
    grid[sx, sy] = 2

    # Frontier edges: each is (wall_cell, dest_cell)
    frontier: List[Edge] = []

    def add_frontier_edges(cell: Coord) -> None:
        """
        From a given cell, add all 2-step edges (wall, dest) to frontier
        using the same neighbor logic as DFS.
        """
        steps = possible_next_steps(grid_dim, cell)
        for wall, dest in steps:
            if dest not in visited:
                frontier.append((wall, dest))

    # Initialize frontier from the start cell
    add_frontier_edges(start)

    # Main Prim loop
    while frontier:
        # Pick a random frontier edge
        idx = np.random.randint(len(frontier))
        wall, dest = frontier.pop(idx)

        if dest in visited:
            # Would create a cycle; skip
            continue

        wx, wy = wall
        dx, dy = dest

        # Carve wall and destination cell as passages
        if grid[wx, wy] == 0:
            grid[wx, wy] = 1
        if grid[dx, dy] == 0:
            grid[dx, dy] = 1

        visited.add(dest)
        add_frontier_edges(dest)

    # Mark goal at bottom-right
    gx, gy = height - 1, width - 1
    grid[gx, gy] = 3

    return Maze(grid=grid)


def prim_step_sequence(width: int, height: int, seed: int | None = None):
    """
    Generator that yields the grid state at each step of Prim's maze generation.
    This enables animation in the visualizer.
    """
    if seed is not None:
        np.random.seed(seed)

    # All walls at start
    grid = np.zeros((height, width), dtype=int)

    grid_dim = (height, width)

    start = (0, 0)
    visited: Set[Coord] = set([start])
    grid[0, 0] = 2  # start

    frontier: List[Edge] = []

    def add_frontier_edges(cell: Coord):
        for wall, dest in possible_next_steps(grid_dim, cell):
            if dest not in visited:
                frontier.append((wall, dest))

    add_frontier_edges(start)

    # Yield initial grid
    yield grid.copy()

    # Main Prim loop
    while frontier:
        idx = np.random.randint(len(frontier))
        wall, dest = frontier.pop(idx)

        if dest in visited:
            continue

        wx, wy = wall
        dx, dy = dest

        # Carve wall and destination
        if grid[wx, wy] == 0:
            grid[wx, wy] = 1
        if grid[dx, dy] == 0:
            grid[dx, dy] = 1

        visited.add(dest)
        add_frontier_edges(dest)

        # Yield current state
        yield grid.copy()

    # Mark goal
    grid[-1, -1] = 3
    yield grid.copy()
