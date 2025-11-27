# maze_visualizer/algorithms/pathfinding/bfs_solver.py

from __future__ import annotations

from collections import deque
from typing import Tuple, Dict, Iterable
import numpy as np

Coord = Tuple[int, int]


def bfs_step_sequence(grid: np.ndarray) -> Iterable[np.ndarray]:
    """
    Step-by-step BFS maze solving animation.
    Yields the grid state after each exploration / path step.

    Grid encoding (visualizer):
        0 = wall      (black)
        1 = open      (white)
        2 = start     (green)
        3 = goal      (red)
        4 = visited   (blue)
        5 = final path (magenta)
    """
    # Work on a copy so original isn't mutated outside
    grid = grid.copy()
    rows, cols = grid.shape

    # ---------- Normalize grid ----------
    # If the CSV was a previous solution (with 4/5 labels),
    # or has any other junk, reset every non-wall cell to "open" (1).
    mask_non_wall = (grid != 0)
    grid[mask_non_wall] = 1

    # ---------- Find / define start & goal ----------
    start: Coord | None = None
    goal: Coord | None = None

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 2:
                start = (r, c)
            elif grid[r, c] == 3:
                goal = (r, c)

    # Fallback: implicit corners if not present
    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (rows - 1, cols - 1)

    # Ensure start/goal are walkable
    if grid[start[0], start[1]] == 0:
        grid[start[0], start[1]] = 1
    if grid[goal[0], goal[1]] == 0:
        grid[goal[0], goal[1]] = 1

    # Mark them explicitly for visualization
    if grid[start[0], start[1]] not in (2, 3):
        grid[start[0], start[1]] = 2
    if grid[goal[0], goal[1]] not in (2, 3):
        grid[goal[0], goal[1]] = 3

    # If start == goal, just show it
    if start == goal:
        yield grid.copy()
        return

    # ---------- BFS setup ----------
    queue = deque([start])
    parent: Dict[Coord, Coord] = {}
    visited: set[Coord] = {start}

    # Show initial grid (start + goal + clean maze)
    yield grid.copy()

    found_goal = False

    # ---------- BFS search phase ----------
    while queue:
        r, c = queue.pop()  # or popleft() for true BFS; pop gives DFS-like frontier order

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            nxt = (nr, nc)
            if nxt in visited:
                continue

            cell_val = grid[nr, nc]

            if cell_val == 0:
                continue  # wall, skip

            visited.add(nxt)
            parent[nxt] = (r, c)

            # If it's a normal open cell, mark as explored (blue = 4)
            if cell_val == 1:
                grid[nr, nc] = 4

            # If goal, stop after marking visited
            if cell_val == 3:
                found_goal = True
                yield grid.copy()
                queue.clear()
                break

            queue.append(nxt)

            # Yield after each exploration step
            yield grid.copy()

        if found_goal:
            break

    if not found_goal:
        # No path, just show explored region
        yield grid.copy()
        return

    # ---------- Path reconstruction phase ----------
    cur = goal
    while cur != start:
        r, c = cur
        # Don't overwrite start/goal colors
        if grid[r, c] not in (2, 3):
            grid[r, c] = 5  # path (magenta)
        cur = parent[cur]

        # Show each backtracking step
        yield grid.copy()

    # Final frame
    yield grid.copy()
