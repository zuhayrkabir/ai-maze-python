# maze_visualizer/algorithms/pathfinding/dfs_solver.py

from __future__ import annotations

from typing import Tuple, Iterable, List
import numpy as np

Coord = Tuple[int, int]


def dfs_step_sequence(grid: np.ndarray) -> Iterable[np.ndarray]:
    """
    Step-by-step DFS maze solving animation.

    Visually:
      - 0 = wall        (black)
      - 1 = open        (white)
      - 2 = start       (green)
      - 3 = goal        (red)
      - 4 = visited / dead-end (blue)
      - 5 = current path / final path (magenta)

    Behavior:
      - DFS walks forward along a path, marking it as 5.
      - When it hits a dead end, it backtracks, turning 5 -> 4.
      - When it finally reaches the goal, the remaining 5's are the solution path.
    """

    # Keep a copy so we don't mutate caller's grid
    original = grid.copy()
    rows, cols = original.shape

    # ---------- 1) Find explicit start/goal BEFORE normalization ----------
    start: Coord | None = None
    goal: Coord | None = None

    for r in range(rows):
        for c in range(cols):
            if original[r, c] == 2:
                start = (r, c)
            elif original[r, c] == 3:
                goal = (r, c)

    # ---------- 2) Normalize: everything non-wall becomes open (1) ----------
    grid = original.copy()
    mask_non_wall = (grid != 0)
    grid[mask_non_wall] = 1

    # ---------- 3) Fallback start/goal if not explicitly set ----------
    if start is None:
        start = (0, 0)
    if goal is None:
        goal = (rows - 1, cols - 1)

    # Ensure start/goal are walkable
    if grid[start[0], start[1]] == 0:
        grid[start[0], start[1]] = 1
    if grid[goal[0], goal[1]] == 0:
        grid[goal[0], goal[1]] = 1

    # Mark them visually
    grid[start[0], start[1]] = 2
    grid[goal[0],  goal[1]]  = 3

    # If start == goal, just show it
    if start == goal:
        yield grid.copy()
        return

    # ---------- 4) DFS setup ----------
    # Stack holds (cell, next_direction_index)
    # next_direction_index tells us which neighbor to try next when we come back to this cell
    stack: List[tuple[Coord, int]] = []
    stack.append((start, 0))

    visited = set([start])

    # For animation, treat current DFS path as 5
    def mark_path(cell: Coord):
        r, c = cell
        if grid[r, c] not in (2, 3):  # don't overwrite start/goal
            grid[r, c] = 5

    def mark_deadend(cell: Coord):
        r, c = cell
        if grid[r, c] not in (2, 3):  # don't overwrite start/goal
            grid[r, c] = 4

    # Mark start on the path
    mark_path(start)

    # Initial frame
    yield grid.copy()

    found_goal = False

    # Directions: down, up, right, left (can tweak order)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # ---------- 5) DFS search loop ----------
    while stack:
        (r, c), dir_idx = stack[-1]

        # If we've reached the goal, stop exploring
        if (r, c) == goal:
            found_goal = True
            yield grid.copy()
            break

        if dir_idx >= len(directions):
            # Tried all neighbors from this cell -> backtrack
            stack.pop()
            mark_deadend((r, c))
            yield grid.copy()
            continue

        # Otherwise, try the next direction from this cell
        dr, dc = directions[dir_idx]
        # Update the direction index for this cell in the stack
        stack[-1] = ((r, c), dir_idx + 1)

        nr, nc = r + dr, c + dc
        nxt = (nr, nc)

        # Bounds check
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue

        # Skip if already visited or is a wall
        if nxt in visited or grid[nr, nc] == 0:
            continue

        # Otherwise, move into this neighbor
        visited.add(nxt)
        stack.append((nxt, 0))

        # Mark as path cell for visualization
        if grid[nr, nc] == 1:
            mark_path(nxt)
        # If it's the goal, we still mark path but will terminate next loop iteration
        yield grid.copy()

    # ---------- 6) Ensure final path is drawn nicely ----------
    # At this point, cells that remain marked as 5 form the DFS-discovered path
    # We can just yield one last frame to be safe
    yield grid.copy()
