# maze_visualizer/algorithms/pathfinding/dijkstra_solver.py

from __future__ import annotations

from typing import Tuple, Dict, Iterable
import heapq
import numpy as np

Coord = Tuple[int, int]


def dijkstra_step_sequence(grid: np.ndarray) -> Iterable[np.ndarray]:
    """
    Step-by-step Dijkstra maze solving animation.

    Grid encoding (visualizer):
        0 = wall      (black)
        1 = open      (white)
        2 = start     (green)
        3 = goal      (red)
        4 = visited   (blue)
        5 = final path (magenta)
    """
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

    # ---------- 4) Dijkstra setup ----------
    dist: Dict[Coord, float] = {start: 0.0}
    came_from: Dict[Coord, Coord] = {}

    # priority queue of (distance, tie_breaker, cell)
    heap: list[tuple[float, int, Coord]] = []
    heapq.heappush(heap, (0.0, 0, start))

    visited: set[Coord] = set()
    step_counter = 1
    found_goal = False

    # Initial frame
    yield grid.copy()

    # ---------- 5) Main Dijkstra loop ----------
    while heap:
        cur_dist, _, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        r, c = current

        # Mark as visited (blue), but don't overwrite start/goal
        if grid[r, c] == 1:
            grid[r, c] = 4

        # If we reached goal, stop search
        if current == goal:
            found_goal = True
            yield grid.copy()
            break

        # Explore neighbors (4-connected)
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            if grid[nr, nc] == 0:  # wall
                continue

            new_dist = cur_dist + 1.0  # uniform cost

            if new_dist < dist.get(nxt, float("inf")):
                dist[nxt] = new_dist
                came_from[nxt] = current
                heapq.heappush(heap, (new_dist, step_counter, nxt))
                step_counter += 1

                # Mark frontier as "visited-ish" (blue) if it's just open
                if grid[nr, nc] == 1:
                    grid[nr, nc] = 4

        # Yield after expanding this node
        yield grid.copy()

    if not found_goal:
        # No path found; just show visited region
        yield grid.copy()
        return

    # ---------- 6) Reconstruct path ----------
    cur = goal
    while cur != start:
        r, c = cur
        if grid[r, c] not in (2, 3):
            grid[r, c] = 5  # final path
        cur = came_from[cur]
        yield grid.copy()

    # Final frame
    yield grid.copy()
