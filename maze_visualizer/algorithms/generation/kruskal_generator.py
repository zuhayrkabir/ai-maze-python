# maze_visualizer/algorithms/generation/kruskal_generator.py

from __future__ import annotations

from typing import Tuple, List, Dict
import numpy as np

from maze_visualizer.core.maze import Maze

Coord = Tuple[int, int]


def generate_maze_kruskal(width: int, height: int, seed: int | None = None) -> Maze:
    """
    Generate a maze using randomized Kruskal's algorithm, assuming a grid where
    "cells" live at even coordinates (0,2,4,...) and walls are between them.

    Encoding:
      0 = wall
      1 = passage
      2 = start
      3 = goal
    """
    if seed is not None:
        np.random.seed(seed)

    # Full grid, all walls initially
    grid = np.zeros((height, width), dtype=int)

    # ---- Define cells on a coarse grid (even indices) ----
    cells: List[Coord] = [
        (r, c)
        for r in range(0, height, 2)
        for c in range(0, width, 2)
    ]

    # Helper to get neighbors (2 steps away) and the wall between
    edges: List[Tuple[Coord, Coord, Coord]] = []
    for (r, c) in cells:
        for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                # Neighbor cell
                cell_a = (r, c)
                cell_b = (nr, nc)
                # Wall halfway between them
                wall = (r + dr // 2, c + dc // 2)
                # To avoid duplicate undirected edges, only keep one ordering
                if cell_a < cell_b:
                    edges.append((cell_a, cell_b, wall))

    # Randomize edge order
    np.random.shuffle(edges)

    # ---- Union-Find over cell coordinates ----
    parent: Dict[Coord, Coord] = {c: c for c in cells}
    rank: Dict[Coord, int] = {c: 0 for c in cells}

    def find(a: Coord) -> Coord:
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a: Coord, b: Coord) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # ---- Kruskal: build spanning tree over cells ----
    for cell_a, cell_b, wall in edges:
        if union(cell_a, cell_b):
            # Carve corridor: open both cells + the wall between
            ar, ac = cell_a
            br, bc = cell_b
            wr, wc = wall

            grid[ar, ac] = 1  # cell A passage
            grid[br, bc] = 1  # cell B passage
            grid[wr, wc] = 1  # wall converted to passage

    # Mark start and goal
    grid[0, 0] = 2          # start
    grid[-1, -1] = 3        # goal

    return Maze(grid=grid)


def kruskal_step_sequence(width: int, height: int, seed: int | None = None):
    """
    Generator that yields the grid at each step of Kruskal's algorithm.
    """
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros((height, width), dtype=int)

    cells: List[Coord] = [
        (r, c)
        for r in range(0, height, 2)
        for c in range(0, width, 2)
    ]

    edges: List[Tuple[Coord, Coord, Coord]] = []
    for (r, c) in cells:
        for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                cell_a = (r, c)
                cell_b = (nr, nc)
                wall = (r + dr // 2, c + dc // 2)
                if cell_a < cell_b:
                    edges.append((cell_a, cell_b, wall))

    np.random.shuffle(edges)

    parent: Dict[Coord, Coord] = {c: c for c in cells}
    rank: Dict[Coord, int] = {c: 0 for c in cells}

    def find(a: Coord) -> Coord:
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a: Coord, b: Coord) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # initial all-walls state
    yield grid.copy()

    for cell_a, cell_b, wall in edges:
        if union(cell_a, cell_b):
            ar, ac = cell_a
            br, bc = cell_b
            wr, wc = wall

            grid[ar, ac] = 1
            grid[br, bc] = 1
            grid[wr, wc] = 1

            yield grid.copy()

    grid[0, 0] = 2
    grid[-1, -1] = 3
    yield grid.copy()
