# 🧩 AI Maze Generator & Pathfinding Visualizer

A fully modular, extensible Python project for generating and solving mazes using classic algorithms — featuring real-time visualization, multiple generation methods, animated exploration, and a clean software-architecture–driven codebase.

This project began as a fork of the original DFS generator/solver and has been transformed into a complete framework for experimenting with maze generation + pathfinding algorithms.




# Features

🔨 Maze Generation Algorithms (Static + Animated)

Implemented in a clean, extensible architecture under maze_visualizer/algorithms/generation/:

1. DFS (Recursive Backtracker)
- Depth-first exploration
- Produces long, winding mazes
- Fully animatable (see the maze carve itself)

2. Prim’s Algorithm (Randomized)
- Uniformly expanding frontier
- Creates bushier, more organic mazes
- Fully animatable

3. Kruskal’s Algorithm (Union–Find)
- Graph-theoretic spanning tree construction
- Carves corridors by merging components
- Fully animatable

Use --gen_algo dfs/prim/kruskal to switch.



## About
This project consists of two parts: An automatic maze generator and an automatic Maze solver. 

1. The maze generator uses DFS (Depth first search) to create a maze from a blanck grid. The generated maze is into a csv file. 

In order to increase the speed its better not to visualize them with Pygame (--display=0). Folder _Mazes_ contains 30 mazes generated with this method.

2. Three automatic Maze solvers for comparison.
* `dfs_pathfinder` uses DFS (Depth First Search) to solve the maze blindly. It's inefficient (most of the times runs through the whole maze to find the solution).
* `bfs_pathfinder` uses DFS (Breadth First Search) to solve the maze blindly. It's inefficient (most of the times runs through the whole maze to find the solution).
* `aStar_pathfinder` uses A* algorithm (its an informed algorithm that takes decissions based on a cost funtion).

Using Pygame one can visualize the process of solving the maze. When the solution is found, the script backtracks the path to show the solution found in magenta, as seen in the image below (NOTE: Blue colored cells are explored cells that are not part of the solution)

After solving the maze the solution is then saved into a csv file. Folder `mazes_solutions` contain all the solutions found using A*, DFS, BFS for the mazes in folder _mazes_.

<p float="center">
  <img src="files/maze_generator.gif" alt="maze generation gif" height="250" />
</p>
<div>
  <p>Figure: Maze generation</p>
</div>
<p float="center">
  <img src="files/aStar.gif" alt="solver aStar" height="250"/>
  <img src="files/bfs.gif" alt="solver bfs" height="250"/>
  <img src="files/dfs.gif" alt="solver dfs" height="250"/>
</p>
<div>
  <p>Figure: From left to right, maze solvers: A*, BFS, DFS solving the same maze</p>
</div>

* info on A* search algorithm: https://en.wikipedia.org/wiki/A*_search_algorithm
* info on DFS algorithm: https://en.wikipedia.org/wiki/Depth-first_search
* info on BFS algorithm: https://en.wikipedia.org/wiki/Breadth-first_search

## Requirements
* Install requirements `pip install -r requirements.txt`
