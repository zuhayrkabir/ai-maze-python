from .dfs_generator import generate_maze_dfs

def generate_maze(algo: str, width: int, height: int, seed: int | None = None):
    algo = algo.lower()
    if algo == "dfs":
        return generate_maze_dfs(width, height, seed=seed)
    # prim, kruskal will go here later
    else:
        raise ValueError(f"Unknown maze generation algorithm: {algo}")
