from .dfs_generator import generate_maze_dfs
from .prim_generator import generate_maze_prim
from .kruskal_generator import generate_maze_kruskal  # <- make sure this is here

def generate_maze(algo: str, width: int, height: int, seed: int | None = None):
    algo = algo.lower()
    if algo == "dfs":
        return generate_maze_dfs(width, height, seed=seed)
    elif algo == "prim":
        return generate_maze_prim(width, height, seed=seed)
    elif algo == "kruskal":
        return generate_maze_kruskal(width, height, seed=seed)
    else:
        raise ValueError(f"Unknown maze generation algorithm: {algo}")
