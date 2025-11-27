from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Maze:
    grid: np.ndarray  # 2D array of ints

    @property
    def num_rows(self) -> int:
        return self.grid.shape[0]

    @property
    def num_cols(self) -> int:
        return self.grid.shape[1]
