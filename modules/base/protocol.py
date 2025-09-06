from typing import Callable, Protocol
import numpy as np

Vector = np.ndarray
Bounds = list[tuple[float, float]]

Objective = Callable[[Vector], float]
Crossover = Callable[[Vector, Vector], tuple[Vector, Vector]]
Mutation = Callable[[Vector, Bounds], Vector]

class NSGA3Callable(Protocol):
    def __call__(  # assinatura “fixada”
        self,
        pop_size: int,
        generations: int,
        bounds: Bounds,
        functions: list[Objective],
        crossover: Crossover,
        mutation: Mutation,
        initial_pop: list[Vector] | None = None,
        divisions: int = 10,
        ref_points: np.ndarray | None = None,
    ) -> list[tuple[float, ...]]: ...