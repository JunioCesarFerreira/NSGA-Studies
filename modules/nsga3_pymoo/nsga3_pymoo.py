from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.optimize import minimize
import numpy as np
from typing import Callable
    
def nsga3_pymoo_func(
    pop_size: int,
    generations: int,
    bounds: list[tuple[float, float]],
    functions: list[Callable[[np.ndarray], float]],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray,np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None,
    divisions: int = 10,
    ref_points: np.ndarray = None
) -> list[tuple[float, ...]]:
    """
    Utiliza PyMoo para resolver o NSGA-III com os parâmetros especificados.
    """    
    # Número de objetivos e variáveis de decisão
    n_obj = len(functions)
    n_var = len(bounds)

    # Definir o problema personalizado para PyMoo
    class CustomProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, 
                             n_obj=n_obj, 
                             xl=np.array([b[0] for b in bounds]), 
                             xu=np.array([b[1] for b in bounds]))
        
        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.array([[f(ind) for f in functions] for ind in x])

    problem = CustomProblem()

    # Direções de referência para NSGA-III
    if ref_points is None:
        ref_points = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=divisions)

    # Configurar operadores personalizados
    class CustomCrossover(Crossover):
        def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
            super().__init__(n_parents=2, n_offsprings=2)
            self.func = func

        def _do(self, problem, X, **kwargs):
            children = self.func(X[0], X[1])
            # Converte a lista de filhos para um array NumPy
            return np.array(children)

    crossover_operator = CustomCrossover(crossover)
    
    class CustomMutation(Mutation):
        def __init__(self, func: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray], bounds: list[tuple[float, float]]):
            super().__init__()
            self.func = func
            self.bounds = bounds

        def _do(self, problem, X, **kwargs):
            return np.array([self.func(ind, self.bounds) for ind in X])
    
    mutation_operator = CustomMutation(mutation, bounds)

    # Configuração inicial da população, se fornecida
    initial_population = None
    if initial_pop:
        initial_population = Population.new("X", np.array(initial_pop))

    # Configurar algoritmo NSGA-III
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_points,
        crossover=crossover_operator,
        mutation=mutation_operator,
    )

    # Resolver o problema
    result = minimize(
        problem,
        algorithm,
        termination=('n_gen', generations),
        seed=1,
        verbose=False,
        save_history=False,
        initial_population=initial_population,
    )

    # Extrair a solução
    pareto_front = [tuple(ind) for ind in result.F]

    return pareto_front

