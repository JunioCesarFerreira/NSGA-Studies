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
    functions: list[Callable[[np.ndarray], float]] | Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray,np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None,
    divisions: int = 10,
    ref_points: np.ndarray = None
) -> list[tuple[float, ...]]:
    """
    Utiliza PyMoo para resolver o NSGA-III com os parâmetros especificados.
    """    
    # Número de objetivos
    if isinstance(functions, list):
        n_obj = len(functions)
    elif callable(functions):
        test_obj = functions(np.zeros(len(bounds)))
        if not isinstance(test_obj, np.ndarray):
            raise ValueError("A função multiobjetivo deve retornar np.ndarray")
        n_obj = test_obj.shape[0]
    else:
        raise ValueError("Parâmetro 'functions' inválido")

    n_var = len(bounds)

    # Definir o problema personalizado para PyMoo
    class CustomProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, 
                             n_obj=n_obj, 
                             xl=np.array([b[0] for b in bounds]), 
                             xu=np.array([b[1] for b in bounds]))
        
        def _evaluate(self, X, out, *args, **kwargs):
            if isinstance(functions, list):
                out["F"] = np.array([[f(ind) for f in functions] for ind in X])
            elif callable(functions):
                out["F"] = np.array([functions(ind) for ind in X])

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
            # Corrente do PyMoo: X.shape = (n_parents, n_matings, n_var)
            n_parents, n_matings, n_var_local = X.shape
            assert n_parents == 2, "Este crossover requer 2 pais."
            assert n_var_local == problem.n_var, "Dimensão de variáveis inconsistente."

            # Saída no formato exigido
            Q = np.empty((self.n_offsprings, n_matings, n_var_local), dtype=float)

            for k in range(n_matings):
                p1: np.ndarray = np.asarray(X[0, k, :], dtype=float)
                p2: np.ndarray = np.asarray(X[1, k, :], dtype=float)
                c1, c2 = self.func(p1, p2)
                c1 = np.asarray(c1, dtype=float).reshape(n_var_local)
                c2 = np.asarray(c2, dtype=float).reshape(n_var_local)
                Q[0, k, :] = c1
                Q[1, k, :] = c2

            return Q

    crossover_operator = CustomCrossover(crossover)
    
    class CustomMutation(Mutation):
        def __init__(self, func: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray], bounds: list[tuple[float, float]]):
            super().__init__()
            self.func = func
            self.bounds = bounds

        def _do(self, problem, X, **kwargs):
            Y = np.empty_like(X, dtype=float)
            for i, ind in enumerate(X):
                yi = np.asarray(self.func(ind, self.bounds), dtype=float).reshape(problem.n_var)
                Y[i, :] = yi
            return Y
    
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

