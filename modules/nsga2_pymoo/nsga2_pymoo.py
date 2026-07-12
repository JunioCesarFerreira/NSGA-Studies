from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.optimize import minimize
import numpy as np
from typing import Callable

def nsga2_pymoo_func(
    pop_size: int,
    generations: int,
    bounds: list[tuple[float, float]],
    functions: list[Callable[[np.ndarray], float]] | Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None,
    seed: int = 1
) -> list[tuple[float, ...]]:
    """
    Utiliza PyMoo para resolver o NSGA-II com os parâmetros especificados.

    :param seed: semente do PyMoo. Por padrão 1.
    :param initial_pop: se fornecida, é injetada como população inicial via ``sampling``
        (o parâmetro ``initial_population`` de ``minimize`` é ignorado pelo PyMoo).
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

    # Configurar operadores personalizados
    class CustomCrossover(Crossover):
        def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
            super().__init__(n_parents=2, n_offsprings=2)
            self.func = func

        def _do(self, problem, X, **kwargs):
            n_parents, n_matings, n_var_local = X.shape
            assert n_parents == 2, "Este crossover requer 2 pais."
            assert n_var_local == problem.n_var, "Dimensão de variáveis inconsistente."

            Q = np.empty((self.n_offsprings, n_matings, n_var_local), dtype=float)
            for k in range(n_matings):
                p1: np.ndarray = np.asarray(X[0, k, :], dtype=float)
                p2: np.ndarray = np.asarray(X[1, k, :], dtype=float)
                c1, c2 = self.func(p1, p2)
                Q[0, k, :] = np.asarray(c1, dtype=float).reshape(n_var_local)
                Q[1, k, :] = np.asarray(c2, dtype=float).reshape(n_var_local)
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
                Y[i, :] = np.asarray(self.func(ind, self.bounds), dtype=float).reshape(problem.n_var)
            return Y

    mutation_operator = CustomMutation(mutation, bounds)

    # População inicial: injetada via `sampling` (o `minimize` ignora `initial_population`).
    nsga2_kwargs = dict(
        pop_size=pop_size,
        crossover=crossover_operator,
        mutation=mutation_operator,
    )
    if initial_pop is not None and len(initial_pop) > 0:
        nsga2_kwargs["sampling"] = Population.new("X", np.array(initial_pop))

    algorithm = NSGA2(**nsga2_kwargs)

    result = minimize(
        problem,
        algorithm,
        termination=('n_gen', generations),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    pareto_front = [tuple(ind) for ind in result.F]

    return pareto_front
