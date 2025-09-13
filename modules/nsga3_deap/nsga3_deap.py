from deap import base, creator, tools, algorithms
import numpy as np
from typing import Callable

def nsga3_deap_func(
    pop_size: int,
    generations: int,
    bounds: list[tuple[float, float]],
    functions: list[Callable[[np.ndarray], float]] | Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None,
    divisions: int = 10,
    ref_points: np.ndarray = None
) -> list[tuple[float, ...]]:
    """
    Utiliza DEAP para resolver NSGA-III com os parâmetros especificados.
    Suporta tanto lista de funções escalares [f1, f2, ..., fM]
    quanto uma única função multiobjetivo f(x) -> np.ndarray.
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

    # Criação dos tipos básicos para DEAP
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * n_obj)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Inicializador de indivíduos
    toolbox.register(
        "individual",
        lambda: creator.Individual(
            [np.random.uniform(b[0], b[1]) for b in bounds]
        ),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Avaliação personalizada
    def evaluate(individual):
        x = np.array(individual, dtype=float)
        if isinstance(functions, list):
            return tuple(f(x) for f in functions)
        elif callable(functions):
            obj_vec = functions(x)
            if not isinstance(obj_vec, np.ndarray):
                raise ValueError("A função multiobjetivo deve retornar np.ndarray")
            return tuple(float(v) for v in obj_vec)

    toolbox.register("evaluate", evaluate)

    # Geração dos pontos de referência para o NSGA-III
    if ref_points is None:
        ref_points = tools.uniform_reference_points(nobj=n_obj, p=divisions)
        
    # Crossover personalizado
    def custom_crossover(ind1, ind2):
        child1, child2 = crossover(np.array(ind1), np.array(ind2))
        return creator.Individual(child1.tolist()), creator.Individual(child2.tolist())

    toolbox.register("mate", custom_crossover)

    # Mutação personalizada
    def custom_mutation(individual):
        mutated = mutation(np.array(individual), bounds)
        individual[:] = mutated.tolist()
        return individual,

    toolbox.register("mutate", custom_mutation)

    # Operador de seleção
    toolbox.register("select", tools.selNSGA3)

    # Inicialização da população
    if initial_pop:
        population = [creator.Individual(ind.tolist()) for ind in initial_pop]
    else:
        population = toolbox.population(n=pop_size)
    
    # Avaliação inicial da população
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    # Loop evolutivo
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=1.0, mutpb=1.0)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population), ref_points=ref_points)
        
    front = tools.emo.sortNondominated(population, len(population), first_front_only=True)[0]

    # Retornar a frente de Pareto
    pareto_front = [tuple(ind.fitness.values) for ind in front]
            
    return pareto_front
