import random
import numpy as np
from typing import Callable, Optional, Sequence

def nsga2_func(
    pop_size: int,
    generations: int,
    bounds: list[tuple[float, float]],
    functions: list[Callable[[np.ndarray], float]],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: Optional[list[np.ndarray]] = None
) -> list[tuple[float, ...]]:
    """
    NSGA-II generalizado para N dimensões.

    :param pop_size: Tamanho da população
    :param generations: Número de gerações
    :param bounds: Lista de tuplas [(min1, max1), (min2, max2), ...] definindo os limites para cada dimensão
    :param functions: Lista de funções objetivo [f1, f2, ..., fM]
    :param crossover: Função de crossover que aceita dois pais e retorna um filho
    :param mutation: Função de mutação que aceita um indivíduo e retorna um indivíduo mutado
    :return: Fronteira de Pareto da última geração
    """
    # Inicialização da população
    def initialize_population(size: int, bounds: list[tuple[float, float]]) -> list[np.ndarray]:
        return [
            np.array([random.uniform(b[0], b[1]) for b in bounds], dtype=float)
            for _ in range(size)
        ]

    # Avaliação da população
    def evaluate_population(
        population: Sequence[np.ndarray],
        functions: list[Callable[[np.ndarray], float]]
    ) -> list[tuple[float, ...]]:
        return [tuple(f(x) for f in functions) for x in population]

    # Dominância
    def dominates(obj1: tuple[float, ...], obj2: tuple[float, ...]) -> bool:
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    # Ordenação não-dominada (Implementação NSGA-II)
    def fast_nondominated_sort(objectives: Sequence[tuple[float, ...]]) -> list[list[int]]:
        fronts: list[list[int]] = [[]]
        domination_count: list[int] = [0] * len(objectives)
        dominated_solutions: list[list[int]] = [[] for _ in range(len(objectives))]

        for p in range(len(objectives)):
            for q in range(len(objectives)):
                if dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        i: int = 0
        while len(fronts[i]) > 0:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()
        return fronts

    # Distância de crowding
    def crowding_distance(front: list[int], objectives: list[tuple[float, ...]]) -> list[float]:
        if len(front) == 1:
            return [float('inf')]

        distances: list[float] = [0.0] * len(front)
        num_obj: int = len(objectives[0])
        for m in range(num_obj):
            sorted_front: list[int] = sorted(range(len(front)), key=lambda i: objectives[front[i]][m])
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            for i in range(1, len(front) - 1):
                distances[sorted_front[i]] += (
                    objectives[front[sorted_front[i + 1]]][m] - objectives[front[sorted_front[i - 1]]][m]
                )
        return distances

    # Seleção com elitismo
    def select_next_population(
        fronts: list[list[int]],
        objectives: list[tuple[float, ...]],
        population: list[np.ndarray],
        pop_size: int
    ) -> list[np.ndarray]:
        next_population: list[int] = []
        for front in fronts:
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                distances: list[float] = crowding_distance(front, objectives)
                sorted_front: list[tuple[int, float]] = sorted(
                    zip(front, distances), key=lambda x: x[1], reverse=True
                )
                next_population.extend(
                    [solution for solution, _ in sorted_front[:pop_size - len(next_population)]]
                )
                break
        return [population[i] for i in next_population]  # Seleciona os indivíduos correspondentes

    # Inicializa a população
    if initial_pop is None:
        population: list[np.ndarray] = initialize_population(pop_size, bounds)
    else:
        population = initial_pop

    for gen in range(generations):
        # Avaliação
        objectives: list[tuple[float, ...]] = evaluate_population(population, functions)

        # Ordenação não-dominada
        fronts: list[list[int]] = fast_nondominated_sort(objectives)

        # Visualização no final
        if gen == generations - 1:
            pareto_front: list[tuple[float, ...]] = [objectives[i] for i in fronts[0]]
            pareto_front.sort()

        # Nova geração
        new_population: list[np.ndarray] = []
        while len(new_population) < pop_size:
            # Seleção de dois pais
            parent1, parent2 = random.sample(population, 2)

            # Cruzamento
            children: tuple[np.ndarray, np.ndarray] = crossover(parent1, parent2)

            # Mutação
            child: np.ndarray = mutation(children[0], bounds)

            new_population.append(child)

        # Combinar pais e descendentes
        combined_population: list[np.ndarray] = population + new_population
        combined_objectives: list[tuple[float, ...]] = evaluate_population(combined_population, functions)

        # Realizar nova ordenação não-dominada
        combined_fronts: list[list[int]] = fast_nondominated_sort(combined_objectives)

        # Selecionar próxima geração com elitismo
        population = select_next_population(combined_fronts, combined_objectives, combined_population, pop_size)

    return pareto_front
