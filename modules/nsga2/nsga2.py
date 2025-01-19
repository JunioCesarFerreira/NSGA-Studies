import random
import numpy as np
from typing import Callable

def nsga2_func(
    pop_size: int, 
    generations: int, 
    bounds: list[tuple[float,float]], 
    functions: list[Callable[[np.ndarray], float]],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray,np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None
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
    def initialize_population(size, bounds):
        return [
            np.array([random.uniform(b[0], b[1]) for b in bounds])
            for _ in range(size)
        ]

    # Avaliação da população
    def evaluate_population(population, functions):
        return [tuple(f(x) for f in functions) for x in population]

    # Dominância
    def dominates(obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    # Ordenação não-dominada (Implementação NSGA-II)
    def fast_nondominated_sort(objectives):
        fronts = [[]]
        domination_count = [0] * len(objectives)
        dominated_solutions = [[] for _ in range(len(objectives))]

        for p in range(len(objectives)):
            for q in range(len(objectives)):
                if dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
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
    def crowding_distance(front, objectives):
        if len(front) == 1:
            return [float('inf')]
        
        distances = [0] * len(front)
        for m in range(len(objectives[0])):
            sorted_front = sorted(range(len(front)), key=lambda i: objectives[front[i]][m])
            distances[sorted_front[0]] = distances[sorted_front[-1]] = float('inf')
            for i in range(1, len(front) - 1):
                distances[sorted_front[i]] += (
                    objectives[front[sorted_front[i + 1]]][m] - objectives[front[sorted_front[i - 1]]][m]
                )
        return distances

    # Seleção com elitismo
    def select_next_population(fronts, objectives, population, pop_size):
        next_population = []
        for front in fronts:
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                distances = crowding_distance(front, objectives)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                next_population.extend([solution for solution, _ in sorted_front[:pop_size - len(next_population)]])
                break
        return [population[i] for i in next_population]  # Seleciona os indivíduos correspondentes

    # Inicializa a população
    if initial_pop is None:
        population = initialize_population(pop_size, bounds)
    else:
        population = initial_pop

    for gen in range(generations):
        # Avaliação
        objectives = evaluate_population(population, functions)

        # Ordenação não-dominada
        fronts = fast_nondominated_sort(objectives)

        # Visualização no final
        if gen == generations - 1:
            pareto_front = [objectives[i] for i in fronts[0]]
            pareto_front.sort()

        # Nova geração
        new_population = []
        while len(new_population) < pop_size:
            # Seleção de dois pais
            parent1, parent2 = random.sample(population, 2)

            # Cruzamento
            children = crossover(parent1, parent2)

            # Mutação
            child = mutation(children[0], bounds)

            new_population.append(child)
            
        # Combinar pais e descendentes
        combined_population = population + new_population
        combined_objectives = evaluate_population(combined_population, functions)
        
        # Realizar nova ordenação não-dominada
        combined_fronts = fast_nondominated_sort(combined_objectives)
        
        # Selecionar próxima geração com elitismo
        population = select_next_population(combined_fronts, combined_objectives, combined_population, pop_size)

    return pareto_front
