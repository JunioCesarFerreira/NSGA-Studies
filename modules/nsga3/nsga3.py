import random
import numpy as np
from typing import Callable 
from collections import defaultdict

def nsga3_func(
    pop_size: int, 
    generations: int, 
    bounds: list[tuple[float,float]], 
    functions: list[Callable[[np.ndarray], float]],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: list[np.ndarray] = None,
    divisions: int=10
    ) -> list[tuple[float, ...]]:
    """
    NSGA-III generalizado para N dimensões.

    :param pop_size: Tamanho da população
    :param generations: Número de gerações
    :param bounds: Lista de tuplas [(min1, max1), (min2, max2), ...] definindo os limites para cada dimensão
    :param functions: Lista de funções objetivo [f1, f2, ..., fM]
    :param crossover: Função de crossover que aceita dois pais e retorna um filho
    :param mutation: Função de mutação que aceita um indivíduo e retorna um indivíduo mutado
    :param divisions: Número de divisões para geração dos pontos de referência
    :return: Fronteira de Pareto da última geração
    """
    def initialize_population(size, bounds):
        return [
            np.array([random.uniform(b[0], b[1]) for b in bounds])
            for _ in range(size)
        ]

    def evaluate_population(population, functions):
        return [tuple(f(x) for f in functions) for x in population]

    def dominates(obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    def fast_nondominated_sort(objectives):
        population_size = len(objectives)
        S = [[] for _ in range(population_size)]
        n = [0] * population_size
        rank = [0] * population_size
        fronts = [[]]

        for p in range(population_size):
            for q in range(population_size):
                if dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    def generate_reference_points(M, p):
        def generate_recursive(points, num_objs, left, total, depth, current_point):
            if depth == num_objs - 1:
                current_point.append(left / total)
                points.append(current_point.copy())
                current_point.pop()
            else:
                for i in range(left + 1):
                    current_point.append(i / total)
                    generate_recursive(points, num_objs, left - i, total, depth + 1, current_point)
                    current_point.pop()
        points = []
        generate_recursive(points, M, p, p, 0, [])
        return np.array(points)

    def environmental_selection(population, objectives, fronts, reference_points, pop_size):
        next_population_indices = []
        for front in fronts:
            if len(next_population_indices) + len(front) <= pop_size:
                next_population_indices.extend(front)
            else:
                N = pop_size - len(next_population_indices)
                selected_indices = niching_selection(front, objectives, reference_points, N)
                next_population_indices.extend(selected_indices)
                break
        next_population = [population[i] for i in next_population_indices]
        return next_population

    def niching_selection(front, objectives, reference_points, N):
        selected = []
        objs = np.array([objectives[i] for i in front])
        ideal_point = np.min(objs, axis=0)
        normalized_objs = objs - ideal_point

        max_values = np.max(normalized_objs, axis=0)
        max_values[max_values == 0] = 1
        normalized_objs = normalized_objs / max_values

        associations = []
        for idx, obj in zip(front, normalized_objs):
            distances = np.linalg.norm(obj - reference_points, axis=1)
            min_index = np.argmin(distances)
            associations.append((idx, min_index, distances[min_index]))

        reference_associations = defaultdict(list)
        for idx, ref_idx, dist in associations:
            reference_associations[ref_idx].append((idx, dist))

        niche_counts = {i: 0 for i in range(len(reference_points))}
        selected_flags = {idx: False for idx in front}

        while len(selected) < N:
            min_niche_count = min(niche_counts.values())
            min_refs = [ref for ref, count in niche_counts.items() if count == min_niche_count]

            for ref_idx in min_refs:
                assoc_inds = reference_associations.get(ref_idx, [])
                unselected_inds = [(idx, dist) for idx, dist in assoc_inds if not selected_flags[idx]]

                if unselected_inds:
                    unselected_inds.sort(key=lambda x: x[1])
                    selected_idx = unselected_inds[0][0]
                    selected.append(selected_idx)
                    selected_flags[selected_idx] = True
                    niche_counts[ref_idx] += 1
                    break
            else:
                remaining = [idx for idx in front if not selected_flags[idx]]
                if remaining:
                    selected_idx = random.choice(remaining)
                    selected.append(selected_idx)
                    selected_flags[selected_idx] = True
                else:
                    break

        return selected[:N]

    def compute_individual_ranks(fronts):
        individual_ranks = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                individual_ranks[idx] = rank
        return individual_ranks

    def tournament_selection(population, individual_ranks):
        i1, i2 = random.sample(range(len(population)), 2)
        rank1 = individual_ranks[i1]
        rank2 = individual_ranks[i2]
        if rank1 < rank2:
            return population[i1]
        elif rank2 < rank1:
            return population[i2]
        else:
            return population[random.choice([i1, i2])]
    
    # Inicializa a população
    if initial_pop is None:
        population = initialize_population(pop_size, bounds)
    else:
        population = initial_pop
        
    M = len(functions)
    reference_points = generate_reference_points(M, divisions)

    for gen in range(generations):
        objectives = evaluate_population(population, functions)
        fronts = fast_nondominated_sort(objectives)
        individual_ranks = compute_individual_ranks(fronts)

        offspring_population = []
        while len(offspring_population) < pop_size:
            parent1 = tournament_selection(population, individual_ranks)
            parent2 = tournament_selection(population, individual_ranks)
            children = crossover(parent1, parent2)
            child = mutation(children[0], bounds)
            offspring_population.append(child)

        combined_population = population + offspring_population
        combined_objectives = evaluate_population(combined_population, functions)
        combined_fronts = fast_nondominated_sort(combined_objectives)
        population = environmental_selection(combined_population, combined_objectives, combined_fronts, reference_points, pop_size)

    objectives = evaluate_population(population, functions)
    fronts = fast_nondominated_sort(objectives)
    pareto_front = [objectives[i] for i in fronts[0]]
    pareto_front.sort()

    return pareto_front
