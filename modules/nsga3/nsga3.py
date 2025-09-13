import random
import numpy as np
from typing import Callable, Optional, Sequence, DefaultDict

def nsga3_func(
    pop_size: int,
    generations: int,
    bounds: list[tuple[float, float]],
    functions: list[Callable[[np.ndarray], float]] | Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    mutation: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
    initial_pop: Optional[list[np.ndarray]] = None,
    divisions: int = 10,
    ref_points: Optional[np.ndarray] = None
) -> list[tuple[float, ...]]:
    """
    NSGA-III generalizado para N dimensões.

    :param pop_size: Tamanho da população
    :param generations: Número de gerações
    :param bounds: Lista de tuplas [(min1, max1), (min2, max2), ...] definindo os limites para cada dimensão
    :param functions: Lista de funções objetivo [f1, f2, ..., fM] ou única função multiobjetivo f(x) -> np.ndarray
    :param crossover: Função de crossover que aceita dois pais e retorna filhos
    :param mutation: Função de mutação que aceita um indivíduo e retorna um indivíduo mutado
    :param divisions: Número de divisões para geração dos pontos de referência
    :return: Fronteira de Pareto da última geração
    """

    def initialize_population(size: int, bounds: list[tuple[float, float]]) -> list[np.ndarray]:
        return [
            np.array([random.uniform(b[0], b[1]) for b in bounds], dtype=float)
            for _ in range(size)
        ]

    def evaluate_population(
        population: Sequence[np.ndarray],
        functions: list[Callable[[np.ndarray], float]] | Callable[[np.ndarray], np.ndarray]
    ) -> list[tuple[float, ...]]:
        """
        Avalia a população em dois modos:
        - Lista de funções objetivos: [f1, f2, ..., fM], cada uma retornando float.
        - Única função multiobjetivo: f(x) -> np.ndarray com M objetivos.
        """
        objectives: list[tuple[float, ...]] = []

        if isinstance(functions, list):
            # Modo antigo: lista de funções escalar
            for x in population:
                obj = tuple(f(x) for f in functions)
                objectives.append(obj)
        elif callable(functions):
            # Novo modo: função que retorna np.ndarray
            for x in population:
                obj_vec = functions(x)
                if isinstance(obj_vec, np.ndarray):
                    objectives.append(tuple(float(v) for v in obj_vec))
                else:
                    raise ValueError("A função multiobjetivo deve retornar um np.ndarray")
        else:
            raise ValueError("Parâmetro 'functions' inválido")

        return objectives

    def dominates(obj1: tuple[float, ...], obj2: tuple[float, ...]) -> bool:
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    def fast_nondominated_sort(objectives: Sequence[tuple[float, ...]]) -> list[list[int]]:
        population_size: int = len(objectives)
        S: list[list[int]] = [[] for _ in range(population_size)]
        n: list[int] = [0] * population_size
        fronts: list[list[int]] = [[]]

        for p in range(population_size):
            for q in range(population_size):
                if dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)

        i: int = 0
        while fronts[i]:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    def generate_reference_points(M: int, p: int) -> np.ndarray:
        def generate_recursive(
            points: list[list[float]],
            num_objs: int,
            left: int,
            total: int,
            depth: int,
            current_point: list[float]
        ) -> None:
            if depth == num_objs - 1:
                current_point.append(left / total)
                points.append(current_point.copy())
                current_point.pop()
            else:
                for i in range(left + 1):
                    current_point.append(i / total)
                    generate_recursive(points, num_objs, left - i, total, depth + 1, current_point)
                    current_point.pop()

        points: list[list[float]] = []
        generate_recursive(points, M, p, p, 0, [])
        return np.array(points, dtype=float)

    def environmental_selection(
        population: list[np.ndarray],
        objectives: list[tuple[float, ...]],
        fronts: list[list[int]],
        reference_points: np.ndarray,
        pop_size: int
    ) -> list[np.ndarray]:
        next_population_indices: list[int] = []
        for front in fronts:
            if len(next_population_indices) + len(front) <= pop_size:
                next_population_indices.extend(front)
            else:
                N: int = pop_size - len(next_population_indices)
                selected_indices: list[int] = niching_selection(front, objectives, reference_points, N)
                next_population_indices.extend(selected_indices)
                break
        next_population: list[np.ndarray] = [population[i] for i in next_population_indices]
        return next_population

    def niching_selection(
        front: list[int],
        objectives: list[tuple[float, ...]],
        reference_points: np.ndarray,
        N: int
    ) -> list[int]:
        selected: list[int] = []
        objs: np.ndarray = np.array([objectives[i] for i in front], dtype=float)
        ideal_point: np.ndarray = np.min(objs, axis=0)
        normalized_objs: np.ndarray = objs - ideal_point

        max_values: np.ndarray = np.max(normalized_objs, axis=0)
        max_values[max_values == 0] = 1
        normalized_objs = normalized_objs / max_values

        associations: list[tuple[int, int, float]] = []
        for idx, obj in zip(front, normalized_objs):
            distances: np.ndarray = np.linalg.norm(obj - reference_points, axis=1)
            min_index: int = int(np.argmin(distances))
            associations.append((idx, min_index, float(distances[min_index])))

        reference_associations: DefaultDict[int, list[tuple[int, float]]] = DefaultDict(list)
        for idx, ref_idx, dist in associations:
            reference_associations[ref_idx].append((idx, dist))

        niche_counts: dict[int, int] = {i: 0 for i in range(len(reference_points))}
        selected_flags: dict[int, bool] = {idx: False for idx in front}

        while len(selected) < N:
            min_niche_count: int = min(niche_counts.values()) if niche_counts else 0
            min_refs: list[int] = [ref for ref, count in niche_counts.items() if count == min_niche_count]

            for ref_idx in min_refs:
                assoc_inds: list[tuple[int, float]] = reference_associations.get(ref_idx, [])
                unselected_inds: list[tuple[int, float]] = [(idx, dist) for idx, dist in assoc_inds if not selected_flags[idx]]

                if unselected_inds:
                    unselected_inds.sort(key=lambda x: x[1])
                    selected_idx: int = unselected_inds[0][0]
                    selected.append(selected_idx)
                    selected_flags[selected_idx] = True
                    niche_counts[ref_idx] += 1
                    break
            else:
                remaining: list[int] = [idx for idx in front if not selected_flags[idx]]
                if remaining:
                    selected_idx = random.choice(remaining)
                    selected.append(selected_idx)
                    selected_flags[selected_idx] = True
                else:
                    break

        return selected[:N]

    def compute_individual_ranks(fronts: list[list[int]]) -> dict[int, int]:
        individual_ranks: dict[int, int] = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                individual_ranks[idx] = rank
        return individual_ranks

    def tournament_selection(population: list[np.ndarray], individual_ranks: dict[int, int]) -> np.ndarray:
        i1, i2 = random.sample(range(len(population)), 2)
        rank1: int = individual_ranks[i1]
        rank2: int = individual_ranks[i2]
        if rank1 < rank2:
            return population[i1]
        elif rank2 < rank1:
            return population[i2]
        else:
            return population[random.choice([i1, i2])]

    # Inicializa a população
    if initial_pop is None:
        population: list[np.ndarray] = initialize_population(pop_size, bounds)
    else:
        population = initial_pop

    # Descobre número de objetivos M
    if isinstance(functions, list):
        M: int = len(functions)
    elif callable(functions):
        test_obj = functions(np.zeros(len(bounds)))
        if not isinstance(test_obj, np.ndarray):
            raise ValueError("A função multiobjetivo deve retornar np.ndarray")
        M: int = test_obj.shape[0]
    else:
        raise ValueError("Parâmetro 'functions' inválido")

    if ref_points is None:
        ref_points = generate_reference_points(M, divisions)
    else:
        ref_points = np.asarray(ref_points, dtype=float)

    for gen in range(generations):
        objectives: list[tuple[float, ...]] = evaluate_population(population, functions)
        fronts: list[list[int]] = fast_nondominated_sort(objectives)
        individual_ranks: dict[int, int] = compute_individual_ranks(fronts)
        offspring_population: list[np.ndarray] = []
        while len(offspring_population) < pop_size:
            parent1: np.ndarray = tournament_selection(population, individual_ranks)
            parent2: np.ndarray = tournament_selection(population, individual_ranks)
            children: tuple[np.ndarray, np.ndarray] = crossover(parent1, parent2)
            child: np.ndarray = mutation(children[0], bounds)
            offspring_population.append(child)

        combined_population: list[np.ndarray] = population + offspring_population
        combined_objectives: list[tuple[float, ...]] = evaluate_population(combined_population, functions)
        combined_fronts: list[list[int]] = fast_nondominated_sort(combined_objectives)
        population = environmental_selection(combined_population, combined_objectives, combined_fronts, ref_points, pop_size)

    objectives = evaluate_population(population, functions)
    fronts = fast_nondominated_sort(objectives)
    pareto_front: list[tuple[float, ...]] = [objectives[i] for i in fronts[0]]
    pareto_front.sort()

    return pareto_front
