import random
import numpy as np
from typing import Callable, Optional, Sequence

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

    def _adaptive_normalization(St_objs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalização adaptativa do NSGA-III (Deb & Jain, 2014).
        Estima o ponto ideal (z*) e a nadir via pontos extremos / interceptos do
        hiperplano, calculados sobre o conjunto S_t (fronteiras consideradas).

        :param St_objs: (S x M) objetivos de S_t
        :return: (ideal, intercepts) — ambos (M,)
        """
        M: int = St_objs.shape[1]
        ideal: np.ndarray = St_objs.min(axis=0)
        translated: np.ndarray = St_objs - ideal

        # Pontos extremos: minimizam a Achievement Scalarizing Function em cada eixo
        extreme_idx: list[int] = []
        for j in range(M):
            w = np.full(M, 1e-6, dtype=float)
            w[j] = 1.0
            asf = np.max(translated / w, axis=1)
            extreme_idx.append(int(np.argmin(asf)))
        extremes: np.ndarray = translated[extreme_idx]  # (M x M)

        # Interceptos do hiperplano que passa pelos extremos: sum(f/a) = 1
        try:
            b = np.linalg.solve(extremes, np.ones(M))
            if np.any(b <= 1e-12) or np.any(~np.isfinite(b)):
                raise np.linalg.LinAlgError
            intercepts = 1.0 / b
            if np.any(intercepts <= 1e-12) or np.any(~np.isfinite(intercepts)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            intercepts = translated.max(axis=0)  # fallback: nadir por máximo

        intercepts = np.where(intercepts < 1e-12, 1e-12, intercepts)
        return ideal, intercepts

    def _associate(norm_objs: np.ndarray, reference_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Associa cada solução normalizada à direção de referência mais próxima
        (menor distância perpendicular).

        :return: (assoc, dist) — índice da ref e distância perpendicular por solução
        """
        dirs = reference_points / (np.linalg.norm(reference_points, axis=1, keepdims=True) + 1e-32)
        proj = norm_objs @ dirs.T                                  # (S x K)
        proj_vec = proj[:, :, None] * dirs[None, :, :]             # (S x K x M)
        perp = np.linalg.norm(norm_objs[:, None, :] - proj_vec, axis=2)  # (S x K)
        assoc = np.argmin(perp, axis=1)
        dist = perp[np.arange(perp.shape[0]), assoc]
        return assoc, dist

    def environmental_selection(
        population: list[np.ndarray],
        objectives: list[tuple[float, ...]],
        fronts: list[list[int]],
        reference_points: np.ndarray,
        pop_size: int
    ) -> list[np.ndarray]:
        objs_all: np.ndarray = np.array(objectives, dtype=float)

        # Aceita fronteiras inteiras enquanto couberem
        selected: list[int] = []
        splitting: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
                if len(selected) == pop_size:
                    return [population[i] for i in selected]
            else:
                splitting = front
                break

        if not splitting:  # completou exatamente com fronteiras inteiras
            return [population[i] for i in selected[:pop_size]]

        # S_t = já selecionados + fronteira que será dividida
        St: list[int] = selected + splitting
        ideal, intercepts = _adaptive_normalization(objs_all[St])
        norm = (objs_all[St] - ideal) / intercepts
        assoc, dist = _associate(norm, reference_points)

        n_sel: int = len(selected)                 # membros já selecionados (F_1..F_{l-1})
        K: int = len(reference_points)
        niche_count: np.ndarray = np.zeros(K, dtype=int)
        for i in range(n_sel):
            niche_count[assoc[i]] += 1

        cand: list[bool] = [i >= n_sel for i in range(len(St))]  # candidatos = fronteira dividida
        chosen: list[int] = []
        while len(selected) + len(chosen) < pop_size:
            avail_refs = {assoc[i] for i in range(len(St)) if cand[i]}
            if not avail_refs:
                break
            min_nc = min(niche_count[r] for r in avail_refs)
            min_refs = [r for r in avail_refs if niche_count[r] == min_nc]
            j = random.choice(min_refs)
            members = [i for i in range(len(St)) if cand[i] and assoc[i] == j]
            if niche_count[j] == 0:
                pick = min(members, key=lambda i: dist[i])   # mais próximo da direção
            else:
                pick = random.choice(members)                # aleatório entre os associados
            chosen.append(St[pick])
            cand[pick] = False
            niche_count[j] += 1

        selected.extend(chosen)
        return [population[i] for i in selected[:pop_size]]

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
            # Mantém AMBOS os filhos do crossover. Usar apenas children[0] enviesava
            # cada gene para o menor dos pais (E[c1] ~ min), prejudicando a convergência.
            offspring_population.append(mutation(children[0], bounds))
            if len(offspring_population) < pop_size:
                offspring_population.append(mutation(children[1], bounds))

        combined_population: list[np.ndarray] = population + offspring_population
        combined_objectives: list[tuple[float, ...]] = evaluate_population(combined_population, functions)
        combined_fronts: list[list[int]] = fast_nondominated_sort(combined_objectives)
        population = environmental_selection(combined_population, combined_objectives, combined_fronts, ref_points, pop_size)

    objectives = evaluate_population(population, functions)
    fronts = fast_nondominated_sort(objectives)
    pareto_front: list[tuple[float, ...]] = [objectives[i] for i in fronts[0]]
    pareto_front.sort()

    return pareto_front
