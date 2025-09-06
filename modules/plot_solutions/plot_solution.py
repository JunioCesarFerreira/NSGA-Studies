import numpy as np
import matplotlib.pyplot as plt

def plot_solutions(pareto_front):
    """
    Plota gráficos paralelos lado a lado e o gráfico de dispersão da fronteira de Pareto para 2 objetivos.
    
    :param pareto_front: Lista ou array de pontos da fronteira de Pareto em 2D.
    """
    # Normalizar os valores dos objetivos para [0, 1]
    def normalize(solutions):
        solutions = np.array(solutions)
        return (solutions - solutions.min(axis=0)) / (solutions.max(axis=0) - solutions.min(axis=0))

    # Normalizar as soluções
    normalized_solutions = normalize(pareto_front)

    # Configuração dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico paralelo 1
    for sol in normalized_solutions:
        axes[0].plot(np.arange(normalized_solutions.shape[1]), sol, marker='o', alpha=0.7)
    axes[0].set_xticks(np.arange(normalized_solutions.shape[1]))
    axes[0].set_xticklabels([f"Objetivo {i+1}" for i in range(normalized_solutions.shape[1])])
    axes[0].set_xlabel("Objetivos")
    axes[0].set_ylabel("Valores Normalizados")
    axes[0].set_title("Gráfico Paralelo - Soluções Normalizadas")
    axes[0].grid(alpha=0.3)

    # Gráfico de dispersão da fronteira de Pareto
    axes[1].scatter(*zip(*pareto_front), label='Fronteira de Pareto', color='blue', alpha=0.7)
    axes[1].set_xlabel("f1(x)")
    axes[1].set_ylabel("f2(x)")
    axes[1].set_title("Pontos na Fronteira de Pareto")
    axes[1].grid(alpha=0.3)
    axes[1].axis("equal")
    axes[1].legend()

    # Ajustar layout
    plt.tight_layout()
    plt.show()