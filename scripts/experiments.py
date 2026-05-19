import matplotlib.pyplot as plt


def plot_fitness_history(history, avg_history=None, title='GA — fitness over generations'):
    """Plots best (and optionally average) RMSE per generation."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history, label='Best RMSE', linewidth=1.5)
    if avg_history:
        ax.plot(avg_history, label='Avg RMSE', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
