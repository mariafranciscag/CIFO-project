import matplotlib.pyplot as plt

def plot_fitess_evolution(history):
    plt.plot(history["best_fitness"], label="Best RMSE")
    plt.plot(history["avg_fitness"], label="Avg RMSE", linestyle="--")
    plt.title(f"Evolution of fitness")
    plt.xlabel("Generation")
    plt.ylabel("RMSE")
    plt.show