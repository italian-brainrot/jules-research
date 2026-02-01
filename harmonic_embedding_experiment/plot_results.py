import json
import matplotlib.pyplot as plt

def plot_results():
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("results.json not found")
        return

    methods = list(results.keys())
    accs = [results[m]["best_acc"] for m in methods]
    params = [results[m]["params"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy', color=color)
    bars = ax1.bar(methods, accs, color=color, alpha=0.6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Parameters', color=color)
    ax2.plot(methods, params, color=color, marker='o', linestyle='-', linewidth=2, label='Parameters')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Comparison of Embedding Methods')
    fig.tight_layout()
    plt.savefig('comparison.png')
    print("Plot saved as comparison.png")

if __name__ == "__main__":
    plot_results()
