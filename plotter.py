import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

lato = FontProperties(fname='Lato-Regular.ttf')

def plot_graph(data, title, x, xlabel, ylabel, **y):
    sns.set_style("whitegrid")
    csfont = {'font': lato}

    plt.rcParams['figure.figsize'] = [8, 6]
    plt.title(title, fontsize=26)

    for i, (ydata, yname) in enumerate(y.items()):
        sns.lineplot(data, x=x, y=ydata, label=yname, linewidth=2.5, color=sns.color_palette()[i])

    plt.xlabel(xlabel, fontsize=28, **csfont)
    plt.ylabel(ylabel, fontsize=28, **csfont)

    plt.xticks(fontsize=24, **csfont)
    plt.yticks(fontsize=24, **csfont)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.legend(title="Legend", fontsize=24, title_fontsize=26)
    plt.tight_layout()

    plt.show()

plt.rcParams["font.family"] = "cursive"
df = pd.read_csv("./material_prediction_trial.csv")

plot_graph(df, "Loss against epochs", "Epoch", "Epoch", "Loss", **{"Train Loss": "Train Loss", "Test Loss": "Test Loss"})

plot_graph(df, "Albedo Loss against epochs", "Epoch", "Epoch", "Albedo Loss", **{"Train Albedo Loss": "Train Loss", "Test Albedo Loss": "Test Loss"})

plot_graph(df, "Metallic Loss against epochs", "Epoch", "Epoch", "Metallic Loss", **{"Train Metallic Loss": "Train Loss", "Test Metallic Loss": "Test Loss"})

plot_graph(df, "Smoothness Loss against epochs", "Epoch", "Epoch", "Smoothness Loss", **{"Train Smoothness Loss": "Train Loss", "Test Smoothness Loss": "Test Loss"})

plot_graph(df, "Render Loss against epochs", "Epoch", "Epoch", "Render Loss", **{"Train Render Loss": "Train Loss", "Test Render Loss": "Test Loss"})
