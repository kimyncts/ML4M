import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def regression_plot(y_true, y_pred, output_file="regression.png", ax=None):
    """
    Trace un graphique de régression.
    - Si 'ax' est fourni, le tracé se fait dans cet axe.
    - Si 'ax' est None, une nouvelle figure est créée et sauvegardée.
    """
    # 1. Dessiner sur l'axe fourni ou l'axe actif (gca)
    sns.regplot(x=y_true, y=y_pred, ci=None, ax=ax)
    
    # 2. Configurer les labels via l'objet axe
    target_ax = ax if ax is not None else plt.gca()
    target_ax.set_xlabel("True values")
    target_ax.set_ylabel("Predicted values")
    target_ax.set_title("Regression predictions")
    
    # 3. Sauvegarde automatique uniquement si on n'est pas dans une grille
    if ax is None:
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def confusion_matrix_plot(y_true, y_pred, output_file="confusion_matrix.png", ax=None):
    """
    Trace une matrice de confusion.
    - Si 'ax' est fourni, le tracé se fait dans cet axe.
    - Si 'ax' est None, une nouvelle figure est créée et sauvegardée.
    """
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 1. Dessiner sur l'axe fourni ou l'axe actif (gca)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    # 2. Configurer les labels via l'objet axe
    target_ax = ax if ax is not None else plt.gca()
    target_ax.set_xlabel("Predicted")
    target_ax.set_ylabel("True")
    target_ax.set_title("Confusion Matrix")
    
    # 3. Sauvegarde automatique uniquement si on n'est pas dans une grille
    if ax is None:
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()