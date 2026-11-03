import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut


# ─────────────────────────────────────────────
# Utilitaire : binning adaptatif robuste
# ─────────────────────────────────────────────

def get_n_bins(y):
    """Nombre de bins initial selon la règle de Sturges."""
    n = len(y)
    if n < 4:
        return 2
    return int(np.floor(np.log2(n) + 1))



def safe_bin(y, min_samples_per_bin=2):
    n_bins = get_n_bins(y)

    while n_bins >= 2:
        bins = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
        counts = bins.value_counts()

        if len(counts) >= 2 and counts.min() >= min_samples_per_bin:
            return bins

        n_bins -= 1

    raise ValueError(
        "Stratification impossible : impossible de créer au moins 2 bins "
        f"avec au moins {min_samples_per_bin} échantillons par bin."
    )


def get_stratify_arg(y, min_samples_per_bin=2):
    """
    Retourne l'argument de stratification adapté :
    - y numérique ET plus de 10 valeurs uniques → bins adaptatifs via safe_bin
    - Sinon (catégoriel) → y directement
    """
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return safe_bin(y)
    return y


# ─────────────────────────────────────────────
# 1. Holdout simple
# ─────────────────────────────────────────────

def split_holdout(X, y, test_size=0.2, shuffle=True, random_state=42):
    """Holdout sans stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    return [(X_train, X_test, y_train, y_test)]


# ─────────────────────────────────────────────
# 2. Holdout stratifié
# ─────────────────────────────────────────────

def split_holdout_strat(X, y, test_size=0.2, shuffle=True, random_state=42):
    """
    Holdout stratifié.
    - y continu  → bins adaptatifs (toujours au moins 2 échantillons/bin)
    - y catégoriel → stratification directe
    """
    if not shuffle:
        raise ValueError(
            "Holdout stratifié impossible avec shuffle=False. "
            "La stratification nécessite shuffle=True."
        )
    
    stratify_arg = get_stratify_arg(y, min_samples_per_bin=2)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state, shuffle=shuffle
        )
        return [(X_train, X_test, y_train, y_test)]
    except ValueError as e:
        n_classes = stratify_arg.nunique()
        min_test_size = round(n_classes / len(y), 2)

        raise ValueError(
            f"Stratification impossible : test_size={test_size} est trop petit.\n"
            f"ou une classe/bin contient trop peu d'échantillons. "
            f"→ {n_classes} classes/bins détectés\n"
            f"→ testSize minimum recommandé : {min_test_size}"
        )


# ─────────────────────────────────────────────
# 3. K-Fold simple
# ─────────────────────────────────────────────

def split_kfold(X, y, n_splits=5, shuffle=True, random_state=42):
    """K-Fold sans stratification."""
    if shuffle:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=False)

    return [
        (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
        for tr, te in kf.split(X)
    ]


# ─────────────────────────────────────────────
# 4. K-Fold stratifié
# ─────────────────────────────────────────────


def split_kfold_strat(X, y, n_splits=5, shuffle=True, random_state=42):
    """
    K-Fold stratifié.
    - y continu  → bins adaptatifs (toujours au moins 2 échantillons/bin)
    - y catégoriel → stratification directe
    """
    stratify_arg = get_stratify_arg(y, min_samples_per_bin=2)
    if shuffle:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    try:
        return [
            (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
            for tr, te in skf.split(X, stratify_arg)
        ]
    except ValueError as e:
        counts = stratify_arg.value_counts()
        max_n_splits = int(counts.max())

        raise ValueError(
            f"K-Fold stratifié impossible : n_splits={n_splits} est trop grand "
            f"pour la plus petite classe/bin.\n"
            f"→ Valeur maximale autorisée : {max_n_splits}\n" 
            f"Détail : {e}"
        )
    

# ─────────────────────────────────────────────
# 5. LOOCV
# ─────────────────────────────────────────────


"""
def split_loocv(X, y, **kwargs):
    
    Leave-One-Out Cross-Validation.
    Chaque échantillon est utilisé une fois comme ensemble de test (taille 1),
    tandis que les échantillons restants forment l'ensemble d'apprentissage (taille n-1).
    
    loo = LeaveOneOut()
    
    # Retourne une liste de n tuples (X_train, X_test, y_train, y_test)
    return [
        (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
        for tr, te in loo.split(X)
    ]
"""

def split_loocv(X, y, **kwargs):
    """
    Leave-One-Out Cross-Validation.
    Chaque échantillon est utilisé une fois comme ensemble de test (taille 1),
    tandis que les échantillons restants forment l'ensemble d'apprentissage (taille n-1).
    Sécurisée : 
    - Crash direct (Error) si >= 50 échantillons.
    - Message d'avertissement (Warning) si >= 20 et < 50 échantillons.
    """
    n_samples = len(X)


    if n_samples >= 50:
        sys.exit(
            f"ERREUR CRITIQUE : Leave-One-Out (LOOCV) refusé.\n"
            f"Votre jeu de données contient {n_samples} échantillons. Le LOOCV est limité à "
            f"un maximum de 49 échantillons pour éviter de saturer le serveur.\n"
            f"Veuillez choisir une autre méthode de split (ex: K-Fold ou Holdout) dans l'interface."
        )


    elif n_samples >= 20:
        print(
            f"ATTENTION [WARNING] : Vous utilisez LOOCV sur un jeu de {n_samples} échantillons.\n"
            f"Le script va devoir entraîner {n_samples} modèles différents en coulisses. "
            f"Le temps d'exécution de l'outil Galaxy risque d'être plus long que d'habitude.",
            file=sys.stderr  
        )


    loo = LeaveOneOut()
    
    return [
        (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
        for tr, te in loo.split(X)
    ]

