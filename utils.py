import pandas as pd
import numpy as np
import sys
from typing import List, Tuple


def load_and_align(
    data_matrix_path: str,
    sample_meta_path: str,
    var_meta_path:    str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load 3 TSV files 

    Retourne:
        dm : DataFrame (variables x samples)
        sm : DataFrame (samples x metadata)
        vm : DataFrame (variables x metadata)
    """
    dm = pd.read_csv(data_matrix_path, sep="\t", index_col=0)
    sm = pd.read_csv(sample_meta_path, sep="\t", index_col=0)
    vm = pd.read_csv(var_meta_path,    sep="\t", index_col=0)


    if sm.index.intersection(dm.columns).empty and not sm.index.intersection(dm.index).empty:
        print(
            "INFO : DataMatrix détectée comme samples x variables. Transposition automatique.",
            file=sys.stderr,
        )
        dm = dm.T

    common_samples = sm.index.intersection(dm.columns)
    if common_samples.empty:
        sys.exit(
            "ERREUR : Aucun échantillon commun entre DataMatrix et SampleMeta.\n"
            f"DataMatrix colonnes (5 premiers) : {dm.columns[:5].tolist()}\n"
            f"SampleMeta index   (5 premiers) : {sm.index[:5].tolist()}"
        )

    dm = dm.reindex(columns=common_samples).reindex(index=vm.index)
    dm = dm.dropna(axis=1, how="all").dropna(axis=0, how="all")
    sm = sm.loc[dm.columns]
    vm = vm.loc[dm.index]

    missing_samples = [s for s in sm.index if s not in dm.columns]
    missing_vars    = [v for v in vm.index if v not in dm.index]

    if missing_samples:
        print(
            f"AVERTISSEMENT : {len(missing_samples)} échantillon(s) absent(s) supprimé(s).",
            file=sys.stderr
        )
    if missing_vars:
        print(
            f"AVERTISSEMENT : {len(missing_vars)} variable(s) absente(s) supprimée(s).",
            file=sys.stderr
        )

    print(
        f"Alignement : {dm.shape[1]} échantillons x {dm.shape[0]} variables.",
        file=sys.stderr
    )
    return dm, sm, vm


def extract_targets(sm: pd.DataFrame, response_col: str) -> List[str]:
    """
    Analyse responseCol (noms ou indices basés sur 1, séparés par des virgules).
    Retourne une liste ordonnée de noms de colonnes.
    """
    items    = [i.strip() for i in str(response_col).split(",") if i.strip()]
    selected = []

    for item in items:
        if item.isdigit():
            idx = int(item) - 2
            if 0 <= idx < len(sm.columns):
                selected.append(sm.columns[idx])
            else:
                sys.exit(
                    f"ERREUR : Index '{item}' hors limites "
                    f"(sampleMeta a {len(sm.columns)} colonnes)."
                )
        elif item in sm.columns:
            selected.append(item)
        else:
            sys.exit(
                f"ERREUR : Colonne cible '{item}' introuvable dans sampleMeta.\n"
                f"Colonnes disponibles : {sm.columns.tolist()}"
            )

    target_names = list(dict.fromkeys(selected))
    if not target_names:
        sys.exit(f"ERREUR : Aucune colonne cible valide (reçu : '{response_col}').")

    print(f"Colonnes cibles : {target_names}", file=sys.stderr)
    return target_names


def validate_partition_col(sm: pd.DataFrame, partition_col: str) -> pd.Series:
    """
    Valide que la colonne de partition existe et contient les libellés attendus.
    Retourne une série normalisée (en minuscules, espaces supprimés).
    """
    if partition_col not in sm.columns:
        sys.exit(
            f"ERREUR : Colonne de partition '{partition_col}' introuvable.\n"
            f"Colonnes disponibles : {sm.columns.tolist()}"
        )

    labels       = sm[partition_col].astype(str).str.strip().str.lower()
    non_excluded = set(labels.unique()) - {"excluded", "x_val", "super_test", "nan"}
    is_holdout   = non_excluded.issubset({"train", "test"})
    is_kfold     = all(lbl.isdigit() for lbl in non_excluded)

    if not is_holdout and not is_kfold:
        sys.exit(
            f"ERREUR : Colonne '{partition_col}' contient des valeurs inattendues : "
            f"{sorted(non_excluded)}\n"
            f"Attendu : 'train'/'test' (holdout) ou entiers 1,2,... (K-Fold/LOOCV)."
        )
    return labels
