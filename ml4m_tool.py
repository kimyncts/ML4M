import pandas as pd
import numpy as np
import sys
import argparse
import os
from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split

## AJOUT DE MATPLOTLIB ICI ##
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
############################

from ML.visualization import (
    regression_plot,
    confusion_matrix_plot,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ML import MLManager
except ImportError as e:
    sys.exit(f"ERREUR : ML package introuvable → {e}")

try:
    import split
except ImportError as e:
    sys.exit(f"ERREUR : split.py introuvable → {e}")

try:
    from utils import load_and_align, extract_targets
except ImportError as e:
    sys.exit(f"ERREUR : utils.py introuvable → {e}")


class Ml4mPartitionInputs(BaseModel):
    model_config = ConfigDict(extra='ignore')

    # Files
    dataMatrix:    str
    sampleMeta:    str
    varMeta:       str
    responseCol:   str
    partitionMode: str

    # Outputs
    sampleOut:     str
    reportOut:     str
    metricsOut:      str
    coefficientsOut: str
    visualizationOut: str

    # Partition
    advMode:           str           = "default"
    masterSeed:        int           = 42
    seedList:          Optional[str] = None
    selectionMode:     str           = "all"
    selectedSamples:   str           = "NONE"
    selectedRole:      str           = "split_set"
    superTestMode:     str           = "none"
    superTestSamples:  str           = "NONE"
    superTestSize:     float         = 0.1
    splitMethod:       str           = "holdout_strat"
    testSize:          float         = 0.2
    nSplits:           int           = 5
    shuffle:           str           = "True"

    # ML --> task + model
    taskType:       str = "regression"
    modelType:      str = "linear"
    scaleData:      str = "True"
 
    # ML --> regularization
    regularization: str = "none"       
    alphaMode:      str = "automatic"  
    alpha:          float = 1.0

    # ML --> hidden/default params
    fitIntercept: str = "True"
    l1Ratio:      float = 0.5
    maxIter:      int = 5000
    cvFolds:      int = 5

    # ML --> repetitions
    nRepeats:       int = 1
    


class PartitionTool:
    def __init__(self, inputs: Ml4mPartitionInputs):
        self.inputs         = inputs
        self.dm             = None
        self.sm             = None
        self.vm             = None
        self.target_names   = []
        self.final_metadata = None
        self.shuffle_bool   = (
            True if inputs.advMode == "default"
            else inputs.shuffle == "True"
        )

    def _resolve_seeds(self) -> List[int]:
        """Retourne une liste de graines correspondant au nombre de répétitions."""
        if self.inputs.advMode == "default":
            # Mode auto : génère nRepeats graines aléatoires
            rng = np.random.default_rng()
            return rng.integers(0, 100000, size=self.inputs.nRepeats).tolist()

        if not self.inputs.seedList:
            sys.exit("ERREUR : seedList requis en mode manuel.")

        try:
            values = [int(x.strip()) for x in self.inputs.seedList.split(",") if x.strip()]
        except ValueError:
            sys.exit("ERREUR : seedList doit contenir des entiers.")

        # ICI : On vérifie par rapport au nombre de répétitions
        if len(values) != self.inputs.nRepeats:
            sys.exit(f"ERREUR : Tu as demandé {self.inputs.nRepeats} répétitions, mais fourni {len(values)} graines.")
            
        return values

    def _resolve_active_ids(self) -> List[str]:
        all_ids = self.sm.index.tolist()
        if self.inputs.selectionMode == "all":
            return all_ids

        selected = [
            x.strip() for x in self.inputs.selectedSamples.split(",")
            if x.strip() and x not in ("NONE", "ALL")
        ]
        valid = [i for i in selected if i in all_ids]
        if not valid:
            sys.exit("ERREUR : Aucun échantillon valide sélectionné.")
        return valid

    def _resolve_super_test(self, valid_active, y, seed, phenotype):
        mode = self.inputs.superTestMode
        if mode == "none":
            return [], valid_active

        if mode == "manual":
            st_list = [x.strip() for x in self.inputs.superTestSamples.split(",") if x.strip() and x != "NONE"]
            st_ids = [i for i in valid_active if i in st_list]
            if not st_ids:
                sys.exit(f"ERREUR : Aucun ID super test valide pour '{phenotype}'.")
            remain = [i for i in valid_active if i not in st_ids]
            return st_ids, remain

        if mode == "stratified":
            y_aligned = y.loc[valid_active]
            try:
                stratify_arg = split.get_stratify_arg(y_aligned)
                remain_idx, st_idx = train_test_split(
                    y_aligned.index, test_size=self.inputs.superTestSize,
                    random_state=int(seed), shuffle=True, stratify=stratify_arg
                )
                return list(st_idx), list(remain_idx)
            except ValueError as e:
                sys.exit(f"ERREUR : Super test stratifié impossible pour '{phenotype}' : {e}")

        sys.exit(f"ERREUR : superTestMode inconnu : '{mode}'")

    def _apply_split_algo(self, X, y, col, seed):
        method = self.inputs.splitMethod
        if method == "loo":
            folds = split.split_loocv(X, y)
            for i, (_, X_te, _, _) in enumerate(folds):
                self.final_metadata.loc[X_te.index, col] = str(i + 1)
        elif method in ["holdout", "holdout_strat"]:
            func  = split.split_holdout_strat if "strat" in method else split.split_holdout
            folds = func(X, y, test_size=self.inputs.testSize, shuffle=self.shuffle_bool, random_state=int(seed))
            X_tr, X_te, _, _ = folds[0]
            self.final_metadata.loc[X_tr.index, col] = "train"
            self.final_metadata.loc[X_te.index, col] = "test"
        elif method in ["kfold", "kfold_strat"]:
            func  = split.split_kfold_strat if "strat" in method else split.split_kfold
            folds = func(X, y, n_splits=self.inputs.nSplits, shuffle=self.shuffle_bool, random_state=int(seed))
            for i, (_, X_te, _, _) in enumerate(folds):
                self.final_metadata.loc[X_te.index, col] = str(i + 1)

    def process_phenotype(self, phenotype, seed, suffix=""):
    # Utilisation du suffixe pour nommer la colonne (ex: split_phenotype1_r1)
        col_label = f"split_{phenotype}{suffix}" 
        self.final_metadata[col_label] = "excluded"

        if self.inputs.partitionMode == "manual":
            if "partition" not in self.sm.columns:
                sys.exit("ERREUR : Colonne 'partition' manquante en mode manuel.")
            for sid in self.sm.index:
                if pd.notna(self.sm.loc[sid, phenotype]):
                    self.final_metadata.loc[sid, col_label] = str(self.sm.loc[sid, "partition"]).lower()
            return

        active_ids = self.final_metadata[self.final_metadata["analysis_set"] == "included"].index.tolist()
        valid_active = [i for i in active_ids if pd.notna(self.sm.loc[i, phenotype])]

        if not valid_active:
            sys.exit(f"ERREUR : Aucun échantillon valide pour '{phenotype}'.")

        if self.inputs.selectedRole == "test":
            self.final_metadata.loc[valid_active, col_label] = "test"
            others = [i for i in self.sm.index if i not in valid_active and pd.notna(self.sm.loc[i, phenotype])]
            self.final_metadata.loc[others, col_label] = "train"
            return

        y_active = self.sm.loc[valid_active, phenotype]
        super_ids, pool_ids = self._resolve_super_test(valid_active, y_active, seed, phenotype)

        if super_ids:
            self.final_metadata.loc[super_ids, col_label] = "x_val"
        self._apply_split_algo(self.dm.T.loc[pool_ids], self.sm.loc[pool_ids, phenotype], col_label, seed)

    def write_report(self, seeds_by_phenotype):
        rows = []
        for phenotype, seed in seeds_by_phenotype.items():
            rows.append({
                "target_column":      phenotype,
                "seed":               seed,
                "shuffle":            self.shuffle_bool,
                "adv_mode":           self.inputs.advMode,
                "partition_mode":     self.inputs.partitionMode,
                "selection_mode":     self.inputs.selectionMode,
                "selected_role":      self.inputs.selectedRole,
                "split_method":       self.inputs.splitMethod,
                "test_size":          self.inputs.testSize,
                "n_splits":           self.inputs.nSplits,
                "super_test_mode":    self.inputs.superTestMode,
                "super_test_size":    self.inputs.superTestSize,
                "selected_samples":   self.inputs.selectedSamples,
                "super_test_samples": self.inputs.superTestSamples,
                "n_repeats":          self.inputs.nRepeats,
            })
        pd.DataFrame(rows).to_csv(self.inputs.reportOut, sep="\t", index=False)

    def _build_ml_kwargs(self) -> dict:
        return {
            "regularization": self.inputs.regularization,
            "alpha_mode": self.inputs.alphaMode,
            "alpha": self.inputs.alpha,
            "l1_ratio": self.inputs.l1Ratio,
            "cv_folds": self.inputs.cvFolds,
        }
    

    def _extract_coefficients(self, model, X_columns):
        current = model
        
        # 1. Si Pipeline, extraire l'étape nommée "model"
        if hasattr(current, "named_steps") and "model" in current.named_steps:
            current = current.named_steps["model"]
        
        # 2. Si GridSearchCV, extraire le best_estimator_
        if hasattr(current, "best_estimator_"):
            current = current.best_estimator_
            
        # 3. Si c'est encore un pipeline, on prend la dernière étape
        if hasattr(current, "named_steps"):
            steps = list(current.named_steps.values())
            current = steps[-1]

        # 4. Extraction
        if hasattr(current, "coef_"):
            return np.ravel(current.coef_), "coefficient"
        if hasattr(current, "feature_importances_"):
            return np.ravel(current.feature_importances_), "feature_importance"
            
        return None, None

    
    def run(self):
        # 1. Préparation des données
        self.dm, self.sm, self.vm = load_and_align(
            self.inputs.dataMatrix, self.inputs.sampleMeta, self.inputs.varMeta
        )
        self.target_names = extract_targets(self.sm, self.inputs.responseCol)
        seeds_for_repeats = self._resolve_seeds()

        # Initialisation propre des métadonnées
        self.final_metadata = self.sm[self.target_names].copy()
        self.final_metadata["analysis_set"] = "excluded"
        active_ids = self._resolve_active_ids()
        self.final_metadata.loc[active_ids, "analysis_set"] = "included"

        ml_kwargs = self._build_ml_kwargs()
        X = self.dm.T
        self.final_metadata = self.final_metadata.loc[X.index]
        
        all_metrics, all_coefs = [], []
        n_repeats = self.inputs.nRepeats

        # 2. Boucle ML principale
        for target in self.target_names:
            print(f"DEBUG: Traitement de la cible {target}", flush=True)
            target_y_tests, target_y_preds = [], []

            for repeat_idx in range(n_repeats):
                current_seed = seeds_for_repeats[repeat_idx]
                suffix = f"_r{repeat_idx + 1}"
                split_col = f"split_{target}{suffix}"
                
                print(f"--- Répétition {repeat_idx + 1}/{n_repeats} | Seed: {current_seed} ---", flush=True)
                
                # Génération du split (cette fonction doit remplir la colonne split_col)
                self.process_phenotype(target, current_seed, suffix=suffix)
                
                # Récupération des indices
                idx_xval = self.final_metadata.index[self.final_metadata[split_col] == "x_val"].tolist()
                
                # Configuration du modèle
                active_model_type = self.inputs.modelType
                if active_model_type in ["linear", "logistic"] and self.inputs.regularization != "none":
                    translation_map = {
                        "l1": "lasso" if self.inputs.taskType == "regression" else "logistic",
                        "l2": "ridge" if self.inputs.taskType == "regression" else "logistic",
                        "elasticnet": "elasticnet", "lasso": "lasso", "ridge": "ridge"
                    }
                    active_model_type = translation_map.get(self.inputs.regularization, active_model_type)

                ml = MLManager(task_type=self.inputs.taskType, model_type=active_model_type, masterSeed=current_seed, **ml_kwargs)

                # Gestion des splits
                if self.inputs.splitMethod in ["holdout", "holdout_strat"]:
                    idx_train = self.final_metadata.index[self.final_metadata[split_col] == "train"].tolist()
                    idx_test  = self.final_metadata.index[self.final_metadata[split_col] == "test"].tolist()
                    X_train, y_train = X.loc[idx_train], self.final_metadata.loc[idx_train, target]
                    X_test, y_test   = X.loc[idx_test], self.final_metadata.loc[idx_test, target]
                else:
                    idx_pool = self.final_metadata.index[~self.final_metadata[split_col].isin(["excluded", "x_val"])].tolist()
                    X_train, y_train = X.loc[idx_pool], self.final_metadata.loc[idx_pool, target]
                    X_test, y_test   = pd.DataFrame(columns=X.columns), pd.Series(dtype=float)

                # Entraînement
                list_run_results = ml.train_and_evaluate(
                    X_train=X_train, y_train=y_train, 
                    X_test=X_test if len(X_test) > 0 else X_train,
                    y_test=y_test if len(y_test) > 0 else y_train,
                    feature_names=X.columns.tolist()
                )
                

                y_test_current = y_test if len(X_test) > 0 else y_train
                y_pred_current = list_run_results[0]["y_pred"] if "y_pred" in list_run_results[0] else ml.model.predict(X_test if len(X_test) > 0 else X_train)
                
                target_y_tests.append(y_test_current) 
                target_y_preds.append(y_pred_current)

                # Dans votre boucle run() :
                trained_model = list_run_results[0].get("model", ml.model)

                # Utiliser ce modèle entraîné pour l'extraction
                coefs, coef_type = self._extract_coefficients(trained_model, X.columns)
                # ----------------------


                if coefs is not None:
                    for feature, val in zip(X.columns.tolist(), coefs):
                        all_coefs.append({
                            "target_column": target,
                            "repeat_id": repeat_idx + 1,
                            "model_type": active_model_type,
                            "feature": feature,
                            "coefficient": float(val)
                        })
                else:
                    print(f"DEBUG: Impossible d'extraire des coefficients pour {active_model_type}", file=sys.stderr)
                    
                    # DEBUG : Vérifions la nature de l'objet
                    print(f"DEBUG: Type de trained_model : {type(trained_model)}", file=sys.stderr)
                    if hasattr(trained_model, "named_steps"):
                        print(f"DEBUG: Étapes du pipeline : {list(trained_model.named_steps.keys())}", file=sys.stderr)
                    if hasattr(trained_model, "best_estimator_"):
                        print(f"DEBUG: best_estimator_ présent : Oui", file=sys.stderr)
                    else:
                        print(f"DEBUG: best_estimator_ présent : Non", file=sys.stderr)

                # Collecte métriques
                for run_results in list_run_results:
                    metrics_data = {"target_column": target, "repeat_id": repeat_idx + 1, "model_type": run_results["model_type"], "evaluation_set": run_results["evaluation_set"]}
                    if self.inputs.taskType == "regression":
                        metrics_data.update({k: run_results.get(k) for k in ["r2", "rmse", "mse", "mae"]})
                    else:
                        metrics_data.update({"accuracy": run_results.get("accuracy"), "f1": run_results.get("f1"), "auc_roc": run_results.get("auc_roc") or run_results.get("auc")})
                    all_metrics.append(metrics_data)

                # Évaluation x_val
                if len(idx_xval) > 0:
                    y_pred_xval = ml.model.predict(X.loc[idx_xval])
                    xval_scores = ml.metrics_engine.compute(y_true=self.final_metadata.loc[idx_xval, target], y_pred=y_pred_xval)
                    xval_data = {"target_column": target, "repeat_id": repeat_idx + 1, "model_type": active_model_type, "evaluation_set": "x_val"}
                    xval_data.update({k: xval_scores.get(k) for k in (["r2", "rmse", "mse", "mae"] if self.inputs.taskType == "regression" else ["accuracy", "f1", "auc_roc"])})
                    all_metrics.append(xval_data)

            # 3. Génération Graphique
            #base_out, ext = os.path.splitext(self.inputs.visualizationOut)
            #target_visualization_out = f"{base_out}_{target}{ext}"
            target_visualization_out = self.inputs.visualizationOut

            fig, axes = plt.subplots(1, n_repeats, figsize=(5 * n_repeats, 5))
            if n_repeats == 1: 
                axes = [axes]
            for i in range(n_repeats):
                if self.inputs.taskType == "classification":
                    confusion_matrix_plot(y_true=target_y_tests[i], y_pred=target_y_preds[i], ax=axes[i])
                else:
                    regression_plot(y_true=target_y_tests[i], y_pred=target_y_preds[i], ax=axes[i])
                axes[i].set_title(f"Répétition {i+1}\nSeed: {seeds_for_repeats[i]}")

            plt.tight_layout()
            plt.savefig(target_visualization_out, format="png", bbox_inches="tight")
            plt.close(fig)

            if os.path.exists(target_visualization_out):
                print(f"DEBUG: Fichier {target_visualization_out} créé avec succès ({os.path.getsize(target_visualization_out)} octets)")
            

        print(f"DEBUG coefficients collectés = {len(all_coefs)}", file=sys.stderr)



        # 4. Finalisation

        

        pd.DataFrame(all_metrics).to_csv(self.inputs.metricsOut, sep="\t", index=False)
        pd.DataFrame(all_coefs).to_csv(self.inputs.coefficientsOut, sep="\t", index=False)
        print("Succès.", file=sys.stderr)

        self.write_report(dict(zip(self.target_names, [seeds_for_repeats]*len(self.target_names))))


        # On lit le rapport généré, on modifie la colonne 'seed', on réécrit
        try:
            report_df = pd.read_csv(self.inputs.reportOut, sep="\t")
            seeds_str = ";".join(map(str, seeds_for_repeats))
            report_df['seed'] = seeds_str
            report_df.to_csv(self.inputs.reportOut, sep="\t", index=False)
        except Exception as e:
            print(f"Note: Impossible de mettre à jour le rapport manuellement : {e}")


        self.final_metadata.to_csv(self.inputs.sampleOut, sep="\t", index=True)

        


def main():
    parser = argparse.ArgumentParser(description="ML4M Partition + ML Tool")
    parser.add_argument('--dataMatrix',       required=True)
    parser.add_argument('--sampleMeta',       required=True)
    parser.add_argument('--varMeta',          required=True)
    parser.add_argument('--responseCol',      required=True)
    parser.add_argument('--sampleOut',        required=True)
    parser.add_argument('--reportOut',        required=True)
    parser.add_argument('--partitionMode',    default='auto')
    parser.add_argument('--metricsOut',       required=True)
    parser.add_argument('--coefficientsOut',  required=True)
    parser.add_argument('--visualizationOut', required=True)
    parser.add_argument('--advMode',          default='default')
    parser.add_argument('--seedList',         default=None)
    parser.add_argument('--masterSeed',       type=int,   default=42)
    parser.add_argument('--selectionMode',    default='all')
    parser.add_argument('--selectedSamples',  default='NONE')
    parser.add_argument('--selectedRole',     default='split_set')
    parser.add_argument('--superTestMode',    default='none')
    parser.add_argument('--superTestSamples', default='NONE')
    parser.add_argument('--superTestSize',    type=float, default=0.1)
    parser.add_argument('--splitMethod',      default='holdout_strat')
    parser.add_argument('--testSize',         type=float, default=0.2)
    parser.add_argument('--nSplits',          type=int,   default=5)
    parser.add_argument('--shuffle',          default='True')
    parser.add_argument('--taskType',       default='regression')
    parser.add_argument('--modelType',      default='linear')
    parser.add_argument('--scaleData',      default='True')
    parser.add_argument('--regularization', default='none')
    parser.add_argument('--alphaMode',      default='automatic')
    parser.add_argument('--alpha',          type=float, default=1.0)
    parser.add_argument('--fitIntercept',   default='True')
    parser.add_argument('--l1Ratio',        type=float, default=0.5)
    parser.add_argument('--maxIter',        type=int,   default=5000)
    parser.add_argument('--cvFolds',        type=int,   default=5)
    parser.add_argument('--nRepeats', type=int, default=1, help="Nombre de répétitions globales du pipeline ML")

    args, _ = parser.parse_known_args()
    input_data = Ml4mPartitionInputs(**vars(args))
    PartitionTool(input_data).run()

if __name__ == "__main__":
    main()