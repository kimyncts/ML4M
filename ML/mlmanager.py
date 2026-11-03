"""
mlmanager.py --> Gestionnaire AutoML 
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from ML.models import get_model
from ML.metrics import get_metrics, BaseMetrics


class MLManager:
    def __init__(
        self,
        task_type: str = "regression",
        model_type: str = "linear",
        scale_data: bool = True,
        **model_kwargs,
    ):
        self.task_type  = task_type.lower()
        self.model_type = model_type.lower()
        self.cv_folds_initial = model_kwargs.pop("cv_folds", 10)
        # Prise en compte de la graine passée par le script principal (ex: masterSeed)
        self.random_state = model_kwargs.pop("masterSeed", 42)
        self.model_kwargs = model_kwargs

        # Pas de scaling pour les arbres, obligatoire pour le reste
        if self.model_type in ["decisiontree", "tree", "randomforest", "rf"]:
            self.scale_data = False
        else:
            self.scale_data = True  # Obligatoire pour linéaire, SVM, KNN...

        self.metrics_engine: BaseMetrics = get_metrics(self.task_type)
        self._predictions = []
        self.model = None  

    def _get_base_pipeline(self, cv_folds: int):
        """
        Appelle l'usine 'models.py' pour récupérer un pipeline 
        neuf (Scaler + Modèle brut).
        """
        return get_model(
            task_type=self.task_type,
            model_type=self.model_type,
            scale_data=self.scale_data,
            cv_folds=cv_folds,
            random_state=self.random_state,
            **self.model_kwargs,
        )

    def _safe_cv_folds(self, y_train) -> int:
        """
        Vérifie si le nombre de plis demandé (cv_folds) est mathématiquement possible.
        Si ce n'est pas le cas, fait crasher proprement l'outil avec un message explicite.
        """
        requested = int(self.cv_folds_initial)

        if self.task_type == "classification":
            counts = y_train.value_counts()
            max_allowed = int(counts.min()) if not counts.empty else 1

            if max_allowed < 2:
                sys.exit(
                    "ERREUR CRITIQUE : Validation croisée impossible.\n"
                    f"Une de vos classes contient moins de 2 échantillons : {counts.to_dict()}.\n"
                    "Veuillez équilibrer votre jeu de données ou utiliser une autre méthode."
                )
                
            # crash si l'utilisateur demande plus de plis qu'il n'y a d'échantillons dans la classe minoritaire
            if requested > max_allowed:
                sys.exit(
                    f"ERREUR : Le nombre de plis demandé ({requested}) est trop grand.\n"
                    f"Votre classe la plus petite ne contient que {max_allowed} échantillons.\n"
                    f"Veuillez configurer un 'Internal CV folds' de maximum {max_allowed} dans l'interface."
                )
        else:
            max_allowed = int(len(y_train))
            if max_allowed < 2:
                sys.exit(
                    "ERREUR CRITIQUE : Validation croisée impossible.\n"
                    f"Le nombre d'échantillons total dans le jeu d'apprentissage est insuffisant ({max_allowed})."
                )

            # crash si l'utilisateur demande plus de plis qu'il n'y a de lignes au total
            if requested > max_allowed:
                sys.exit(
                    f"ERREUR : Le nombre de plis demandé ({requested}) est trop grand.\n"
                    f"Votre jeu de données d'apprentissage ne contient que {max_allowed} lignes.\n"
                    f"Veuillez configurer un 'Internal CV folds' de maximum {max_allowed} dans l'interface."
                )

        return requested
    
    def _get_best_params(self, fitted_model) -> dict:
        """ Extracteur d'hyperparamètres après entraînement adapté aux pipelines imbriqués. """
        params = {}
        
        # Détecter si le modèle est un Pipeline ou un GridSearchCV direct
        if hasattr(fitted_model, "named_steps"):
            actual_model = fitted_model.named_steps.get("model")
        else:
            actual_model = fitted_model

        # Si le modèle imbriqué dans le pipeline est un GridSearchCV
        if isinstance(actual_model, GridSearchCV):
            params["cv_best_score"] = float(actual_model.best_score_)
            for k, v in actual_model.best_params_.items():
                params[f"best_{k}"] = v
            return params

        # Cas de secours (ex: LassoCV)
        if hasattr(actual_model, "alpha_"):
            params["best_alpha"] = float(actual_model.alpha_)
        return params

    def _get_feature_importance(self, fitted_model, feature_names) -> dict:
        """ Extrait les coefficients ou importances des variables de manière agnostique. """
        importances = {}
        
        if hasattr(fitted_model, "named_steps"):
            actual_model = fitted_model.named_steps.get("model")
        else:
            actual_model = fitted_model

        # Si c'est un GridSearchCV, on doit aller chercher le modèle final dans 'best_estimator_'
        if isinstance(actual_model, GridSearchCV):
            estimator = actual_model.best_estimator_
        else:
            estimator = actual_model

        if estimator is None:
            return importances

        # Extraction des coefficients (Modèles Linéaires)
        if hasattr(estimator, "coef_"):
            coefs = estimator.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            for name, coef in zip(feature_names, coefs):
                importances[f"coef_{name}"] = float(coef)

        # Extraction des importances (Arbres)
        elif hasattr(estimator, "feature_importances_"):
            for name, imp in zip(feature_names, estimator.feature_importances_):
                importances[f"importance_{name}"] = float(imp)

        return importances

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list) -> list:
        """
        Entraîne sur le Train, et évalue séparément sur le Train et le Test 
        pour retourner une liste de deux dictionnaires de résultats.
        """
        safe_cv = self._safe_cv_folds(y_train)
        model_to_train = self._get_base_pipeline(cv_folds=safe_cv)
        
        # 1. Entraînement uniquement sur le TRAIN
        model_to_train.fit(X_train, y_train)
        
        # Sauvegarde du modèle entraîné pour le script principal
        self.model = model_to_train
        
        # 2. Récupération des métadonnées globales (communes)
        best_params = self._get_best_params(model_to_train)
        feat_importances = self._get_feature_importance(model_to_train, feature_names)
        
        results_sets = []

        # Détermination sécurisée et robuste de la présence de predict_proba
        actual_estimator = model_to_train.best_estimator_ if isinstance(model_to_train, GridSearchCV) else model_to_train
        if hasattr(actual_estimator, "named_steps") and actual_estimator.named_steps.get("model") is not None:
            has_proba = hasattr(actual_estimator.named_steps.get("model"), "predict_proba")
        else:
            has_proba = hasattr(actual_estimator, "predict_proba")

        # 3. Évaluation sur le jeu d'apprentissage (TRAIN)
        y_pred_train = model_to_train.predict(X_train)
        y_proba_train = model_to_train.predict_proba(X_train) if self.task_type == "classification" and has_proba else None
        scores_train = self.metrics_engine.compute(y_true=y_train, y_pred=y_pred_train, y_proba=y_proba_train)
        
        results_sets.append({
            "model_type": self.model_type,
            "evaluation_set": "train",
            **scores_train,
            **best_params,
            **feat_importances
        })

        # 4. Évaluation sur le jeu de validation indépendant (TEST)
        y_pred_test = model_to_train.predict(X_test)
        y_proba_test = model_to_train.predict_proba(X_test) if self.task_type == "classification" and has_proba else None
        scores_test = self.metrics_engine.compute(y_true=y_test, y_pred=y_pred_test, y_proba=y_proba_test)
        
        results_sets.append({
            "model_type": self.model_type,
            "evaluation_set": "test",
            **scores_test,
            **best_params,
            **feat_importances
        })
        
        return results_sets