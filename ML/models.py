"""
models.py --> Model Factory 
"""

import sys
import os
import numpy as np
from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Imports pour la Régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Imports pour la Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOL_DIR)


class BaseModelBuilder(ABC):
    """
    Classe mère abstraite pour configurer les pipelines.
    Centralise le scaling, les plis de validation croisée et la graine aléatoire.
    """
    def __init__(
        self,
        model_type: str,
        scale_data: bool = True,
        cv_folds: int = 5,
        **kwargs,
    ):
        self.model_type = str(model_type).lower()
        self.scale_data = scale_data
        self.cv_folds = int(cv_folds)
        self.random_state = kwargs.get("random_state", 42)

    def build(self):
        steps = []

        # Si self.scale_data est True (calculé automatiquement par MLmanager),
        # on applique le StandardScaler. Sinon, on passe directement au modèle.
        if self.scale_data:
            steps.append(("scaler", StandardScaler()))
            
        steps.append(("model", self.build_estimator()))
        return Pipeline(steps)

    @abstractmethod
    def build_estimator(self):
        pass


# ==========================================
#  CONSTRUCTEURS DE RÉGRESSION
# ==========================================

class RegressionLinearBuilder(BaseModelBuilder):
    def build_estimator(self):
        return LinearRegression()


class RegressionLassoBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_lasso = Lasso(random_state=self.random_state, max_iter=10000)
        param_grid = {
            "alpha": list(np.logspace(-4, 4, 50)),
        }
        return GridSearchCV(
            estimator=base_lasso,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring="r2",
            n_jobs=-1
        )


class RegressionRidgeBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_ridge = Ridge(random_state=self.random_state)
        param_grid = {
            "alpha": list(np.logspace(-4, 4, 50)),
        }
        return GridSearchCV(
            estimator=base_ridge,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring="r2",
            n_jobs=-1
        )


class RegressionElasticNetBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_en = ElasticNet(random_state=self.random_state)
        param_grid = {
            "alpha": list(np.logspace(-4, 4, 30)),
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        }
        return GridSearchCV(
            estimator=base_en,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring="r2",
            n_jobs=-1
        )


class RegressionRandomForestBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_rf = RandomForestRegressor(random_state=self.random_state)
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 5, 10, None],
        }
        return GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=self.cv_folds, scoring="r2", n_jobs=-1)


class RegressionDecisionTreeBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_tree = DecisionTreeRegressor(random_state=self.random_state)
        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [2, 5, 10, 20]
        }
        return GridSearchCV(estimator=base_tree, param_grid=param_grid, cv=self.cv_folds, scoring="r2", n_jobs=-1)


class RegressionSVRBuilder(BaseModelBuilder):
    def build_estimator(self):
        # FIX 2 : SVR n'accepte pas de random_state
        base_svr = SVR()
        param_grid = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2]
        }
        return GridSearchCV(estimator=base_svr, param_grid=param_grid, cv=self.cv_folds, scoring="r2", n_jobs=-1)


class RegressionKNNBuilder(BaseModelBuilder):
    def build_estimator(self):
        base_knn = KNeighborsRegressor()
        param_grid = {
            "n_neighbors": [3, 5, 7, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
        return GridSearchCV(estimator=base_knn, param_grid=param_grid, cv=self.cv_folds, scoring="r2", n_jobs=-1)


# ==========================================
#  CONSTRUCTEURS DE CLASSIFICATION
# ==========================================

class ClassificationLinearBuilder(BaseModelBuilder):
    def build_estimator(self): 
        return LogisticRegression(penalty=None, max_iter=10000, random_state=self.random_state)
    

class ClassificationLassoBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_lr = LogisticRegression(penalty='l1', solver="saga", random_state=self.random_state, max_iter=10000)
        param_grid = {
            "C": list(10.0 ** np.arange(-4, 5))
        }
        return GridSearchCV(estimator=base_lr, param_grid=param_grid, cv=self.cv_folds, scoring='f1_macro', n_jobs=-1)


class ClassificationRidgeBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_lr = LogisticRegression(penalty='l2', random_state=self.random_state)
        param_grid = {
            "C": list(10.0 ** np.arange(-4, 5))
        }
        return GridSearchCV(estimator=base_lr, param_grid=param_grid, cv=self.cv_folds, scoring='f1_macro', n_jobs=-1)


class ClassificationElasticNetBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_lr = LogisticRegression(penalty='elasticnet', solver="saga", random_state=self.random_state)
        param_grid = {
            "C": list(10.0 ** np.arange(-4, 5)),
            "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
        }
        return GridSearchCV(estimator=base_lr, param_grid=param_grid, cv=self.cv_folds, scoring='f1_macro', n_jobs=-1)


class ClassificationRandomForestBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_rf = RandomForestClassifier(random_state=self.random_state)
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [2, 5, 10, 15],
            "max_features": ["sqrt", "log2", 0.1, 0.2]
        }
        return GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=self.cv_folds, scoring="f1_macro", n_jobs=-1)


class ClassificationDecisionTreeBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_tree = DecisionTreeClassifier(random_state=self.random_state)
        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [2, 5, 10, 20]
        }
        return GridSearchCV(estimator=base_tree, param_grid=param_grid, cv=self.cv_folds, scoring="f1_macro", n_jobs=-1)


class ClassificationSVCBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_svc = SVC()
        param_grid = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10, 100]
        }
        return GridSearchCV(estimator=base_svc, param_grid=param_grid, cv=self.cv_folds, scoring="f1_macro", n_jobs=-1)


class ClassificationKNNBuilder(BaseModelBuilder):
    def build_estimator(self): 
        base_knn = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 5, 7, 11, 15, 21],
            "weights": ["uniform", "distance"],
        }
        return GridSearchCV(estimator=base_knn, param_grid=param_grid, cv=self.cv_folds, scoring="f1_macro", n_jobs=-1)


MODEL_REGISTRY = {
    "regression": {
        "linear": RegressionLinearBuilder,
        "ridge": RegressionRidgeBuilder,
        "l2": RegressionRidgeBuilder,
        "lasso": RegressionLassoBuilder,
        "l1": RegressionLassoBuilder,
        "elasticnet": RegressionElasticNetBuilder,
        "randomforest": RegressionRandomForestBuilder,
        "rf": RegressionRandomForestBuilder,
        "decisiontree": RegressionDecisionTreeBuilder,
        "tree": RegressionDecisionTreeBuilder,
        "svr": RegressionSVRBuilder,
        "svm": RegressionSVRBuilder,
        "knn": RegressionKNNBuilder,
    },
    "classification": {
        "linear": ClassificationLinearBuilder,
        "logistic": ClassificationLinearBuilder, 
        "ridge": ClassificationRidgeBuilder,
        "l2": ClassificationRidgeBuilder,
        "lasso": ClassificationLassoBuilder,
        "l1": ClassificationLassoBuilder,
        "elasticnet": ClassificationElasticNetBuilder,
        "randomforest": ClassificationRandomForestBuilder,
        "rf": ClassificationRandomForestBuilder,
        "decisiontree": ClassificationDecisionTreeBuilder,
        "tree": ClassificationDecisionTreeBuilder,
        "svr": ClassificationSVCBuilder,
        "svm": ClassificationSVCBuilder,  
        "knn": ClassificationKNNBuilder,
    }
}


def get_model(
    task_type: str = "regression",
    model_type: str = "ridge",
    scale_data: bool = True,
    cv_folds: int = 5,
    **kwargs,
):
    """Point d'entrée de la factory appelé par le MLManager pour obtenir le pipeline."""
    task_type = str(task_type).lower()
    model_type = str(model_type).lower()

    if task_type not in MODEL_REGISTRY:
        sys.exit(f"ERREUR : task_type inconnu : {task_type}.")

    if model_type not in MODEL_REGISTRY[task_type]:
        sys.exit(f"ERREUR : model_type '{model_type}' non supporté pour task_type='{task_type}'.")

    return MODEL_REGISTRY[task_type][model_type](
        model_type=model_type,
        scale_data=scale_data,
        cv_folds=cv_folds,
        **kwargs,
    ).build()