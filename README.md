# ML4M - Machine Learning for Metabolomics

ML4M is a Python-based tool developed during an internship at INRAE (UMR BFP, META team) to facilitate supervised machine learning analysis of metabolomics data.

Integrated into the **Galaxy platform** using **Planemo**, ML4M enables users to train, validate, and evaluate machine learning models for phenotype prediction from high-dimensional metabolomics datasets, without requiring programming skills.

It implements multiple machine learning algorithms, including Linear and Logistic Regression, LASSO, Ridge, Elastic Net, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees.

The tool supports several data partitioning and validation strategies, including Holdout, Stratified Holdout, K-Fold, Stratified K-Fold, and Leave-One-Out Cross-Validation (LOOCV), as well as hyperparameter optimization using GridSearchCV.

**Technologies:** Python, scikit-learn, Galaxy, Planemo, Pydantic, Linux.
