# ML4M – Machine Learning for Metabolomics

ML4M is a Python-based tool developed during an internship at INRAE (UMR BFP, META team) to facilitate supervised machine learning analysis of metabolomics data.

Integrated into the **Galaxy platform** using **Planemo**, ML4M enables users to train, validate, and evaluate machine learning models for phenotype prediction from high-dimensional metabolomics datasets, without requiring programming skills.

It implements several machine learning algorithms, including Linear and Logistic Regression, LASSO, Ridge, Elastic Net, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees.

The tool supports multiple data partitioning and validation strategies, including Holdout, Stratified Holdout, K-Fold, Stratified K-Fold, and Leave-One-Out Cross-Validation (LOOCV), as well as hyperparameter optimization using GridSearchCV.

### Input Data

ML4M requires three input files:

- **Data Matrix:** Contains quantitative metabolomics data, with samples as rows and metabolic variables as columns.
- **Sample Metadata:** Contains sample-related information, including the phenotype or response variable to be predicted.
- **Variable Metadata:** Provides annotations and information describing the metabolic variables (features) included in the data matrix.

Users select the response variable and configure the machine learning algorithm, data partitioning strategy, and validation parameters through the Galaxy interface.

### Outputs

ML4M trains and evaluates the selected models and generates several outputs:

- Prediction results.
- Model performance metrics.
- Model coefficients (when applicable).
- A summary report.

**Technologies:** Python, Galaxy, Planemo, Linux.
