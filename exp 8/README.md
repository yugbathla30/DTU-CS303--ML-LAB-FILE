# Experiment 8: Support Vector Machines (SVM) and the Kernel Trick

## Overview
This experiment explores Support Vector Machines (SVMs) and the kernel trick. The notebook demonstrates training linear and non-linear SVM classifiers, explains why a linear model may fail on non-linearly separable data, and uses GridSearchCV to tune hyperparameters for the RBF kernel.

## Objectives
- Introduce SVM concepts (margin, support vectors, C parameter)
- Show limitations of a linear SVM on non-linearly separable data
- Demonstrate kernel methods (RBF, polynomial)
- Perform hyperparameter tuning using GridSearchCV
- Visualize decision boundaries, confusion matrices and model comparison

## Dataset
- The notebook uses a synthetically generated dataset created with scikit-learn's `make_moons` function (a non-linearly separable 2D dataset). This is ideal to demonstrate the kernel trick and decision boundaries.

Key properties:
- Samples: 500 (as generated in the notebook)
- Labels: 2 classes (Class 0 / Class 1)

## Notebook Steps (high level)
1. Generate moons dataset and visualize the raw data.
2. Split data into training and validation sets (70/30 split).
3. Standardize features with `StandardScaler`.
4. Train a linear SVM and evaluate (`SVC(kernel='linear')`). Observe its limitations on the moons dataset.
5. Train non-linear SVMs: RBF (default) and Polynomial (degree=3).
6. Use `GridSearchCV` to tune RBF hyperparameters (`C` and `gamma`) with 5-fold CV.
7. Evaluate the best model on validation data, show confusion matrix and classification report.
8. Visualize decision boundaries for linear, default RBF, and tuned RBF models.

## Hyperparameter Search
The notebook uses a parameter grid similar to:

```
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100],
    'kernel': ['rbf']
}
```

GridSearchCV is configured with 5-fold cross validation and `accuracy` as the scoring metric.

## Visualizations & Outputs
- Scatter plot of the raw moons dataset
- Confusion matrices for each model
- Classification reports (precision, recall, f1-score)
- Decision boundary plots (linear, RBF default, RBF tuned)
- Model comparison bar chart showing validation accuracies

## Typical Results / Notes
- Linear SVM: performs poorly on the moons dataset because the data is not linearly separable.
- RBF SVM: usually achieves much better performance by producing non-linear decision boundaries.
- GridSearchCV: helps find good `C` and `gamma` values; final accuracy and best params are printed in the notebook.

See the notebook `SVM.ipynb` for exact numeric results and plots generated during the run.

## Dependencies
The notebook requires the following Python packages:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install with pip if needed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

If you use the project virtual environment in `ml_lab/`, activate it before installing or running the notebook.

## How to run
1. Activate your virtual environment (if using `ml_lab/`):

```bash
# macOS / Linux (zsh or bash)
source "ml_lab/bin/activate"

# Windows (PowerShell)
# .\ml_lab\Scripts\Activate.ps1
```

2. Install dependencies (if not present):

```bash
pip install -r requirements.txt  # if you maintain one
# or install packages directly:
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Open the notebook and run all cells:

```bash
jupyter notebook "exp 8/SVM.ipynb"
```

4. Recommended workflow: Restart kernel and Run → Run All to ensure consistent outputs.
