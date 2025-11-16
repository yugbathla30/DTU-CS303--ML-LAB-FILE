# Experiment 9: Implementing a Neural Network and Backpropagation from Scratch

## Objectives
- Implement forward and backward propagation with ReLU and Sigmoid activations
- Implement Binary Cross-Entropy (BCE) and Mean Squared Error (MSE) losses
- Train multiple architectures and compare performance
- Visualize training loss curves and compare implementations with sklearn

## Dataset
- Dataset: Breast Cancer Wisconsin (available from scikit-learn via `load_breast_cancer()`)
- Problem: Binary classification (malignant vs benign)
- Features: 30 numeric features

## Notebook Structure 
1. Import libraries and load the Breast Cancer dataset
2. Train/validation split (70/30) and standardize features with `StandardScaler`
3. Implement utility functions: activation functions (sigmoid, ReLU) and their derivatives, loss functions (BCE, MSE)
4. Implement `MyANNClassifier` — a from-scratch feedforward neural network class with:
   - parameter initialization
   - forward propagation
   - backward propagation
   - parameter updates with vanilla gradient descent
   - fit and predict methods
5. Train and evaluate three custom models:
   - Model 1: BCE loss, 1 hidden layer
   - Model 2: MSE loss, 1 hidden layer
   - Model 3: BCE loss, 2 hidden layers
6. Train and evaluate scikit-learn's `MLPClassifier` for comparison
7. Plot loss curves and save `loss_curves.png`
8. Present a results summary table with accuracy, precision, recall, and F1-score

## Implementation Details
- Activation functions:
  - Sigmoid for output layer (binary probability)
  - ReLU for hidden layers
- Loss functions:
  - Binary Cross-Entropy (BCE) — recommended for binary classification
  - Mean Squared Error (MSE) — included for contrast
- Training:
  - Batch gradient descent (vanilla) with a fixed learning rate
  - Parameters initialized with small random values (seeded for reproducibility)
  - Training progress printed periodically

## Experiments & Findings
- Models trained with BCE typically outperform those trained with MSE for binary classification.
- A single hidden layer often performs well on this dataset; deeper models can help but require careful tuning.
- `MLPClassifier` (sklearn) usually converges faster and can achieve better performance due to built-in optimizers (Adam) and features like early stopping.

## Visualizations
- Loss curves comparing BCE vs MSE and all custom models
- Saved plot: `loss_curves.png`
- Final results table printed in the notebook comparing custom models and `MLPClassifier`

## Dependencies
The notebook requires the following packages (typical install):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

