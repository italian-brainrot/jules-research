# Stacking and Bagging: Synchronization and Fold Set Averaging Experiment

This experiment investigates the impact of two design choices in repeated stratified k-fold cross-validation for ensemble methods:
1. **Synchronization**: Whether all models (base and meta) use the same fold indices.
2. **Fold Set Averaging**: Whether out-of-fold (OOF) predictions are averaged across repeats before training the meta-model.

## Definitions

### Synchronization
- **Synchronized**: All base models and the meta-model use the exact same fold indices for all repeats.
- **Non-synchronized**: Each model uses its own random seed for generating fold indices, resulting in different splits.

### Fold Set Averaging (Stacking Only)
- **Averaged**: Base model OOF predictions are averaged across all repeats to form a single set of features. The meta-model is then trained using repeated k-fold on these averaged features.
- **Non-averaged**: The meta-model is trained separately on the OOF predictions of each repeat. The final meta-OOF predictions are averaged across the meta-models from all repeats.

## Experimental Setup
- **Datasets**: Breast Cancer, Wine, Digits, Iris (from sklearn).
- **Model Sets**:
  - `diverse`: RandomForest, SVC, LogisticRegression, KNeighborsClassifier.
  - `trees`: RandomForest, ExtraTrees, GradientBoostingClassifier.
- **Meta-Model**: LogisticRegression.
- **CV**: 3 repeats, 5 folds.
- **Metrics**: Accuracy, ROC AUC, Log Loss.

## Results Summary (Averaged across all datasets and model sets)

| Method | Sync | Avg | Accuracy | ROC AUC | Log Loss |
|--------|------|-----|----------|---------|----------|
| Bagging | False | N/A | 0.9738 | 0.9976 | 0.1177 |
| Bagging | True | N/A | **0.9751** | 0.9973 | 0.1183 |
| Stacking | False | False | 0.9730 | 0.9959 | **0.0741** |
| Stacking | False | True | 0.9705 | 0.9960 | 0.0754 |
| Stacking | True | False | 0.9730 | 0.9962 | 0.0753 |
| Stacking | True | True | 0.9750 | 0.9962 | 0.0751 |

## Key Findings

1. **Stacking vs. Bagging**: Stacking consistently achieved much lower **Log Loss** than Bagging. In terms of **Accuracy**, both methods performed similarly, with Bagging (Synchronized) and Stacking (Synchronized + Averaged) being the top performers.
2. **Best Stacking Configuration**: The **Synchronized + Averaged** configuration was the most robust stacking variant across different model sets and datasets.
3. **Effect of Synchronization**:
   - For **Stacking**, synchronization generally improved results or maintained performance compared to non-synchronized counterparts when averaging was used.
   - For **Bagging**, synchronization also showed a slight benefit in accuracy in this broader experiment, contrary to the initial smaller run.
4. **Effect of Fold Set Averaging**:
   - For Synchronized stacking, averaging fold sets (**Averaged**) performed better than training on individual repeats (**Non-averaged**).

## Detailed Summary by Model Set

### Diverse Set
| Method | Sync | Avg | Accuracy | ROC AUC | Log Loss |
|--------|------|-----|----------|---------|----------|
| Bagging | True | N/A | 0.9781 | 0.9978 | 0.1017 |
| Stacking | True | True | **0.9790** | 0.9970 | **0.0632** |

### Trees Set
| Method | Sync | Avg | Accuracy | ROC AUC | Log Loss |
|--------|------|-----|----------|---------|----------|
| Bagging | True | N/A | **0.9720** | 0.9967 | 0.1348 |
| Stacking | True | True | 0.9710 | 0.9954 | **0.0871** |

## Visualizations
- `stacking_accuracy_by_set.png`: Accuracy comparison for stacking variants across datasets and model sets.
- `overall_accuracy_box.png`: Accuracy distribution for all methods.

## Conclusion
For Stacking, **Synchronized** folds combined with **Fold Set Averaging** provides the best and most consistent results. While Bagging can sometimes achieve comparable or even slightly better accuracy on simple datasets, Stacking offers superior probability calibration (as seen in Log Loss).
