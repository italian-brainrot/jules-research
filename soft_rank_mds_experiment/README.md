# Soft Rank MDS: Differentiable Non-metric Multidimensional Scaling

## Hypothesis
Differentiable Non-metric MDS using soft ranks (SoftRank-MDS) provides a better preservation of the ordinal structure of high-dimensional distances compared to traditional metric MDS and PCA. By optimizing for the rank of distances rather than their exact values, the method should be more robust to the non-linearities and density variations common in high-dimensional datasets.

## Methodology
- **Dataset**: MNIST-1D (300 samples, 40 features).
- **Proposed Method (Soft Rank MDS)**:
    - Target: Hard ranks of the pairwise distance matrix in high-dimensional space (row-wise).
    - Loss: Mean Squared Error between high-dimensional hard ranks and low-dimensional soft ranks.
    - Soft Ranks: Computed using a sigmoid-based differentiable rank transformation: $r_{ij} = \sum_k \sigma(\alpha(d_{ij} - d_{ik}))$.
    - $\alpha$ is a learnable temperature parameter.
    - Optimization: Adam optimizer for 500 steps, initialized with PCA coordinates.
- **Baselines**:
    - **PCA**: Linear dimensionality reduction.
    - **t-SNE**: Neighborhood-based non-linear dimensionality reduction (focused on local structure).
    - **Metric MDS**: Stress minimization ($\sum (D_{ij} - d_{ij})^2$).

## Results

### Quantitative Metrics
| Method | KNN Accuracy (k=5) | Spearman Correlation |
| :--- | :--- | :--- |
| PCA | 0.3833 | 0.7753 |
| t-SNE | 0.5750 | 0.6623 |
| Metric MDS | 0.4333 | 0.7588 |
| **Soft Rank MDS** | **0.4542** | **0.8661** |

- **Spearman Correlation**: Soft Rank MDS achieves the highest Spearman correlation (0.8661) among all tested methods. This indicates that it preserves the relative ordering of distances significantly better than Metric MDS and PCA.
- **Clustering (KNN Acc)**: Soft Rank MDS (0.4542) outperforms PCA (0.3833) and Metric MDS (0.4333) in terms of cluster separation in the 2D space, although it is still behind t-SNE (0.5750), which is explicitly designed to emphasize local clusters.

### Visualizations
- [PCA](pca.png)
- [t-SNE](t-sne.png)
- [Metric MDS](metric_mds.png)
- [Soft Rank MDS](soft_rank_mds.png)

## Conclusion
The experiment demonstrates that Soft Rank MDS is a powerful tool for preserving the global ordinal structure of high-dimensional data. By using a differentiable approximation of the rank function, we can optimize for Non-metric MDS objectives using standard gradient-based optimizers. Soft Rank MDS provides a superior balance between preserving global distance rankings (highest Spearman correlation) and maintaining local cluster structure (better KNN accuracy than PCA and Metric MDS).
