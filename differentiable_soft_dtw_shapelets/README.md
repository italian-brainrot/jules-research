# Differentiable Soft-DTW Shapelet Learning

This experiment explores the use of **Soft Dynamic Time Warping (Soft-DTW)** as a distance metric in a differentiable shapelet learning framework for 1D signal classification.

## Hypothesis
Standard shapelet learning uses Euclidean distance to compare local signal windows with learned prototypes. However, signals often exhibit temporal warping (dilations or compressions) that Euclidean distance cannot capture. We hypothesize that using a **Soft-DTW** distance layer will allow the model to learn shapelets that are invariant to local temporal distortions, potentially improving classification performance on signals with such characteristics.

## Methodology

### 1. Soft-DTW Shapelet Layer
- **Core Algorithm**: We implemented a differentiable version of Dynamic Time Warping using the "Soft-DTW" approach, where the `min` operation in the DP recursion is replaced by a `soft-min` (Log-Sum-Exp).
- **Implementation**: The DP table calculation was optimized using `torch.jit.script` to mitigate the overhead of nested loops in Python.
- **Layer Structure**:
    - A sliding window unfolds the input signal.
    - Each window is compared to a set of learnable shapelets using Soft-DTW.
    - Soft-min pooling is applied over the windows to find the best match for each shapelet.

### 2. Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Models Compared**:
    - **MLP Baseline**: A standard 3-layer MLP.
    - **Euclidean Shapelet Network**: Uses squared Euclidean distance for shapelet matching.
    - **Soft-DTW Shapelet Network**: Uses the proposed Soft-DTW distance.
- **Hyperparameter Tuning**: Learning rates were tuned for each model using Optuna. Due to the high computational cost of Soft-DTW, its configuration was simplified (fewer shapelets, larger stride) to allow for feasible training times.

## Results

| Model | Test Accuracy | Best Learning Rate |
| :--- | :---: | :--- |
| **MLP Baseline** | 76.40% | 0.00191 |
| **Euclidean Shapelet Network** | **89.70%** | 0.00402 |
| **Soft-DTW Shapelet Network** | 18.35% | 0.00048 |

## Analysis
- **Euclidean Performance**: The Euclidean shapelet network performed exceptionally well, outperforming the MLP baseline significantly. This confirms that shapelet-based feature extraction is highly effective for `mnist1d`.
- **Soft-DTW Challenges**: The Soft-DTW model performed poorly in this specific configuration. Several factors likely contributed:
    - **Computational Complexity**: To make training feasible, we had to use a large stride (8) and fewer shapelets (16). This likely resulted in significant information loss and a lack of discriminative power.
    - **Optimization Difficulty**: The Soft-DTW landscape is more complex than the Euclidean one. The combination of a soft-min over windows and a soft-min within DTW might lead to vanishing or unstable gradients, especially with a short training regime.
    - **Dataset Fit**: `mnist1d` may not contain the types of elastic temporal warping where DTW truly shines. If the signals are mostly aligned or only shifted (which the sliding window already handles), the extra flexibility of DTW might just introduce noise.

## Visualizations
- `comparison.png`: Accuracy comparison across models.
- `learned_shapelets_euclidean.png`: Motifs learned by the Euclidean model.
- `learned_shapelets_soft_dtw.png`: Motifs learned by the Soft-DTW model.

## Conclusion
While Differentiable Shapelet Learning is a powerful technique, the direct application of Soft-DTW in the matching layer remains computationally expensive and difficult to optimize for real-time training. The Euclidean variant remains a strong and efficient baseline for signal classification tasks like `mnist1d`. Future work could focus on more efficient approximations of DTW or applying it to datasets with known elastic deformations.
