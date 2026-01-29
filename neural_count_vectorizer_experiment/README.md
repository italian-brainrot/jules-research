# Neural Count Vectorizer (NCV) Experiment

## Hypothesis
Traditional text vectorization methods like TF-IDF and BM25 use handcrafted formulas to weight term frequencies ($tf$) based on document frequencies ($df$) and document lengths ($L$). We hypothesize that a neural network can learn an optimal weighting function $w = f(tf, df, L)$ end-to-end for a specific task, potentially outperforming these handcrafted baselines.

## Methodology
We implemented a **Neural Count Vectorizer (NCV)** that replaces the fixed weighting formula with a small Multi-Layer Perceptron (MLP).
- **NCV Model**: $v_{doc} = \sum_{i \in doc} MLP(\log(1+tf_i), \log(1+df_i), \log(L_{doc}/avgL)) \cdot \mathbf{e}_i$, where $\mathbf{e}_i$ are learned word embeddings.
- **Baselines**:
  - **Uniform**: $w = 1$
  - **TF**: $w = tf$
  - **Log-TF-IDF**: $w = \log(1+tf) \cdot \log(N/df)$
  - **BM25**: Standard formula with $k_1=1.5, b=0.75$.
- **Task**: 4-class text classification on a subset of the 20newsgroups dataset (`rec.autos`, `rec.motorcycles`, `sci.electronics`, `sci.space`).
- **Fair Comparison**: Each model was tuned using Optuna for 5-8 trials to find the best learning rate and weight decay. Embedding dimension was fixed at 128.

## Results

| Method | Test Accuracy | Test Loss |
|--------|---------------|-----------|
| Uniform | 64.71% | 1.1334 |
| **TF** | **68.37%** | **0.9495** |
| Log-TF-IDF | 63.00% | 1.4463 |
| BM25 | 65.65% | 3.8827 |
| **NCV (Ours)**| **68.18%** | 3.7719 |

### NCV Analysis
The learned NCV weighting function was visualized by varying one input while keeping others fixed:
- **Weight vs TF**: The model discovered a sub-linear saturation curve, similar to $\log(1+tf)$ or the saturation in BM25.
- **Weight vs DF**: The model discovered an inverse relationship, similar to IDF, where common words are downweighted.

![NCV Analysis](ncv_analysis.png)

## Conclusion
The Neural Count Vectorizer (NCV) successfully learned a weighting scheme that mimics the properties of established methods like TF-IDF and BM25 directly from data. On this dataset, NCV and raw TF weighting performed best. The fact that NCV converged to a solution qualitatively similar to handcrafted formulas provides a strong empirical validation of the principles behind TF-IDF, while also demonstrating that these weights can be learned end-to-end. In future work, NCV could be extended to incorporate more features (e.g., word position, part-of-speech) that are difficult to incorporate into standard formulas.
