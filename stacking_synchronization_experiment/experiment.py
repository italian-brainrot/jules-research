import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

# Define different sets of models
MODEL_SETS = {
    'diverse': [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))),
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier()))
    ],
    'trees': [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
}

def get_meta_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))

def get_dataset(name):
    if name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'wine':
        data = load_wine()
    elif name == 'digits':
        data = load_digits()
    elif name == 'iris':
        data = load_iris()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data.data, data.target, len(np.unique(data.target))

def run_experiment(dataset_name, model_set_name, n_repeats=3, n_folds=5):
    X, y, n_classes = get_dataset(dataset_name)
    base_models = MODEL_SETS[model_set_name]

    results = []

    for sync in [True, False]:
        print(f"    Sync={sync}...")
        # 1. Generate Base OOFs
        base_oofs_list = []
        for b_idx, (name, model) in enumerate(base_models):
            seed = 42 if sync else 42 + b_idx
            rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_folds, random_state=seed)

            full_oofs = np.zeros((n_repeats, len(X), n_classes))
            all_folds = list(rskf.split(X, y))

            for r in range(n_repeats):
                for f in range(n_folds):
                    train_idx, test_idx = all_folds[r * n_folds + f]
                    m = clone(model)
                    m.fit(X[train_idx], y[train_idx])
                    full_oofs[r, test_idx] = m.predict_proba(X[test_idx])
            base_oofs_list.append(full_oofs)

        base_oofs = np.stack(base_oofs_list) # (B, N, S, C)

        # Bagging (Simple Average)
        bagging_oofs = np.mean(base_oofs, axis=(0, 1))
        acc_bag = accuracy_score(y, np.argmax(bagging_oofs, axis=1))
        if n_classes == 2:
            auc_bag = roc_auc_score(y, bagging_oofs[:, 1])
        else:
            auc_bag = roc_auc_score(y, bagging_oofs, multi_class='ovr')
        loss_bag = log_loss(y, bagging_oofs)

        results.append({
            'dataset': dataset_name,
            'model_set': model_set_name,
            'method': 'Bagging',
            'sync': sync,
            'avg': 'N/A',
            'accuracy': acc_bag,
            'auc': auc_bag,
            'log_loss': loss_bag
        })

        for avg in [True, False]:
            meta_seed = 42 if sync else 42 + len(base_models)

            if avg:
                avg_base_oofs = np.mean(base_oofs, axis=1) # (B, S, C)
                X_meta = avg_base_oofs.transpose(1, 0, 2).reshape(len(X), -1)

                rskf_meta = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_folds, random_state=meta_seed)
                meta_oofs_accum = np.zeros((len(X), n_classes))
                for train_idx, test_idx in rskf_meta.split(X_meta, y):
                    m = get_meta_model()
                    m.fit(X_meta[train_idx], y[train_idx])
                    meta_oofs_accum[test_idx] += m.predict_proba(X_meta[test_idx])
                meta_oofs_final = meta_oofs_accum / n_repeats
            else:
                meta_oofs_accum = np.zeros((len(X), n_classes))
                rskf_meta = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_folds, random_state=meta_seed)
                all_meta_folds = list(rskf_meta.split(X, y))

                for r in range(n_repeats):
                    X_meta_r = base_oofs[:, r, :, :].transpose(1, 0, 2).reshape(len(X), -1)
                    folds_r = all_meta_folds[r*n_folds : (r+1)*n_folds]

                    meta_oof_r = np.zeros((len(X), n_classes))
                    for train_idx, test_idx in folds_r:
                        m = get_meta_model()
                        m.fit(X_meta_r[train_idx], y[train_idx])
                        meta_oof_r[test_idx] = m.predict_proba(X_meta_r[test_idx])
                    meta_oofs_accum += meta_oof_r
                meta_oofs_final = meta_oofs_accum / n_repeats

            acc = accuracy_score(y, np.argmax(meta_oofs_final, axis=1))
            if n_classes == 2:
                auc = roc_auc_score(y, meta_oofs_final[:, 1])
            else:
                auc = roc_auc_score(y, meta_oofs_final, multi_class='ovr')
            loss = log_loss(y, meta_oofs_final)

            results.append({
                'dataset': dataset_name,
                'model_set': model_set_name,
                'method': 'Stacking',
                'sync': sync,
                'avg': avg,
                'accuracy': acc,
                'auc': auc,
                'log_loss': loss
            })

    return results

if __name__ == "__main__":
    datasets = ['breast_cancer', 'wine', 'digits', 'iris']
    all_results = []
    for model_set_name in MODEL_SETS.keys():
        print(f"Running with model set: {model_set_name}")
        for ds in datasets:
            print(f"  Dataset: {ds}")
            res = run_experiment(ds, model_set_name)
            all_results.extend(res)

    df = pd.DataFrame(all_results)
    df.to_csv('stacking_synchronization_experiment/results.csv', index=False)

    print("\nSummary (averaged across all datasets and model sets):")
    summary = df.groupby(['method', 'sync', 'avg'])[['accuracy', 'auc', 'log_loss']].mean()
    print(summary)

    with open('stacking_synchronization_experiment/summary.txt', 'w') as f:
        f.write("Summary (averaged across all datasets and model sets):\n")
        f.write(summary.to_string())
        f.write("\n\nDetailed summary by model set:\n")
        f.write(df.groupby(['model_set', 'method', 'sync', 'avg'])[['accuracy', 'auc', 'log_loss']].mean().to_string())
