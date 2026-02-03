# gbdt_lr_pipeline_full.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from scipy import sparse


def generate_data(n_samples=100000, n_features=30, n_informative=10, random_state=42):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=random_state)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="label")


def train_gbdt(X_train, y_train, X_val, y_val, num_boost_round=200):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )
    return booster

# 3. GBDT -> leaf one-hot
def gbdt_leaf_onehot_transform(booster, X, ohe=None, concat_original=True, scaler=None):

    leaf_indices = booster.predict(X, pred_leaf=True)
    # print(f'leaf_indices.shape:{leaf_indices.shape}')
    # print(f'leaf_indices:{leaf_indices}')
    # OneHotEncoder
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        leaf_ohe = ohe.fit_transform(leaf_indices)
    else:
        leaf_ohe = ohe.transform(leaf_indices)
    # print(f'leaf_ohe:{leaf_ohe}')

    if concat_original:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        # print(f'X_scaled:{X_scaled}')
        X_sparse = sparse.csr_matrix(X_scaled)
        # print(f'X_sparse:{X_sparse}')
        X_final = sparse.hstack([leaf_ohe, X_sparse], format="csr")
        # print(f'X_final:{X_final}')
        # print(f'X_final[0,6230]:{X_final[0,6199:6229]}')
        return X_final, ohe, scaler
    else:
        return leaf_ohe, ohe, scaler
    


def train_lr(X_sparse_train, y_train, C=1.0, max_iter=300):
    lr = LogisticRegression(penalty="l2", C=C, solver="saga", max_iter=max_iter, tol=1e-4, n_jobs=-1)
    lr.fit(X_sparse_train, y_train)
    return lr


def evaluate(model, X_sparse, y_true):
    y_pred_proba = model.predict_proba(X_sparse)[:, 1]
    auc = roc_auc_score(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)
    return {"auc": auc, "logloss": ll}


if __name__ == "__main__":

    X, y = generate_data(n_samples=100000, n_features=30, n_informative=10)
    print(f'x:{X}')
    print(f'y:{y}')
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'X_train_full:{X_train_full}')
    print(f'y_train_full:{y_train_full}')
    print(f'X_test:{X_test}')
    print(f'y_test:{y_test}')
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f'X_train:{X_train}')
    print(f'y_train:{y_train}')
    print(f'X_val:{X_val}')
    print(f'y_val:{y_val}')

    print("Training LightGBM GBDT ...")
    booster = train_gbdt(X_train.values, y_train.values, X_val.values, y_val.values)

    print("Transforming train set...")
    X_train_sparse, ohe, scaler = gbdt_leaf_onehot_transform(booster, X_train.values,
                                                             concat_original=True)

    print("Transforming validation and test sets...")
    X_val_sparse, _, _ = gbdt_leaf_onehot_transform(booster, X_val.values,
                                                    ohe=ohe, concat_original=True, scaler=scaler)
    X_test_sparse, _, _ = gbdt_leaf_onehot_transform(booster, X_test.values,
                                                     ohe=ohe, concat_original=True, scaler=scaler)

    print("Sparse feature shape:", X_train_sparse.shape)

    print("Training Logistic Regression ...")
    lr = train_lr(X_train_sparse, y_train.values, C=0.5, max_iter=300)

    val_metrics = evaluate(lr, X_val_sparse, y_val.values)
    print(f"Validation AUC: {val_metrics['auc']:.5f}, LogLoss: {val_metrics['logloss']:.5f}")

    test_metrics = evaluate(lr, X_test_sparse, y_test.values)
    print(f"Test AUC: {test_metrics['auc']:.5f}, LogLoss: {test_metrics['logloss']:.5f}")

    fi = booster.feature_importance(importance_type="gain")
    names = X.columns.tolist()
    imp_df = pd.DataFrame({"feature": names, "gain": fi}).sort_values("gain", ascending=False).head(10)
    print("Top 10 GBDT features by gain:")
    print(imp_df.to_string(index=False))