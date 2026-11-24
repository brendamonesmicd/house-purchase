"""
Entrenamiento: ¿el cliente comprará la casa?
Dataset: Global House Purchase Decision (Kaggle, mohankrishnathalla)

Split estratificado 70% entrenamiento / 10% validación / 20% prueba.
Se entrenan dos escenarios:
  - "con_visita": incluye satisfaction_score (interés del cliente tras ver la casa).
  - "sin_visita": sin satisfaction_score (antes de que el cliente vea la casa).
Cada modelo se registra como un run en Weights & Biases (proyecto house-purchase).

Uso:
    python train.py --data ruta/al/global_house_purchase_dataset.csv
    (WANDB_MODE=offline para entrenar sin conexión y luego `wandb sync wandb/offline-run-*`)
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, log_loss, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "decision"
CAT = ["country", "city", "property_type", "furnishing_status"]
NUM_ALL = [
    "property_size_sqft", "price", "constructed_year", "previous_owners", "rooms",
    "bathrooms", "garage", "garden", "crime_cases_reported", "legal_cases_on_property",
    "customer_salary", "loan_amount", "loan_tenure_years", "monthly_expenses",
    "down_payment", "emi_to_income_ratio", "satisfaction_score",
    "neighbourhood_rating", "connectivity_score",
]
SCENARIOS = {
    "con_visita": NUM_ALL,
    "sin_visita": [c for c in NUM_ALL if c != "satisfaction_score"],
}


def load(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df["city"] = df["city"].str.replace("�", "a", regex=False)  # "S�o Paulo" -> "Sao Paulo"
    df["city"] = df["city"].replace({"Sao Paulo": "São Paulo"})
    return df.drop(columns=["property_id"])


def split(df):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    # del 30% restante: 2/3 prueba (20% total) y 1/3 validación (10% total)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def preprocessor(num, scale):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])


def make_models(num):
    return {
        "logistic_regression": (Pipeline([("prep", preprocessor(num, True)),
                                          ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
                                {"C": 1.0}),
        "decision_tree": (Pipeline([("prep", preprocessor(num, False)),
                                    ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=SEED))]),
                          {"max_depth": 8, "min_samples_leaf": 50}),
        "random_forest": (Pipeline([("prep", preprocessor(num, False)),
                                    ("clf", RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                                                   n_jobs=-1, random_state=SEED))]),
                          {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 5}),
        "lightgbm": (None, {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31,
                            "early_stopping_rounds": 100}),
    }


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
    }


def plots(y, p, name, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title=f"ROC (prueba) – {name}", xlabel="FPR", ylabel="TPR")
    axes[0].legend()
    ConfusionMatrixDisplay(confusion_matrix(y, (p >= .5).astype(int)),
                           display_labels=["No compra", "Compra"]).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Matriz de confusión (prueba)")
    fig.tight_layout()
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(df, scenario, num, outdir, project, entity):
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(df)
    cols = CAT + num
    X_tr, X_val, X_te = X_tr[cols], X_val[cols], X_te[cols]
    results = {}
    for name, (model, params) in make_models(num).items():
        run_ = wandb.init(project=project, entity=entity, name=f"{scenario}-{name}",
                          group=scenario, job_type="train", reinit="finish_previous",
                          config={"model": name, "scenario": scenario, "n_features": len(cols),
                                  "split": "70/10/20 (train/val/test)", "seed": SEED,
                                  "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te), **params})
        t0 = time.time()
        if name == "lightgbm":
            Xtr_l, Xval_l, Xte_l = (d.astype({c: "category" for c in CAT}) for d in (X_tr, X_val, X_te))
            cats = Xtr_l[CAT].apply(lambda s: s.cat.categories)
            for d in (Xval_l, Xte_l):
                for c in CAT:
                    d[c] = pd.Categorical(d[c], categories=cats[c])
            model = lgb.LGBMClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                       num_leaves=params["num_leaves"], random_state=SEED, verbose=-1,
                                       metric=["binary_logloss", "auc"])
            hist = {}
            model.fit(Xtr_l, y_tr, eval_set=[(Xtr_l, y_tr), (Xval_l, y_val)], eval_names=["training", "val"],

                      callbacks=[lgb.early_stopping(params["early_stopping_rounds"], first_metric_only=True, verbose=False),
                                 lgb.record_evaluation(hist)])
            # curva de aprendizaje por iteración
            for i in range(len(hist["val"]["binary_logloss"])):
                wandb.log({"iter": i, "curve/train_logloss": hist["training"]["binary_logloss"][i],
                           "curve/val_logloss": hist["val"]["binary_logloss"][i],
                           "curve/val_auc": hist["val"]["auc"][i]})
            wandb.config.update({"best_iteration": model.best_iteration_})
            p_tr, p_val, p_te = (model.predict_proba(d)[:, 1] for d in (Xtr_l, Xval_l, Xte_l))
            fi = pd.Series(model.booster_.feature_importance("gain"), index=cols)
        else:
            model.fit(X_tr, y_tr)
            p_tr, p_val, p_te = (model.predict_proba(d)[:, 1] for d in (X_tr, X_val, X_te))
            clf = model.named_steps["clf"]
            names = model.named_steps["prep"].get_feature_names_out()
            imp = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
            fi = pd.Series(imp, index=[n.split("__", 1)[1] for n in names])
            fi = fi.groupby(lambda n: next((c for c in CAT if n.startswith(c + "_")), n)).sum()
        train_time = time.time() - t0

        # umbral que maximiza F1 en VALIDACIÓN (la clase positiva es ~23%)
        grid = np.linspace(0.05, 0.95, 91)
        thr = float(grid[np.argmax([f1_score(y_val, (p_val >= t).astype(int)) for t in grid])])
        m = {f"{split_}/{k}": v for split_, (y, p) in
             {"train": (y_tr, p_tr), "val": (y_val, p_val), "test": (y_te, p_te)}.items()
             for k, v in metrics(y, p).items()}
        m.update({f"test_tuned/{k}": v for k, v in metrics(y_te, p_te, thr).items() if k not in ("roc_auc", "log_loss")})
        m["best_threshold"] = thr
        m["train_time_s"] = train_time
        wandb.log(m)
        for k, v in m.items():
            wandb.summary[k] = v
        fi = (fi / fi.sum()).sort_values(ascending=False)
        wandb.log({
            "feature_importance": wandb.plot.bar(
                wandb.Table(data=[[k, float(v)] for k, v in fi.items()], columns=["feature", "importance"]),
                "feature", "importance", title=f"Importancia de variables – {scenario}/{name}"),
            "roc_test": wandb.plot.roc_curve(y_te.values, np.c_[1 - p_te, p_te], labels=["No compra", "Compra"]),
            "confusion_matrix_test": wandb.plot.confusion_matrix(y_true=y_te.values, preds=(p_te >= .5).astype(int),
                                                                 class_names=["No compra", "Compra"]),
            "roc_and_cm": wandb.Image(str(plots(y_te, p_te, f"{scenario}-{name}", outdir))),
        })
        run_.finish()
        results[name] = {"model": model, "metrics": m, "importance": fi}
        print(f"[{scenario}] {name:20s} val_auc={m['val/roc_auc']:.4f} test_acc={m['test/accuracy']:.4f} "
              f"test_f1={m['test/f1']:.4f} f1@{thr:.2f}={m['test_tuned/f1']:.4f} test_auc={m['test/roc_auc']:.4f} ({train_time:.1f}s)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default="house-purchase")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    root = Path(__file__).parent
    (root / "models").mkdir(exist_ok=True)
    figs = root / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    summary, meta = {}, {"categories": {c: sorted(df[c].unique().tolist()) for c in CAT},
                         "cities_by_country": df.groupby("country")["city"].unique().apply(sorted).to_dict(),
                         "numeric_ranges": {c: [float(df[c].min()), float(df[c].median()), float(df[c].max())]
                                            for c in NUM_ALL},
                         "base_rate": float(df[TARGET].mean()), "scenarios": {}}
    for scenario, num in SCENARIOS.items():
        res = run(df, scenario, num, figs, args.project, args.entity)
        # se elige el mejor modelo por AUC en VALIDACIÓN (la prueba solo se reporta)
        best = max(res, key=lambda k: (res[k]["metrics"]["val/roc_auc"], -res[k]["metrics"]["val/log_loss"]))
        joblib.dump(res[best]["model"], root / "models" / f"{scenario}.joblib", compress=3)
        meta["scenarios"][scenario] = {"best_model": best, "features": CAT + num,
                                       "test_metrics": {k: v for k, v in res[best]["metrics"].items() if k.startswith("test")},
                                       "threshold": res[best]["metrics"]["best_threshold"],
                                       "importance": res[best]["importance"].head(10).round(4).to_dict()}
        summary[scenario] = {k: {mk: round(mv, 4) for mk, mv in r["metrics"].items()} for k, r in res.items()}
        print(f"==> {scenario}: mejor modelo = {best}")

    (root / "models" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
