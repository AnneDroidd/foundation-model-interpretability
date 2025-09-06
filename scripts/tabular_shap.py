# scripts/tabular_shap.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def main():
    # load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # train small XGBoost
    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # save model
    joblib.dump(model, "results/xgb_model.joblib")

    # SHAP explain (TreeExplainer)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("figures/tabular_shap_summary.png", bbox_inches="tight")
    plt.close()

    # dependence plot for top feature
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = int(np.argmax(mean_abs))
    plt.figure(figsize=(6,4))
    shap.dependence_plot(top_idx, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("figures/tabular_shap_dependence.png", bbox_inches="tight")
    plt.close()

    # save a small csv with model performance
    pd.DataFrame({"metric":["accuracy"], "value":[acc]}).to_csv("results/metrics.csv", index=False)
    print("Saved: results/, figures/")

if __name__ == "__main__":
    main()