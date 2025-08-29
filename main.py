import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import kagglehub

os.makedirs("assets", exist_ok=True)

def find_csv(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV found in {path}")

# Download datasets
heart_path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
heart_csv = find_csv(heart_path)
df_heart = pd.read_csv(heart_csv)

diabetes_path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
diabetes_csv = find_csv(diabetes_path)
df_diabetes = pd.read_csv(diabetes_csv)

# ✅ Correct target columns
datasets = {
    "Heart Disease": (df_heart, "num"),       # 'num' is target
    "Diabetes": (df_diabetes, "Outcome")      # 'Outcome' is target
}

def run_pipeline(name, df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # For Heart dataset: make binary (0 = no disease, 1+ = disease)
    if name == "Heart Disease":
        y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    metrics_list = []

    # ROC curve plot holder
    plt.figure(figsize=(5,5))

    for mname, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": mname,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        }
        metrics_list.append(metrics)

        # Confusion Matrix (smaller + tidy)
        cm = confusion_matrix(y_test, y_pred)
        plt_cm = plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} - {mname}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"assets/{name.replace(' ','_')}_{mname.replace(' ','_')}_cm.png")
        plt.close(plt_cm)

        # Add ROC curve for this model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{mname} (AUC={metrics['ROC-AUC']})")

    # Finalize ROC curve (one per dataset)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title(f"{name} - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"assets/{name.replace(' ','_')}_roc.png")
    plt.close()

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"assets/{name.replace(' ','_')}_metrics.csv", index=False)
    return metrics_df

# Run both datasets
results = {}
for name, (df, target) in datasets.items():
    print(f"\n{name} Dataset:")
    print(df.head(2))
    results[name] = run_pipeline(name, df, target)

for dataset, df in results.items():
    print(f"\nResults for {dataset}")
    print(df)
