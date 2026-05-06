# Heart Disease Classification
# Hyperparameter Tuning with 60/10/30 Split
#
# 60% Train: used to train models during tuning
# 10% Eval:  used as pseudo-test set for hyperparameter tuning
# 30% Test:  final untouched test set
#
# Models:
#   1. Logistic Regression
#   2. Support Vector Machine
#   3. Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


#1. Load the Excel Sheets
file_name = "heart_disease_combined_cleaned.xlsx"

train_df = pd.read_excel(file_name, sheet_name="Train_Scaled")
eval_df = pd.read_excel(file_name, sheet_name="Eval_Scaled")
test_df = pd.read_excel(file_name, sheet_name="Test_Scaled")

print("Data Loaded Successfully")
print("-" * 50)
print(f"Training set:   {train_df.shape[0]} samples, {train_df.shape[1] - 1} features")
print(f"Evaluation set: {eval_df.shape[0]} samples, {eval_df.shape[1] - 1} features")
print(f"Testing set:    {test_df.shape[0]} samples, {test_df.shape[1] - 1} features")

print("\nTraining target counts:")
print(train_df["target"].value_counts())

print("\nEvaluation target counts:")
print(eval_df["target"].value_counts())

print("\nTesting target counts:")
print(test_df["target"].value_counts())



#2. Separate Features and Target
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_eval = eval_df.drop(columns=["target"])
y_eval = eval_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]


#3. Evaluation Helper Functions
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y, y_prob)
    }

    return metrics, y_pred, y_prob


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<12}: {value:.4f}")


#4. Grid Search: Logistic Regression
logistic_param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "solver": ["liblinear", "lbfgs"]
}

logistic_results = []
best_log_model = None
best_log_params = None
best_log_accuracy = -1

print("\n\n===== LOGISTIC REGRESSION GRID SEARCH =====")

for C, solver in product(logistic_param_grid["C"], logistic_param_grid["solver"]):

    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    eval_metrics, _, _ = calculate_metrics(model, X_eval, y_eval)

    result = {
        "Model": "Logistic Regression",
        "C": C,
        "solver": solver,
        "Eval Accuracy": eval_metrics["Accuracy"],
        "Eval Precision": eval_metrics["Precision"],
        "Eval Recall": eval_metrics["Recall"],
        "Eval F1 Score": eval_metrics["F1 Score"],
        "Eval AUROC": eval_metrics["AUROC"]
    }

    logistic_results.append(result)

    print(
        f"C={C:<8} solver={solver:<10} "
        f"Eval Accuracy={eval_metrics['Accuracy']:.4f} "
        f"Eval F1={eval_metrics['F1 Score']:.4f} "
        f"Eval AUROC={eval_metrics['AUROC']:.4f}"
    )

    if eval_metrics["Accuracy"] > best_log_accuracy:
        best_log_accuracy = eval_metrics["Accuracy"]
        best_log_params = {
            "C": C,
            "solver": solver
        }
        best_log_model = model

print("\nBest Logistic Regression Parameters:")
print(best_log_params)
print(f"Best Logistic Regression Eval Accuracy: {best_log_accuracy:.4f}")


#5. Grid Search: SVM
svm_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm_results = []
best_svm_model = None
best_svm_params = None
best_svm_accuracy = -1

print("\n\n===== SVM GRID SEARCH =====")

for C, kernel, gamma in product(
    svm_param_grid["C"],
    svm_param_grid["kernel"],
    svm_param_grid["gamma"]
):

    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    eval_metrics, _, _ = calculate_metrics(model, X_eval, y_eval)

    result = {
        "Model": "SVM",
        "C": C,
        "kernel": kernel,
        "gamma": gamma,
        "Eval Accuracy": eval_metrics["Accuracy"],
        "Eval Precision": eval_metrics["Precision"],
        "Eval Recall": eval_metrics["Recall"],
        "Eval F1 Score": eval_metrics["F1 Score"],
        "Eval AUROC": eval_metrics["AUROC"]
    }

    svm_results.append(result)

    print(
        f"C={C:<6} kernel={kernel:<8} gamma={gamma:<6} "
        f"Eval Accuracy={eval_metrics['Accuracy']:.4f} "
        f"Eval F1={eval_metrics['F1 Score']:.4f} "
        f"Eval AUROC={eval_metrics['AUROC']:.4f}"
    )

    if eval_metrics["Accuracy"] > best_svm_accuracy:
        best_svm_accuracy = eval_metrics["Accuracy"]
        best_svm_params = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma
        }
        best_svm_model = model

print("\nBest SVM Parameters:")
print(best_svm_params)
print(f"Best SVM Eval Accuracy: {best_svm_accuracy:.4f}")



#6. Grid Search: Random Forest
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_results = []
best_rf_model = None
best_rf_params = None
best_rf_accuracy = -1

print("\n\n===== RANDOM FOREST GRID SEARCH =====")

for n_estimators, max_depth, min_samples_split, min_samples_leaf in product(
    rf_param_grid["n_estimators"],
    rf_param_grid["max_depth"],
    rf_param_grid["min_samples_split"],
    rf_param_grid["min_samples_leaf"]
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    model.fit(X_train, y_train)

    eval_metrics, _, _ = calculate_metrics(model, X_eval, y_eval)

    result = {
        "Model": "Random Forest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "Eval Accuracy": eval_metrics["Accuracy"],
        "Eval Precision": eval_metrics["Precision"],
        "Eval Recall": eval_metrics["Recall"],
        "Eval F1 Score": eval_metrics["F1 Score"],
        "Eval AUROC": eval_metrics["AUROC"]
    }

    rf_results.append(result)

    print(
        f"n={n_estimators:<4} depth={str(max_depth):<5} "
        f"split={min_samples_split:<3} leaf={min_samples_leaf:<3} "
        f"Eval Accuracy={eval_metrics['Accuracy']:.4f} "
        f"Eval F1={eval_metrics['F1 Score']:.4f} "
        f"Eval AUROC={eval_metrics['AUROC']:.4f}"
    )

    if eval_metrics["Accuracy"] > best_rf_accuracy:
        best_rf_accuracy = eval_metrics["Accuracy"]
        best_rf_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        best_rf_model = model

print("\nBest Random Forest Parameters:")
print(best_rf_params)
print(f"Best Random Forest Eval Accuracy: {best_rf_accuracy:.4f}")



#7. Combine and Save All Grid Search Results
logistic_results_df = pd.DataFrame(logistic_results)
svm_results_df = pd.DataFrame(svm_results)
rf_results_df = pd.DataFrame(rf_results)

all_tuning_results = pd.concat(
    [logistic_results_df, svm_results_df, rf_results_df],
    ignore_index=True
)

all_tuning_results.to_csv("hyperparameter_tuning_results.csv", index=False)

print("\nHyperparameter tuning results saved to hyperparameter_tuning_results.csv")



#8. Print Top 5 Hyperparameter Results Per Model
print("\n\n===== TOP 5 HYPERPARAMETER RESULTS PER MODEL =====")

for model_name in all_tuning_results["Model"].unique():
    print(f"\n{model_name}")
    print("-" * 50)

    top_results = (
        all_tuning_results[all_tuning_results["Model"] == model_name]
        .sort_values(by="Eval Accuracy", ascending=False)
        .head(5)
    )

    print(top_results.to_string(index=False))

#Save top 5 results per model to a separate CSV
top_5_results = (
    all_tuning_results
    .sort_values(["Model", "Eval Accuracy"], ascending=[True, False])
    .groupby("Model")
    .head(5)
)

top_5_results.to_csv("top_5_hyperparameter_results_per_model.csv", index=False)

print("\nTop 5 hyperparameter results saved to top_5_hyperparameter_results_per_model.csv")



#9. Evaluate Best Models on Final 30% Test Set
best_models = {
    "Logistic Regression": best_log_model,
    "SVM": best_svm_model,
    "Random Forest": best_rf_model
}

final_results = {}

print("\n\n===== FINAL TEST SET RESULTS =====")

for model_name, model in best_models.items():

    test_metrics, y_pred, y_prob = calculate_metrics(model, X_test, y_test)
    final_results[model_name] = test_metrics

    print_metrics(model_name, test_metrics)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

final_results_df = pd.DataFrame(final_results).T.round(4)
final_results_df.to_csv("final_test_results.csv")

print("\nFinal test results saved to final_test_results.csv")

print("\nFinal Test Comparison Table:")
print(final_results_df.to_string())



#10. Bar Chart: Final Model Comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"]
x = np.arange(len(metrics))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, (model_name, scores) in enumerate(final_results.items()):
    values = [scores[m] for m in metrics]
    ax.bar(x + i * bar_width, values, bar_width, label=model_name)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Final Test Set Model Comparison")
ax.set_xticks(x + bar_width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("final_model_comparison.png", dpi=150)
plt.show()

print("Final model comparison chart saved as final_model_comparison.png")



#11. Confusion Matrices for Best Models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (model_name, model) in zip(axes, best_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )

    disp.plot(ax=ax, colorbar=False)
    ax.set_title(model_name)

plt.suptitle("Final Test Set Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("final_confusion_matrices.png", dpi=150)
plt.show()

print("Final confusion matrices saved as final_confusion_matrices.png")



#12. Random Forest Feature Importance
rf_importances = pd.Series(
    best_rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

rf_importances.to_csv("random_forest_feature_importance.csv")

plt.figure(figsize=(10, 5))
rf_importances.plot(kind="bar")
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=150)
plt.show()

print("Random Forest feature importance saved as random_forest_feature_importance.png")



#13. Summary of Best Hyperparameters
print("\n\n===== BEST HYPERPARAMETER SUMMARY =====")

print("Logistic Regression:")
print(best_log_params)
print(f"Best Eval Accuracy: {best_log_accuracy:.4f}")

print("\nSVM:")
print(best_svm_params)
print(f"Best Eval Accuracy: {best_svm_accuracy:.4f}")

print("\nRandom Forest:")
print(best_rf_params)
print(f"Best Eval Accuracy: {best_rf_accuracy:.4f}")

print("\nDone!")



# 14. Extra Figure 1
import matplotlib.pyplot as plt
import numpy as np

LR_COL = "#2E86AB"
ACCENT = "#1A5276"
SVM_COL = "#E84855"
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_unique  = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
auc_lib   = [0.8694, 0.8881, 0.8924, 0.8967, 0.8996, 0.8996, 0.9005]
auc_lbfgs = [0.8886, 0.8881, 0.8895, 0.8953, 0.8986, 0.9005, 0.9005]

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(C_unique, auc_lib,  "o-", color=LR_COL,  linewidth=2, markersize=6, label="liblinear")
ax.semilogx(C_unique, auc_lbfgs,"s--",color=ACCENT,  linewidth=2, markersize=6, label="lbfgs")
ax.axvline(0.1, color=SVM_COL, linewidth=1.5, linestyle=":", alpha=0.8, label="Best C = 0.1")
ax.set_xlabel("Regularisation Strength C (log scale)", fontsize=10)
ax.set_ylabel("Eval AUROC", fontsize=10)
ax.set_title(" LR Hyperparameter Tuning: AUROC vs C", fontsize=11,
             color=ACCENT, fontweight="bold", pad=10)
ax.legend(fontsize=9, framealpha=0.5)
ax.set_ylim(0.855, 0.91)
fig.tight_layout()
fig.savefig("fig5_lr_tuning.png", dpi=180, bbox_inches="tight")

#15. Extra Figure 2
import matplotlib.pyplot as plt
import numpy as np

LR_COL  = "#2E86AB"
SVM_COL = "#E84855"
RF_COL  = "#3BB273"
ACCENT  = "#1A5276"
GREY    = "#888888"
BG      = "#FAFAFA"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "figure.facecolor": "white",
})

final = {
    "LR":  {"Accuracy": 0.8080, "Precision": 0.8378, "Recall": 0.8105, "F1": 0.8239, "AUROC": 0.8897},
    "SVM": {"Accuracy": 0.8188, "Precision": 0.8280, "Recall": 0.8497, "F1": 0.8387, "AUROC": 0.8988},
    "RF":  {"Accuracy": 0.8152, "Precision": 0.8312, "Recall": 0.8366, "F1": 0.8339, "AUROC": 0.8945},
}

cats = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]
N = len(cats)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

def model_vals(key):
    v = [final[key][m] for m in cats]
    return v + v[:1]

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.set_facecolor(BG)

for key, col, lbl in zip(["LR", "SVM", "RF"], [LR_COL, SVM_COL, RF_COL], ["LR", "SVM", "RF"]):
    vals = model_vals(key)
    ax.plot(angles, vals, "o-", linewidth=2, color=col, label=lbl)
    ax.fill(angles, vals, alpha=0.08, color=col)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylim(0.75, 0.92)
ax.set_yticks([0.78, 0.82, 0.86, 0.90])
ax.set_yticklabels(["0.78", "0.82", "0.86", "0.90"], fontsize=7, color=GREY)
ax.set_title(" Radar Chart — Test Set Metrics", fontsize=11,
             color=ACCENT, fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
fig.tight_layout()
fig.savefig("fig7_radar.png", dpi=180, bbox_inches="tight")