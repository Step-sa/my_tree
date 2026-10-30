import numpy as np
import matplotlib.pyplot as plt

from data_loader import DataLoader
from tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaggingClassifier:
    def __init__(self, n_estimators=100, sample_ratio=1.0,
                 criterion="gini_gain", max_depth=3, min_samples_split=2,
                 random_state=42):

        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=int)
        sample_size = int(len(X) * self.sample_ratio)

        for i in range(self.n_estimators):
            indices = rng.choice(len(X), size=sample_size, replace=True)
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X.iloc[indices].reset_index(drop=True), y[indices])
            self.trees.append(tree)
        return self

    def predict(self, X, n_estimators=None):
        n = len(self.trees) if n_estimators is None else n_estimators
        votes = np.array([t.predict(X) for t in self.trees[:n]], dtype=int)
        return (votes.sum(axis=0) > n / 2).astype(int)


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def random_forest(X_train, y_train, X_test, y_test, n_estimators=50, max_depth=3):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return get_metrics(y_test, model.predict(X_test))


def gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=50, max_depth=3):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return get_metrics(y_test, model.predict(X_test))


def plot_results(n_estimators, results, tree_metrics):
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for method, values in results.items():
            ax.plot(n_estimators, values[metric], marker="o", label=method)

        ax.axhline(
            tree_metrics[metric],
            color="red",
            linestyle="--",
            label="Decision Tree"
        )

        ax.set_title(title)
        ax.set_xlabel("Number of estimators")
        ax.set_ylabel(title)
        ax.set_xticks(n_estimators)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    TRAIN_RATIO = 0.7
    N_ESTIMATORS = range(50, 101, 10)
    MAX_DEPTH = 3
    CRITERION = "gini_gain"

    # Загрузка данных
    dataset = DataLoader()
    X_train, y_train, X_test, y_test = dataset.get_train_test(TRAIN_RATIO)

    # Кодирование категориальных признаков для sklearn
    X_train_encoded = pd.get_dummies(X_train, dtype=int)
    X_test_encoded = pd.get_dummies(X_test, dtype=int)
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

    # Одиночное дерево
    single_tree = DecisionTreeClassifier(
        criterion=CRITERION, max_depth=MAX_DEPTH
    )
    single_tree.fit(X_train, y_train)
    tree_metrics = get_metrics(y_test, single_tree.predict(X_test))

    # Результаты ансамблей
    results = {
        method: {metric: [] for metric in tree_metrics}
        for method in ["Bagging", "Random Forest", "Gradient Boosting"]
    }

    # Обучение Bagging
    bagging = BaggingClassifier(
        n_estimators=max(N_ESTIMATORS),
        criterion=CRITERION,
        max_depth=MAX_DEPTH
    )
    bagging.fit(X_train, y_train)

    for n in N_ESTIMATORS:
        bagging_metrics = get_metrics(
            y_test, bagging.predict(X_test, n_estimators=n)
        )

        rf_metrics = random_forest(
            X_train_encoded, y_train,
            X_test_encoded, y_test,
            n_estimators=n, max_depth=MAX_DEPTH
        )

        gb_metrics = gradient_boosting(
            X_train_encoded, y_train,
            X_test_encoded, y_test,
            n_estimators=n, max_depth=MAX_DEPTH
        )

        for method, metrics in zip(
            results, [bagging_metrics, rf_metrics, gb_metrics]
        ):
            for metric, value in metrics.items():
                results[method][metric].append(value)

    plot_results(N_ESTIMATORS, results, tree_metrics)


if __name__ == "__main__":
    main()