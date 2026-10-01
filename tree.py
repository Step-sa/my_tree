import numpy as np
import pandas as pd
import separation_math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, is_numeric=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.is_numeric = is_numeric
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self, criterion="gini_gain", max_depth=10, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        y_arr = np.array(y)
        self.root = self._build_tree(X, y_arr, current_depth=0)
        return self

    def _build_tree(self, X, y, current_depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (current_depth >= self.max_depth) or (n_samples < self.min_samples_split) or (n_classes <= 1):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        best_split = self._best_split(X, y)
        if best_split is None or best_split["gain"] <= 0:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feature = best_split["feature"]
        threshold = best_split["threshold"]
        is_numeric = best_split["is_numeric"]

        if is_numeric:
            mask = X[feature] <= threshold
        else:
            mask = X[feature] == threshold

        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]
        left_child = self._build_tree(X_left, y_left, current_depth + 1)
        right_child = self._build_tree(X_right, y_right, current_depth + 1)
        return Node(feature=feature, threshold=threshold, is_numeric=is_numeric, left=left_child, right=right_child,
                    gain=best_split["gain"])

    def _best_split(self, X, y):
        best_gain = -1
        best_split = None

        for col in X.columns:
            is_numeric = X[col].dtype != "str"
            thresholds = X[col].unique()
            if is_numeric:
                if len(thresholds) > 20:
                    thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 20))

            for thr in thresholds:
                if is_numeric:
                    mask = X[col] <= thr
                else:
                    mask = X[col] == thr

                y_left = y[mask]
                y_right = y[~mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = separation_math.gain(parent=y, left=y_left, right=y_right, criterion=self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {"feature": col, "threshold": thr, "is_numeric": is_numeric, "gain": gain}
        return best_split

    def predict(self, X):
        predictions = [self._predict_row(self.root, row) for _, row in X.iterrows()]
        return np.array(predictions)

    def _predict_row(self, node: Node, row: pd.Series):
        if node.is_leaf():
            return node.value

        val = row[node.feature]

        if node.is_numeric:
            go_left = val <= node.threshold
        else:
            go_left = val == node.threshold

        if go_left:
            return self._predict_row(node.left, row)
        else:
            return self._predict_row(node.right, row)

    def print_tree(self, filename="my_decision_tree.png"):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")

        def draw_node(node, x, y, dx, depth=0):
            if node is None:
                return

            if node.is_leaf():
                color = (
                    "#bbf7d0" if node.value == 1 else "#fed7aa"
                )
                income_label = ">50K" if node.value == 1 else "<=50K"
                text = f"Класс: {node.value}\nДоход: {income_label}"
                bbox = dict(
                    boxstyle="round,pad=0.5",
                    facecolor=color,
                    edgecolor="#4b5563",
                    lw=1.5,
                )
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    bbox=bbox,
                    fontsize=9,
                    weight="bold",
                )
                return

            op = "<=" if node.is_numeric else "=="
            text = f"{node.feature}\n{op} {node.threshold}"
            if node.gain is not None:
                text += f"\nGain: {node.gain:.2f}"

            bbox = dict(
                boxstyle="square,pad=0.5",
                facecolor="#e0f2fe",
                edgecolor="#1e40af",
                lw=1.5,
            )
            ax.text(x, y, text, ha="center", va="center", bbox=bbox, fontsize=9)

            y_next = y - 0.22
            x_left = x - dx
            x_right = x + dx

            if node.left:
                ax.annotate(
                    "",
                    xy=(x_left, y_next + 0.05),
                    xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"),
                )
                ax.text(
                    (x + x_left) / 2 - 0.02,
                    (y + y_next) / 2,
                    "True",
                    color="#16a34a",
                    fontsize=8,
                    weight="bold",
                )
                draw_node(node.left, x_left, y_next, dx / 2, depth + 1)
            if node.right:
                ax.annotate(
                    "",
                    xy=(x_right, y_next + 0.05),
                    xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#dc2626"),
                )
                ax.text(
                    (x + x_right) / 2 + 0.02,
                    (y + y_next) / 2,
                    "False",
                    color="#dc2626",
                    fontsize=8,
                    weight="bold",
                )
                draw_node(node.right, x_right, y_next, dx / 2, depth + 1)

        draw_node(self.root, x=0.5, y=0.95, dx=0.25)
        plt.title("Визуализация дерева решений", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Дерево сохранено в: {filename}")

if __name__ == "__main__":
    import data_loader
    d = data_loader.DataLoader()
    X_train, y_train, X_test, y_test = d.get_train_test(0.8)
    # "gini_gain" "information_gain""gain_ratio"
    model = DecisionTreeClassifier(criterion="gini_gain", max_depth=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.print_tree("my_decision_tree")
