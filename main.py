import matplotlib.pyplot as plt
import numpy as np
import tree
import data_loader

def get_metrics(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    d = data_loader.DataLoader()
    train_power = [power / 10 for power in range(6, 10)]
    metrics = {name:[] for name in ["accuracy", "precision", "recall", "f1"]}

    for power in train_power:
        X_train, y_train, X_test, y_test = d.get_train_test(power)
        model = tree.DecisionTreeClassifier(criterion="gini_gain", max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result = get_metrics(y_test, preds)
        for name in metrics:
            metrics[name].append(result[name])

    plt.subplot(2, 2, 1)
    plt.plot(train_power, metrics["accuracy"])
    plt.title("Accuracy")

    plt.subplot(2, 2, 2)
    plt.plot(train_power, metrics["precision"])
    plt.title("Precision")

    plt.subplot(2, 2, 3)
    plt.plot(train_power, metrics["recall"])
    plt.title("Recall")

    plt.subplot(2, 2, 4)
    plt.plot(train_power, metrics["f1"])
    plt.title("F1")

    plt.tight_layout()
    plt.show()