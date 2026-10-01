import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path_train="train.csv", path_test="test.csv"):
        self.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'income']
        self.dataset = self.load_dataset(path_train, path_test)
        self.feature_engineering()
        self.types = self.dataset.dtypes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self, path_train, path_test):
        train_file = pd.read_csv(path_train, names=self.columns, na_values='?', skipinitialspace=True)
        test_file = pd.read_csv(path_test, names=self.columns, na_values='?', skipinitialspace=True)
        test_file['income'] = test_file['income'].str.strip('.')
        dataset = pd.concat([train_file, test_file])
        dataset = dataset.dropna()
        return dataset

    def feature_engineering(self):
        self.dataset['income'] = self.dataset['income'].map({"<=50K": 0, ">50K": 1})
        self.dataset['sex'] = self.dataset['sex'].map({"Male": 0, "Female": 1})

    def get_dataset(self):
        return self.dataset

    def get_types(self):
        return self.types

    def get_train_test(self, ratio=0.7):
        train = self.dataset.sample(frac=ratio)
        test = self.dataset.drop(train.index)
        self.X_train = train.iloc[:, :-1]
        self.y_train = train.iloc[:, -1]
        self.X_test = test.iloc[:, :-1]
        self.y_test = test.iloc[:, -1]
        return self.X_train, self.y_train, self.X_test, np.asarray(self.y_test)




if __name__ == '__main__':
    d = DataLoader("train.csv", "test.csv")
    dataset = d.get_dataset()
    X_train, y_train, X_test, y_test = d.get_train_test(0.7)
    print(y_test)
