
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from data_prep import get_data

def train_with_partial(fname):
    x, y, test_df, passenger_ids = get_data()
    X_train, X_test, y_train, y_test =\
    train_test_split(x, y, random_state=0)

    model = XGBClassifier(
        n_estimators=10, max_depth=15, tree_method='exact'
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(test_df)

    pd.DataFrame(
        zip(passenger_ids.values, test_predictions),
        columns=['PassengerId', 'Survived']
    )\
    .astype(int)\
    .to_csv(fname, index=False)

def train(fname):
    x, y, test_df, passenger_ids = get_data()

    model = XGBClassifier(
        n_estimators=10, max_depth=15, tree_method='exact'
    )
    model.fit(x, y)

    test_predictions = model.predict(test_df)

    pd.DataFrame(
        zip(passenger_ids.values, test_predictions),
        columns=['PassengerId', 'Survived']
    )\
    .astype(int)\
    .to_csv(fname, index=False)

if __name__ == "__main__":
    train_with_partial('data/titanic_predictions_partial.csv')
    train('data/titanic_predictions.csv')
