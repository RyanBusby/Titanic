import re

import pandas as pd

def get_data():
    train_df = pd.read_csv('data/train.csv')
    x, y = load_features(train_df)

    test_df = pd.read_csv('data/test.csv')
    test_df, passenger_ids = load_features(test_df, train=False)

    return x, y, test_df, passenger_ids

def cabin_mapping(x):
    if pd.isna(x):
        return 'X'
    cabin_class = re.findall(r'[A-Z]', x)[0]
    return cabin_class

def num_rooms(x):
    if pd.isna(x):
        return 0
    num_rooms = len(x.split(' '))
    return num_rooms

def load_features(df, train=True):
    df['cabin_class'] = df.Cabin.apply(cabin_mapping)
    df['num_rooms'] = df.Cabin.apply(num_rooms)
    df['fam_size'] = df.SibSp + df.Parch
    df['nicnamed'] = df.Name.apply(lambda x: 1 if '"' in x else 0)
    df['est_age'] = df.Age.apply(lambda x: 0 if x.is_integer() else 1)

    df.Fare = df.Fare.fillna(df.Fare.mean()) # there is a null value in the test data
    df.Age = df.Age.fillna(df.Age.mean())

    df = df.join(pd.get_dummies(df.Sex, drop_first=True))

    df = df.join(
        pd.get_dummies(
            df.Embarked, drop_first=True
        ).add_prefix('embark_')
    )

    df = df.join(
        pd.get_dummies(
            df.cabin_class, drop_first=True
        ).add_prefix('cabin_')
    )

    df['num_rooms_or_x'] = df.num_rooms + df.cabin_X

    use_feats = [
        'Pclass',
        'Age',
        'Fare',
        'fam_size',
        'nicnamed',
        'est_age',
        'embark_Q',
        'embark_S',
        'male',
        'cabin_B',
        'cabin_C',
        'cabin_D',
        'cabin_E',
        'cabin_F',
        'cabin_G',
        'num_rooms_or_x'
    ]
    x = df.loc[:,use_feats]

    if train:
        y = df.Survived.astype(int).values
        return x, y

    else:
        passenger_ids = df.PassengerId
        return x, passenger_ids
