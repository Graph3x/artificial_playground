import pandas as pd
from fastapi import FastAPI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

train_path = "train.csv"
test_path = "test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


def title(dataset):
    dataset['Title'] = dataset['Name'].apply(
        lambda x: x.split(', ')[1].split(' ')[0])

    for i, row in dataset.iterrows():
        match row['Title']:
            case 'Don.':
                dataset.at[i, 'Title'] = 'Master.'
            case 'Mme.':
                dataset.at[i, 'Title'] = 'Mrs.'
            case 'Ms.':
                dataset.at[i, 'Title'] = 'Miss.'
            case 'Major.':
                dataset.at[i, 'Title'] = 'Master.'
            case 'Lady.':
                dataset.at[i, 'Title'] = 'Mrs.'
            case 'Sir.':
                dataset.at[i, 'Title'] = 'Master.'
            case 'Mlle.':
                dataset.at[i, 'Title'] = 'Miss.'
            case 'Col.':
                dataset.at[i, 'Title'] = 'Master.'
            case 'Capt.':
                dataset.at[i, 'Title'] = 'Master.'
            case 'the':
                dataset.at[i, 'Title'] = 'Mr.'
            case 'Jonkheer.':
                dataset.at[i, 'Title'] = 'Mrs.'


def family_members(dataset):
    dataset['FamilyMembers'] = dataset['Parch'] + dataset["SibSp"]


title(train_data)
family_members(train_data)
title(test_data)
family_members(test_data)


feats = ['Pclass', 'Title', 'FamilyMembers', 'Fare', 'Age']

y = train_data['Survived']
X = train_data[feats]


numerical_feats = ['Fare', 'Age', 'FamilyMembers']
categorical_feats = ['Pclass', 'Title', 'Fare']

numerical_imp = SimpleImputer(strategy='median')
categorical_imp = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_imp, numerical_feats),
    ('cat', categorical_imp, categorical_feats)
])

main_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=1, max_depth=5, n_estimators=500))
])

# REMOVE FOR NON DEPLOYMENT USES
main_pipeline.fit(X, y)


def train():
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    main_pipeline.fit(train_X, train_y)

    preds = main_pipeline.predict(val_X)

    mae = mean_absolute_error(preds, val_y)

    print(1 - mae)


def compete():
    main_pipeline.fit(X, y)

    preds = main_pipeline.predict(test_data[feats])
    output = pd.DataFrame(
        {'PassengerId': test_data.PassengerId, 'Survived': preds})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")


def predict(p_class, title, family, fare, age):
    data = pd.DataFrame({'Pclass': [p_class], 'Title': [title],
                         'FamilyMembers': [family], 'Fare': [fare], 'Age': [age]})
    return main_pipeline.predict(data)


app = FastAPI()


@app.post("/")
async def root(p_class: str, title: str, family: int, fare: int, age: int):
    data = predict(p_class, title, family, fare, age)
    return str(data[0])
