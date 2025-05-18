import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random


def generate_fake_titanic_data(n=500):
    data = []
    for i in range(n):
        pclass = random.choice([1, 2, 3])
        sex = random.choice(['male', 'female'])
        age = round(random.uniform(1, 80), 1)
        sibsp = random.randint(0, 5)
        parch = random.randint(0, 5)
        fare = round(random.uniform(10, 100), 2)
        survived = random.choice([0, 1])
        data.append([pclass, sex, age, sibsp, parch, fare, survived])
    
    df = pd.DataFrame(data, columns=[
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived'
    ])
    return df


def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

df = generate_fake_titanic_data(1000)  # Generate 1000 fake passenger records
X_train, X_test, y_train, y_test = preprocess_data(df)
train_and_evaluate(X_train, X_test, y_train, y_test)
