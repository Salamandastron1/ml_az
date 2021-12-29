import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split   

def preProcess(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # imputer methods all for replacement of missing values
    # you want to ideally apply the fix to all numerical columns
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

def main():
    dataset = pd.read_csv('Data.csv')
    preProcess(dataset=dataset)

main()