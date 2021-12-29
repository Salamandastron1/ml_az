import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
    print(x)

def main():
    dataset = pd.read_csv('Data.csv')
    preProcess(dataset=dataset)

main()