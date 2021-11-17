import pandas as pd
import umap
import umap.plot
from sklearn.datasets import load_digits


if __name__ == '__main__':
    #%% Lets just see umap do its thing
    digits = load_digits()

    mapper = umap.UMAP().fit(digits.data)
    umap.plot.points(mapper, labels=digits.target)

    #%% md


    #%%
    database = pd.read_csv('SSID_Data.csv')
    database.info()

    #%% md


    #%%
    database.describe()
    #%% md


    #%%
    database.head(5)

