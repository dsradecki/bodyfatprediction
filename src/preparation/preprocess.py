import pandas as pd
from sklearn.decomposition import PCA


def read_data():

    return pd.read_csv('../data/bodyfat.csv')


def normalise(data):

    return data.apply(lambda x: (x - min(x))/(max(x) - min(x)), axis=0)


def perform_pca(data):

    pca = PCA(n_components=1)
    pca.fit(data)
    return pd.DataFrame(pca.transform(data))




